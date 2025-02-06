from typing import Optional
import torch
import warnings
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from boludo.utils import instantiate_class


class BotorchOptimizer:
    def __init__(
        self,
        design_space=None,
        surrogate_model_config=None,
        acq_function_config=None,
        train_x=None,
        train_y=None,
        batch_strategy="kriging",
        fixed_features=None,
        tkwargs={"device": torch.device("cpu"), "dtype": torch.double},
    ):
        self.design_space = design_space or self.define_design_space(train_x)
        self.surrogate_model_config = (
            surrogate_model_config or BotorchOptimizer.default_surrogate_model_config()
        )
        self.acq_function_config = (
            acq_function_config or BotorchOptimizer.default_acq_function_config()
        )

        self.pending_x = []

        self.train_x = None
        self.train_y = None
        self.batch_strategy = batch_strategy
        self.fixed_features = fixed_features
        self.update_acquisition_function_params()

        self.tell(train_x, train_y)
        self.tkwargs = tkwargs

    def define_design_space(
        self, train_x: Optional[torch.Tensor] = None, raw_samples: int = 10000
    ) -> torch.Tensor:
        if train_x is not None:
            lower_bounds = train_x.amin(dim=0)
            upper_bounds = train_x.amax(dim=0)
            expansion_factor = 0.2
            lower_bounds = lower_bounds - expansion_factor * (upper_bounds - lower_bounds)
            upper_bounds = upper_bounds + expansion_factor * (upper_bounds - lower_bounds)
            bounds = torch.stack([lower_bounds, upper_bounds])
            return bounds

        else:
            # TODO: define design space based on the problem and user input
            print(
                "Please define the design space bounds (lower and upper bounds for each dimension):"
            )
            lower_bounds = torch.tensor([0.0, 0.0])
            upper_bounds = torch.tensor([1.0, 1.0])
            bounds = torch.stack([lower_bounds, upper_bounds])
            # TODO maybe use optimize_acqf to generate the heldout set instead of sobol samples
            self.heldout_x = (
                draw_sobol_samples(bounds=bounds, n=raw_samples, q=1).to(**self.tkwargs).squeeze()
            )
        return self.heldout_x

    def lie_to_me(self, candidate, train_y, strategy="kriging"):
        supported_strategies = ["cl_min", "cl_mean", "cl_max", "kriging"]
        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of "
                + str(supported_strategies)
                + ", "
                + "got %s" % strategy
            )

        if strategy == "cl_min":
            y_lie = torch.min(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
        elif strategy == "cl_mean":
            y_lie = torch.mean(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
        elif strategy == "cl_max":
            y_lie = torch.max(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
        else:
            y_lie, _ = self.surrogate_model.predict(candidate)
        return y_lie

    def ask(self, n_points=1, strategy="cl_min"):
        if self.surrogate_model is None:
            return self.pending_x

        candidates = []
        if strategy == "parallel":

            candidate, _ = self.next_evaluations(
                self.acq_function,
                self.design_space,
                num_restarts=100,
                raw_samples=300000,
                q=n_points,
            )
            candidates.append(candidate)
        else:
            for i in range(n_points):
                candidate, _ = self.next_evaluations(
                    self.acq_function,
                    self.design_space,
                    num_restarts=50,
                    raw_samples=100000,
                    q=1,
                )
                candidates.append(candidate)
                y_lie = self.lie_to_me(candidate, self.train_y, strategy=self.batch_strategy)
                self.tell(candidate, y_lie, lie=True)

            self.train_x = self.train_x[:-n_points]
            self.train_y = self.train_y[:-n_points]

        print(self.train_x.shape, "train_x")
        self.pending_x.extend(candidates)
        return candidates

    def tell(self, x, y, lie=False):
        self.train_x = torch.cat([self.train_x, x]) if self.train_x is not None else x
        self.train_y = torch.cat([self.train_y, y]) if self.train_y is not None else y

        # if self.pending_x and not lie:
        #     self.pending_x.remove(x)
        # would have to be changed because the experiment might not be exactly the same

        for point in x:
            indices_to_remove = []
            for i, pending_point in enumerate(self.pending_x):
                if torch.all(torch.eq(pending_point, point)):
                    indices_to_remove.append(i)

            for i in sorted(indices_to_remove, reverse=True):
                del self.pending_x[i]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.surrogate_model = instantiate_class(
                self.surrogate_model_config, train_x=self.train_x, train_y=self.train_y
            )
            self.surrogate_model.fit(self.train_x, self.train_y)
            self.acq_function = instantiate_class(
                self.acq_function_config, model=self.surrogate_model
            )

    def next_evaluations(
        self,
        acq_function,
        bounds,
        num_restarts,
        raw_samples,
        categorical_mask=None,
        q=1,
    ):
        # sample a large number of points
        X = draw_sobol_samples(bounds=bounds, n=raw_samples, q=q).to(**self.tkwargs)

        if categorical_mask is not None:
            X[..., categorical_mask] = X[..., categorical_mask].round()

        acq_values = acq_function(X).squeeze()
        best_points = X[acq_values.topk(num_restarts)[1]]

        candidate, acq_val = optimize_acqf(
            acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            batch_initial_conditions=best_points,  # start optimization from the best sampled points
            fixed_features=self.fixed_features,
        )

        return candidate, acq_val

    @staticmethod
    def default_surrogate_model_config():
        # default surrogate model config
        return {
            "class_path": "boludo.surrogate_models.gp.SimpleGP",
            "init_args": {
                "covar_module": {
                    "class_path": "gpytorch.kernels.ScaleKernel",
                    "init_args": {
                        "base_kernel": {
                            "class_path": "gpytorch.kernels.MaternKernel",
                            "init_args": {"nu": 2.5},
                        }
                    },
                },
                "likelihood": {
                    "class_path": "gpytorch.likelihoods.GaussianLikelihood",
                    "init_args": {"noise": 1e-4},
                },
                "normalize": False,
                "standardize": True,
                "initial_noise_val": 1.0e-4,
                "noise_constraint": 1.0e-05,
                "initial_outputscale_val": 2.0,
                "initial_lengthscale_val": 0.5,
            },
        }

    @staticmethod
    def default_acq_function_config():
        # default acquisition function config
        return {
            "class_path": "botorch.acquisition.qExpectedImprovement",
            "init_args": {"best_f": 0},
        }

    def update_acquisition_function_params(self):
        params = {}
        if "ExpectedImprovement" in self.acq_function_config["class_path"]:
            params["X_baseline"] = self.train_x

        return params
