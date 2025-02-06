from typing import Optional
import ast
import pandas as pd
import torch

from bochemian.data.module import BaseDataModule
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from boludo.data.utils import (
    return_singular,
    apply_filters,
)
import time
import pandas as pd
import numpy as np
from cheminfopy import User
from cheminfopy import Experiment

pd.set_option("future.no_silent_downcasting", True)


def create_reagent_mapping(reagent_list):
    reagent_mapping = {}
    for reagent in reagent_list:
        reagent_mapping[reagent["iupac"].lower()] = {
            "mw": reagent["mw"],
            "density": reagent["density"],
        }
    return reagent_mapping


class ELNConnector:
    def __init__(
        self,
        instance: str = "https://eln-beta.epfl.ch/roc/",
        token: str = "your token here",
        include_hidden=False,
        sort_by="$creationDate",
        filters=["Electrochemistry", "Nickel"],
    ) -> None:
        self.instance = instance
        self.token = token

        for _ in range(10):
            try:
                user = User(instance=instance, token=token)
                experiment_uuids = [ex["id"] for ex in user.get_experiment_toc()]
                experiments = [
                    Experiment(
                        instance=instance,
                        experiment_uuid=uuid,
                        token=token,
                    )
                    for uuid in experiment_uuids
                ]

                self.data = pd.json_normalize([experiment.toc for experiment in experiments])
                if not include_hidden:
                    self.data = self.data[self.data["$content.hidden"] != True]
                if sort_by:
                    self.data.sort_values(sort_by, ascending=False, inplace=True)
                self.data.rename(
                    columns={
                        col: col.lower().replace("$content.", "").replace("meta.", "").strip()
                        for col in self.data.columns
                    },
                    inplace=True,
                )
                self.data = self.data[~self.data["$id"].str.contains("NM-KINETIC-002|LZ-PA")]
                for _filter in filters:
                    self.data = apply_filters(self.data, _filter)

            except ConnectionError:
                print("Connection error, retrying...")
                time.sleep(1)
            else:
                break


def get_scales(shape_tuple, mapping):
    try:
        shapes = eval(shape_tuple)
    except:
        shapes = shape_tuple
    if isinstance(shapes, str):
        shapes = shapes
    if isinstance(shapes, list):
        if len(shapes) == 1:
            return pd.Series({f"target_{shapes[0]}": mapping.get(shapes[0], np.nan)})
        else:
            return pd.Series({f"target_{shape}": mapping.get(shape, np.nan) for shape in shapes})

    return pd.Series({"target_unknown": np.nan})


class BOParticlesScalesDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: str = "../data/processed/scale_1_data.csv",
        target_column: str = "target",
        target: float = 20832.0,
        eln_connector: Optional[ELNConnector] = None,
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.target = target
        self.eln_connector = eln_connector
        super().__init__(data_path)

    def objective_function(self, actual, target):
        return abs(target - actual)

    def get_rxn_conditions(
        self,
        conditions=[
            "composition",
            "synthesis",
            "time (min)",
            "temperature (°c)",
            "heating ramp (°c/min)",
            "shape",
        ],
    ):
        self.conditions_len = len(conditions)
        return self.data[conditions]

    def get_ids(self):
        return self.data["$id"]

    def create_composition_mapping(self, df):
        unique_values = df["composition"].astype("category").cat.categories
        codes = range(len(unique_values))
        self.composition_mapping = dict(zip(codes, unique_values))

    def get_rxn_reagents(
        self,
    ):
        def reagents2series(row):
            data_dict = {item["iupac"]: item["mmoles"] for item in row["reagents"]}
            return pd.Series(data_dict)

        reagents_df = self.data.apply(reagents2series, axis=1).fillna(0)
        self.reagents_len = reagents_df.shape[-1]
        return reagents_df

    def mapping(
        self,
    ):
        return {
            "sphere": 17000,
            "octahedron": 17935,
            "cube": 20832,
            "tetrahedron": 22596,
            "rhombic dodecahedron": 19157,
            "rounded cube": 20300,
            "large particle (disregard)": 15000,
            "twinned particle": 16000,
            "small particle": 16000,
            "particle": 16000,
            "elongated sphere, small sphere": 17000,
            "truncated octahedron": 17500,
            "truncated rhombic dodecahedron": 18400,
            "rounded tetrahedron": 21700,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        if self.eln_connector is not None:
            self.data = self.eln_connector.data

            reagent_dicts = self.data["reagents"].apply(create_reagent_mapping)
            self.reagent_mapping = {k: v for dict_ in reagent_dicts for k, v in dict_.items()}

            self.data = pd.concat(
                [self.get_rxn_conditions(), self.get_rxn_reagents(), self.get_ids()],
                axis=1,
            )
            self.data.rename(
                columns={col: col.lower() for col in self.data.columns},
                inplace=True,
            )
            self.data = self.clean_and_fill_columns(self.data)
            # self.filter_by_occurences()

            expanded_scales = self.data["shape"].apply(lambda x: get_scales(x, self.mapping()))
            # self.data.rename(columns={"shape": "target"}, inplace=True)
            self.data["target"] = expanded_scales.mean(axis=1, skipna=True)

            # self.data = self.data.join(expanded_scales)

            # print(self.data.columns)

            # self.data["target"] = self.data.apply(
            #     lambda row: get_scales(row["shape"], self.mapping()), axis=1, result_type="expand"
            # )

        else:
            self.data = pd.read_csv(self.data_path, index_col=None)

        # self.data = self.data.loc[:, (self.data != 0).any(axis=0)]

        train_x = (
            self.data.drop([self.target_column, "$id", "shape"], axis=1).values
            if "$id" in self.data.columns
            else self.data.drop(self.target_column, axis=1).values
        )

        train_y = np.array(
            [
                self.objective_function(labels, self.target)
                for labels in self.data[self.target_column]
            ]
        )
        train_y = train_y.reshape(-1, 1)

        self.train_x = torch.from_numpy(train_x).to(torch.float64)
        self.train_y = torch.from_numpy(train_y).to(torch.float64)

    def clean_and_fill_columns(self, df):
        def clean_shapes(shapes):
            shapes = (
                shapes.replace("-", "unknown")
                .replace("others", "unknown")
                .replace("?", "unknown")
                .strip()
                .split(", ")
            )
            shapes_singular_list = [return_singular(shape) for shape in shapes]
            return str(shapes_singular_list)

        original_composition = df["composition"].copy()

        df["shape"] = df["shape"].apply(clean_shapes)
        df = df.replace("-", np.NaN).replace("", np.NaN)
        df["composition"] = df["composition"].fillna("Cu_unknown_mix")

        column_types = ["category", "category", "float", "float", "float", "str"] + [
            "float"
        ] * self.reagents_len

        self.create_composition_mapping(df)

        df = df.astype(dict(zip(df.columns.tolist(), column_types)))
        for i, col_type in enumerate(column_types):
            if col_type == "category":
                df[df.columns[i]] = df[df.columns[i]].cat.codes

        mean_heating_ramps = df.groupby("shape")["heating ramp (°c/min)"].transform("mean")
        df["heating ramp (°c/min)"] = df["heating ramp (°c/min)"].fillna(mean_heating_ramps)

        # Fill missing values in columns
        # df["heating ramp (°c/min)"] = df["heating ramp (°c/min)"].fillna(
        #     df["heating ramp (°c/min)"].mean()
        # )

        df = df.fillna(0)
        return df

    def filter_by_occurences(self, value_count=1):
        self.data["shape"] = self.data["shape"].apply(lambda x: ast.literal_eval(x))
        self.data["shape"] = self.data["shape"].apply(tuple)

        class_counts = self.data["shape"].value_counts()
        print(class_counts)

        mask = self.data["shape"].isin(class_counts.index[class_counts > value_count])
        self.data = self.data.loc[mask]
        self.data["shape"] = self.data["shape"].apply(str)

    def decode(self, x):
        suggestions = pd.DataFrame(
            torch.vstack(x), columns=list(self.data.drop("shape", axis=1).columns)
        )

        # conversion function
        def convert_to_mass_and_volume(row):
            new_data = {}  # dictionary to hold new data
            for col in row.index:
                if col in self.reagent_mapping.keys():
                    mmol = row[col]
                    mw = self.reagent_mapping[col]["mw"]
                    density = self.reagent_mapping[col]["density"]
                    # convert mmol to mass in mg
                    mass = mmol * mw
                    new_data[col + "_mass_mg"] = mass

                    # convert mass to volume in ml, if density is available
                    if density is not None:
                        volume = mass / density
                        new_data[col + "_vol_μl"] = volume

            return pd.Series(new_data)

        conditions_df = suggestions[
            [
                "composition",
                "synthesis",
                "time (min)",
                "temperature (°c)",
                "heating ramp (°c/min)",
            ]
        ]
        reagents_mass_and_volume_df = suggestions.apply(convert_to_mass_and_volume, axis=1)

        filtered_suggestions = []
        for _, row in reagents_mass_and_volume_df.iterrows():
            # create a new dictionary with only the keys that have values above the threshold
            filtered_reagents = {
                k: v
                for k, v in row.items()
                if ("mass_mg" in k and v >= 2) or ("vol_μl" in k and v >= 2)
            }
            filtered_suggestions.append(filtered_reagents)

        all_keys = set(k for d in filtered_suggestions for k in d.keys())
        default_dict = {k: 0 for k in all_keys}
        filtered_suggestions = [{**default_dict, **d} for d in filtered_suggestions]
        filtered_reagents_df = pd.DataFrame(filtered_suggestions)

        conditions_df = conditions_df.reset_index(drop=True)
        filtered_reagents_df = filtered_reagents_df.reset_index(drop=True)
        final_df = pd.concat([conditions_df, filtered_reagents_df], axis=1)

        list_of_dicts = final_df.to_dict("records")
        sorted_list_of_dicts = [dict(sorted(d.items())) for d in list_of_dicts]

        return sorted_list_of_dicts


class BOParticlesDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: str = "../data/nanoparticles/particles_features.csv",
        shape: str = "rod",
        eln_connector: Optional[ELNConnector] = None,
        filter_by_target_occurences: bool = False,
    ):
        self.data_path = data_path
        self.shape = shape
        self.eln_connector = eln_connector
        self.filter_by_target_occurences = filter_by_target_occurences

        super().__init__(data_path)

    def shape_objective(self, true_labels, shape_index):
        shape_probability = true_labels[shape_index]
        other_probabilities = np.delete(true_labels, shape_index)
        sum_other_probability = np.sum(other_probabilities)
        return shape_probability - sum_other_probability

    def get_ids(self):
        return self.data["$id"]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.eln_connector is not None:
            self.data = self.eln_connector.data

            reagent_dicts = self.data["reagents"].apply(create_reagent_mapping)
            self.reagent_mapping = {k: v for dict_ in reagent_dicts for k, v in dict_.items()}

            # self.data = self.data[self.data["status"].apply(is_not_electrochemistry)]
            # self.data = pd.concat(
            #     [self.get_rxn_conditions(), self.get_rxn_reagents()],
            #     axis=1,
            # )
            self.data = pd.concat(
                [self.get_rxn_conditions(), self.get_rxn_reagents(), self.get_ids()],
                axis=1,
            )
            self.data.rename(
                columns={col: col.lower() for col in self.data.columns},
                inplace=True,
            )
            self.data = self.clean_and_fill_columns(self.data)
            if self.filter_by_target_occurences:
                self.filter_by_occurences()

        else:

            self.data = pd.read_csv(self.data_path, index_col=None)

        # self.data = self.data.loc[:, self.data.nunique() != 1]
        # train_x = self.data.drop("shape", axis=1).values
        train_x = (
            self.data.drop(["shape", "$id"], axis=1).values
            if "$id" in self.data.columns
            else self.data.drop(self.target_column, axis=1).values
        )
        self.convert_shape_labels()

        train_y = np.array(
            [self.shape_objective(labels, self.shape_index) for labels in self.true_labels]
        )
        train_y = train_y.reshape(-1, 1)

        self.train_x = torch.from_numpy(train_x).to(torch.float64)
        self.train_y = torch.from_numpy(train_y).to(torch.float64)

    def create_composition_mapping(self, df):
        unique_values = df["composition"].astype("category").cat.categories
        codes = range(len(unique_values))
        self.composition_mapping = dict(zip(codes, unique_values))

    def convert_shape_labels(self):
        self.data["shape"] = self.data["shape"].apply(lambda x: ast.literal_eval(x))
        self.mlb = MultiLabelBinarizer()
        shape_lists = self.data["shape"]
        self.true_labels = self.mlb.fit_transform(shape_lists).astype(np.float64)
        self.true_labels /= self.true_labels.sum(axis=1)[:, np.newaxis].astype(np.float64)
        self.shape_index = list(self.mlb.classes_).index(self.shape)

    def filter_by_occurences(self, value_count=1):
        self.data["shape"] = self.data["shape"].apply(lambda x: ast.literal_eval(x))
        self.data["shape"] = self.data["shape"].apply(tuple)

        class_counts = self.data["shape"].value_counts()
        mask = self.data["shape"].isin(class_counts.index[class_counts > value_count])
        self.data = self.data.loc[mask]
        self.data["shape"] = self.data["shape"].apply(str)

    def get_rxn_conditions(
        self,
        conditions=[
            "composition",
            "synthesis",
            "time (min)",
            "temperature (°c)",
            "heating ramp (°c/min)",
            "shape",
        ],
    ):
        self.conditions_len = len(conditions)
        return self.data[conditions]

    def get_rxn_reagents(
        self,
    ):
        def reagents2series(row):
            data_dict = {item["iupac"]: item["mmoles"] for item in row["reagents"]}
            return pd.Series(data_dict)

        reagents_df = self.data.apply(reagents2series, axis=1).fillna(0)
        self.reagents_len = reagents_df.shape[-1]
        return reagents_df

    def clean_and_fill_columns(self, df):
        def clean_shapes(shapes):
            shapes = (
                shapes.replace("-", "unknown")
                .replace("others", "unknown")
                .replace("?", "unknown")
                .strip()
                .split(", ")
            )
            shapes_singular_list = [return_singular(shape) for shape in shapes]
            return str(shapes_singular_list)

        original_composition = df["composition"].copy()

        # clean and singularize shapes
        df["shape"] = df["shape"].fillna("unknown")
        df["shape"] = df["shape"].apply(clean_shapes)

        df = df.replace({"-": np.NaN, "": np.NaN})

        df["composition"] = df["composition"].fillna("Cu_unknown_mix")

        column_types = ["category", "category", "float", "float", "float", "str"] + [
            "float"
        ] * self.reagents_len

        # df = df.convert_dtypes().apply(pd.Categorical)
        # df = df.convert_dtypes().apply(pd.to_numeric, errors="ignore")
        self.create_composition_mapping(df)

        df = df.astype(dict(zip(df.columns.tolist(), column_types)))
        for i, col_type in enumerate(column_types):
            if col_type == "category":
                df[df.columns[i]] = df[df.columns[i]].cat.codes

        # fill missing values in columns
        df["heating ramp (°c/min)"] = df["heating ramp (°c/min)"].fillna(
            df["heating ramp (°c/min)"].mean()
        )

        df = df.fillna(0)
        return df

    def decode(self, x):
        suggestions = pd.DataFrame(
            torch.vstack(x), columns=list(self.data.drop(["shape", "$id"], axis=1).columns)
        )
        # root_path = Path(Path.cwd()).parts[0]
        # with open(root_path / "data/nanoparticles/eln_reagent_mapping.json", "r") as f:
        #     mapping_dict = json.load(f)

        # conversion function
        def convert_to_mass_and_volume(row):
            new_data = {}
            for col in row.index:
                if col in self.reagent_mapping.keys():
                    mmol = row[col]
                    mw = self.reagent_mapping[col]["mw"]
                    density = self.reagent_mapping[col]["density"]
                    mass = mmol * mw
                    new_data[col + "_mass_mg"] = mass

                    if density is not None:
                        volume = mass / density
                        new_data[col + "_vol_μl"] = volume

            return pd.Series(new_data)

        conditions_df = suggestions[
            [
                "composition",
                # "synthesis",
                "time (min)",
                "temperature (°c)",
                "heating ramp (°c/min)",
            ]
        ]
        reagents_mass_and_volume_df = suggestions.apply(convert_to_mass_and_volume, axis=1)

        filtered_suggestions = []
        for _, row in reagents_mass_and_volume_df.iterrows():
            filtered_reagents = {
                k: v
                for k, v in row.items()
                if ("mass_mg" in k and v >= 2) or ("vol_μl" in k and v >= 2)
            }
            filtered_suggestions.append(filtered_reagents)

        all_keys = set(k for d in filtered_suggestions for k in d.keys())
        default_dict = {k: 0 for k in all_keys}

        filtered_suggestions = [{**default_dict, **d} for d in filtered_suggestions]
        filtered_reagents_df = pd.DataFrame(filtered_suggestions)

        conditions_df = conditions_df.reset_index(drop=True)
        filtered_reagents_df = filtered_reagents_df.reset_index(drop=True)

        final_df = pd.concat([conditions_df, filtered_reagents_df], axis=1)

        # filtered_reagents_df = pd.DataFrame(filtered_suggestions)
        # final_df = pd.concat([conditions_df, filtered_reagents_df], axis=1)

        list_of_dicts = final_df.to_dict("records")
        sorted_list_of_dicts = [dict(sorted(d.items())) for d in list_of_dicts]

        return sorted_list_of_dicts


if __name__ == "__main__":
    eln_connector = ELNConnector(instance="https://eln-beta.epfl.ch/roc/")
    particles_data = BOParticlesDataModule(eln_connector=eln_connector)

    print(particles_data.data.shape)
