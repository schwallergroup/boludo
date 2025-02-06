import torch
from cheminfopy import User
from cheminfopy import Experiment
import pandas as pd
from operator import itemgetter
from bs4 import BeautifulSoup
import inflect
import logging
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["torch_delete_rows"]


def torch_delete_rows(tensor, indices):
    indices_to_keep = torch.ones(tensor.shape[0], dtype=torch.bool)
    indices_to_keep[indices] = False
    return tensor[indices_to_keep]


def clean_html_text(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    for h1 in soup.find_all("h1"):
        h1.insert_after(" ")
    text_only = soup.get_text()

    return text_only


def filter_experiments(user):
    filtered_experiments = []
    experiments = user.get_experiment_toc()
    for experiment in experiments:
        if (
            "user@epfl.ch" in experiment["key"]
            and experiment["value"].get("status", {}).get("label") != "Electrochemistry"
        ):
            filtered_experiments.append(experiment)
    return filtered_experiments


def extract_features_from_experiment(
    instance="https://eln-beta.epfl.ch/roc/",
    token="",
):
    user = User(instance=instance, token=token)
    experiment_uuids = [ex["id"] for ex in filter_experiments(user)]
    experiments = [
        Experiment(instance=instance, experiment_uuid=uuid, token=token)
        for uuid in experiment_uuids
    ]
    experiments_content = [ex.toc for ex in experiments]

    df = pd.DataFrame(
        {
            "synthesis": pd.Categorical(
                [
                    exp["$content"].get("meta", {}).get("Synthesis", "")
                    for exp in experiments_content
                ]
            ).codes,
            "composition": pd.Categorical(
                [
                    exp["$content"].get("meta", {}).get("Composition", "")
                    for exp in experiments_content
                ]
            ).codes,
            "temp": pd.to_numeric(
                pd.Series(
                    [
                        exp["$content"].get("meta", {}).get("temperature (°C)", "")
                        for exp in experiments_content
                    ]
                ),
                errors="coerce",
            ).astype(float),
            "time": pd.to_numeric(
                pd.Series(
                    [
                        exp["$content"].get("meta", {}).get("time (min)", "")
                        for exp in experiments_content
                    ]
                ),
                errors="coerce",
            ).astype(float),
            "literature": [
                exp["$content"].get("meta", {}).get("literature", "") for exp in experiments_content
            ],
            "shape": [
                exp["$content"].get("meta", {}).get("Shape", "") for exp in experiments_content
            ],
            "heating_ramp": pd.to_numeric(
                pd.Series(
                    [
                        exp["$content"].get("meta", {}).get("heating ramp (°C/min)", "")
                        for exp in experiments_content
                    ]
                ),
                errors="coerce",
            ).astype(float),
            "reagents": list(
                map(
                    itemgetter("reagents"),
                    list(map(itemgetter("$content"), experiments_content)),
                )
            ),
            "reagents_mmoles": [
                list(map(itemgetter("mmoles"), experiment.reagents)) for experiment in experiments
            ],
            "procedure": [clean_html_text(experiment.procedure) for experiment in experiments],
        }
    )

    # class_counts = df["shape"].value_counts()
    # mask = df["shape"].isin(class_counts.index[class_counts > 1])
    # df = df.loc[mask]
    # df = df[df["shape"] != "-"]
    # df['shape_code'] = pd.Categorical(df['shape']).codes

    def map_reagent_quantities(row):
        vals = row["reagents_mmoles"]
        rgts = [reagent["iupac"] for reagent in row["reagents"]]
        rgts2quants = dict(zip(rgts, vals))
        return pd.to_numeric(pd.Series(rgts2quants), errors="coerce").astype("float")

    df = pd.concat([df, df.apply(map_reagent_quantities, axis=1).fillna(0)], axis=1)

    return df


def return_singular(word):
    p = inflect.engine()
    try:
        singular = p.singular_noun(word)
    except:
        singular = word
    return singular if singular else word


def is_not_electrochemistry(statuses, label="Electrochemistry"):
    if statuses is None or isinstance(statuses, float):
        return True
    if not statuses:
        return True
    try:
        for status in statuses:
            if "label" in status and status["label"] == label:
                return False
    except TypeError:
        print(f"Non-iterable value found: {statuses}")
        return True
    return True


def apply_filters(df, label):
    def is_not(statuses):
        if statuses is None or isinstance(statuses, float) or not statuses:
            return True
        try:
            for status in statuses:
                if "label" in status:
                    if status["label"] == label:
                        return False
                    else:
                        return True
        except TypeError:
            logging.warning(f"Non-iterable value found: {statuses}")
            return True

    df = df[df["status"].apply(is_not)]
    return df


def calculate_closest_points(suggestions, originals, metric="cityblock"):
    suggestions_np = suggestions.to_numpy()
    originals_np = originals.to_numpy()

    distances = cdist(suggestions_np, originals_np, metric=metric)

    closest_points_indices = []
    minimum_distances = []

    for i in range(len(suggestions)):
        min_dist_index = np.argmin(distances[i, :])
        min_dist = distances[i, min_dist_index]
        minimum_distances.append(min_dist)
        closest_points_indices.append(min_dist_index)

    return minimum_distances, closest_points_indices


import operator
import pandas as pd


def filter_data(data, filters):
    ops = {
        "==": operator.eq,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "!=": operator.ne,
    }
    for column, (op, value) in filters.items():
        if op in ops:
            data = data[ops[op](data[column], value)]
    return data


def remove_outliers(data, id_column, outliers):
    return data[~data[id_column].isin(outliers)]


def scale_features(data, scale_column, reagent_columns):
    scale_factor = 45.421621 / data[scale_column]
    # data.loc[:, reagent_columns] *= scale_factor
    data.loc[:, reagent_columns] *= scale_factor.values[:, None]
    return data
