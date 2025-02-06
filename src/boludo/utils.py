from collections.abc import MutableMapping
import importlib


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def instantiate_class(input_dict, *args, **kwargs):
    class_path = input_dict["class_path"]
    init_args = input_dict.get("init_args", {})
    init_args.update(kwargs)

    # recursively instantiate classes
    for arg_name, arg_value in init_args.items():
        if isinstance(arg_value, dict) and "class_path" in arg_value:
            init_args[arg_name] = instantiate_class(arg_value)

    module_name, class_name = class_path.rsplit(".", 1)
    MyClass = getattr(importlib.import_module(module_name), class_name)
    instance = MyClass(*args, **init_args)

    return instance


import operator


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
    data.loc[:, reagent_columns] *= scale_factor.values[:, None]
    return data
