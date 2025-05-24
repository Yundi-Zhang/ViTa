from dataclasses import asdict, fields
import yaml
from typing import Any, Dict

from configs import imaging_params
from configs import imaging_tabular_params


def initialize_parameters(args):
    config_path = args.config
    # Override the default parameters by the given configuration file
    if args.module == "imaging":
        params = load_imaging_model_config_from_yaml(config_path)
    elif args.module in ["tabular", "imaging_tabular"]:
        params = load_imaging_tabular_model_config_from_yaml(config_path)
    else:
        raise ValueError("We only support imaging or imaging_tabular module")
    return params


def load_imaging_model_config_from_yaml(file_path):
    config_data = dict()
    if file_path is not None:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

    # Get the default values from the data class
    params = imaging_params.Params(general=imaging_params.GeneralParams(), 
                                   data=imaging_params.DataParams(), 
                                   trainer=imaging_params.TrainerParams(), 
                                   module=imaging_params.ModuleParams(
                                       recon_hparams=imaging_params.ReconMAEParams(),
                                       seg_hparams=imaging_params.SegMAEParams(),
                                       regr_hparams=imaging_params.RegrMAEParams(),
                                       training_hparams=imaging_params.TrainingParams()))
    update_params = update_dataclass_from_dict(params, config_data)

    return update_params


def load_imaging_tabular_model_config_from_yaml(file_path):
    config_data = dict()
    if file_path is not None:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

    # Get the default values from the data class
    params = imaging_tabular_params.Params(general=imaging_tabular_params.GeneralParams(), 
                                           data=imaging_tabular_params.DataParams(
                                               general_data=imaging_tabular_params.GeneralDataParams(),
                                               imaging_data=imaging_tabular_params.ImagingDataParams(),
                                               tabular_data=imaging_tabular_params.TabularDataParams(), 
                                               validation_data=imaging_tabular_params.ValidationDataParams()), 
                                           trainer=imaging_tabular_params.TrainerParams(), 
                                           module=imaging_tabular_params.ModuleParams(
                                               imaging_hparams=imaging_tabular_params.ImagingParams(),
                                               tabular_hparams=imaging_tabular_params.TabularParams(),
                                               training_hparams=imaging_tabular_params.TrainingParams()))
    update_params = update_dataclass_from_dict(params, config_data)

    return update_params


def update_dataclass_from_dict(params, config_data: Dict[str, Any]):
    updated_fields = {}
    instance_dict = asdict(params)
    for key in config_data:
        if is_field_name(params, key):
            value = config_data[key]
            if isinstance(value, dict) and hasattr(getattr(params, key), '__dataclass_fields__'):
                # Recursively update nested dataclass
                updated_value = update_dataclass_from_dict(getattr(params, key), value)
                updated_fields[key] = updated_value
            else:
                updated_fields[key] = value
            instance_dict.update(updated_fields)
        else:
            raise NameError(f"{key} is not defined in the dataclass")
    return params.__class__(**instance_dict)


def is_field_name(dataclass_type, field_name):
    return field_name in [f.name for f in fields(dataclass_type)]