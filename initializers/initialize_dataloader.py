from pathlib import Path
from datasets.imaging_dataloader import CMRImageDataModule
from datasets.imaging_tabular_dataloader import CMRImageTabularDataModule


def initialize_dataloader(args, params, paths):
    if args.module == "imaging":
        cmr_path_pickle_path = Path(paths.image_subject_paths_folder) / params.data.cmr_path_pickle_name
        data_module = CMRImageDataModule(dataloader_image_file_folder=paths.dataloader_image_file_folder,
                                         dataloader_tabular_file_folder=paths.dataloader_tabular_file_folder,
                                         cmr_path_pickle_path=cmr_path_pickle_path,
                                         )
        
        data_module.setup("fit", multi_gpu=params.trainer.devices > 1, **params.data.__dict__)

    elif args.module == "imaging_tabular":
        cmr_path_pickle_path = Path(paths.image_subject_paths_folder) / params.data.imaging_data.cmr_path_pickle_name
        data_module = CMRImageTabularDataModule(dataloader_image_file_folder=paths.dataloader_image_file_folder,
                                                dataloader_tabular_file_folder=paths.dataloader_tabular_file_folder,
                                                cmr_path_pickle_path=cmr_path_pickle_path,
                                                )
            
        data_module.setup("fit", multi_gpu=params.trainer.devices > 1,
                          **params.data.general_data.__dict__,
                          **params.data.imaging_data.__dict__,
                          **params.data.tabular_data.__dict__,
                          **params.data.validation_data.__dict__)

    else:
        raise ValueError("We only supprt imaging and imaging tabulet dataloaders")
    return data_module
