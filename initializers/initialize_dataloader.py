from datasets.imaging_dataloader import CMRImageDataModule
from datasets.imaging_tabular_dataloader import CMRImageTabularDataModule
from datasets.tabular_dataloader import TabularDataModule



def initialize_dataloader(args, params, paths):
    if args.module == "imaging":
        table_condition_dict = {"health_flag": params.data.health_flag, "sorting_with_age": True,}
        data_module = CMRImageDataModule(load_dir=paths.image_root_folder, 
                                         processed_dir=paths.image_processed_folder,
                                         dataloader_image_file_folder=paths.dataloader_image_file_folder,
                                         dataloader_tabular_file_folder=paths.dataloader_tabular_file_folder,
                                         cmr_path_pickle_name=params.data.cmr_path_pickle_name,
                                         subj_ids_with_required_size_pickle_name=params.data.subj_ids_with_required_size_pickle_name,
                                         )
        
        data_module.setup("fit", multi_gpu=params.trainer.devices > 1,
                          table_condition_dict=table_condition_dict, **params.data.__dict__)
        
    elif args.module == "tabular":
        data_module = TabularDataModule(dataloader_image_file_folder=paths.dataloader_image_file_folder,
                                        dataloader_tabular_file_folder=paths.dataloader_tabular_file_folder,
                                        cmr_path_pickle_name=params.data.imaging_data.cmr_path_pickle_name,
                                        )
            
        data_module.setup("fit", multi_gpu=params.trainer.devices > 1,
                          **params.data.general_data.__dict__,
                          **params.data.imaging_data.__dict__,
                          **params.data.tabular_data.__dict__,
                          **params.data.validation_data.__dict__)
        
    elif args.module == "imaging_tabular":
        data_module = CMRImageTabularDataModule(load_dir=paths.image_root_folder, 
                                                processed_dir=paths.image_processed_folder,
                                                dataloader_image_file_folder=paths.dataloader_image_file_folder,
                                                dataloader_tabular_file_folder=paths.dataloader_tabular_file_folder,
                                                cmr_path_pickle_name=params.data.imaging_data.cmr_path_pickle_name,
                                                subj_ids_with_required_size_pickle_name=params.data.imaging_data.subj_ids_with_required_size_pickle_name,
                                                )
            
        data_module.setup("fit", multi_gpu=params.trainer.devices > 1,
                          **params.data.general_data.__dict__,
                          **params.data.imaging_data.__dict__,
                          **params.data.tabular_data.__dict__,
                          **params.data.validation_data.__dict__)

    else:
        raise ValueError("We only supprt imaging, tabulet, and imaging tabulet dataloaders")
    return data_module