from utils.video_augmentation import *
from dataset.Uniformer_dataset import UFOneView_Dataset, UFThreeView_Dataset
from dataset.MaskUniformer_dataset import MaskUFOneView_Dataset, MaskUFThreeView_Dataset

def build_dataset(dataset_cfg, split,model = None,**kwargs):
    dataset = None
    
    if dataset_cfg['model_name'] == 'UFOneView' or dataset_cfg['model_name'] == 'mvit_v2' or dataset_cfg['model_name'] == 'swin':
        dataset = UFOneView_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    if dataset_cfg['model_name'] == 'MaskUFOneView':
        dataset = MaskUFOneView_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == 'UFThreeView' or dataset_cfg['model_name'] == 'UsimKD':
        dataset = UFThreeView_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    assert dataset is not None
    return dataset



    