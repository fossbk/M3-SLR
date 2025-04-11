from dataset.dataset import build_dataset
import torch

def ufoneview_train_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0).permute(0,2,1,3,4) # b,t,c,h,w -> b,c,t,h,w
    gloss = [s[1] for s in batch]
    labels = torch.stack([s[2] for s in batch],dim = 0)
    return {'clip':clip, 'gloss':gloss},labels

def ufoneview_infer_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0).permute(0,2,1,3,4) # b,t,c,h,w -> b,c,t,h,w
    labels = torch.stack([s[2] for s in batch],dim = 0)
    return {'clip':clip},labels

def maskufoneview_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0).permute(0,2,1,3,4)
    mask = torch.stack([s[1] for s in batch],dim=0)

    clip = (clip, mask)
    return {'clip':clip}


def ufthreeview_train_collate_fn_(batch):
    rgb_left = torch.stack([s[0] for s in batch], dim=0).permute(0,2,1,3,4)

    rgb_center = torch.stack([s[1] for s in batch], dim=0).permute(0,2,1,3,4)

    rgb_right = torch.stack([s[2] for s in batch], dim=0).permute(0,2,1,3,4)

    gloss = [s[3] for s in batch]

    labels = torch.stack([s[4] for s in batch], dim=0)

    return {'rgb_left': rgb_left, 'rgb_center': rgb_center, 'rgb_right': rgb_right, 'gloss': gloss}, labels

def ufthreeview_infer_collate_fn_(batch):
    rgb_left = torch.stack([s[0] for s in batch], dim=0).permute(0,2,1,3,4)

    rgb_center = torch.stack([s[1] for s in batch], dim=0).permute(0,2,1,3,4)

    rgb_right = torch.stack([s[2] for s in batch], dim=0).permute(0,2,1,3,4)

    labels = torch.stack([s[4] for s in batch], dim=0)

    return {'rgb_left': rgb_left, 'rgb_center': rgb_center, 'rgb_right': rgb_right}, labels

def build_dataloader(cfg, split, is_train=True, model = None,labels = None):
    dataset = build_dataset(cfg['data'], split,model,train_labels = labels)

    if cfg['data']['model_name'] == 'UFOneView' or cfg['data']['model_name'] == 'mvit_v2' or cfg['data']['model_name'] == 'swin':
        if is_train:
            collate_func = ufoneview_train_collate_fn_
        else:
            collate_func = ufoneview_infer_collate_fn_
    if cfg['data']['model_name'] == 'UFThreeView' or cfg['data']['model_name'] == 'UsimKD':
        if is_train:
            collate_func = ufthreeview_train_collate_fn_
        else:
            collate_func = ufthreeview_infer_collate_fn_
    
    if cfg['data']['model_name'] == 'MaskUFOneView':
        collate_func = maskufoneview_collate_fn_

    dataloader = torch.utils.data.DataLoader(dataset,
                                            collate_fn = collate_func,
                                            batch_size = cfg['training']['batch_size'],
                                            num_workers = cfg['training'].get('num_workers',2),                                            
                                            shuffle = is_train,
                                            # prefetch_factor = cfg['training'].get('prefetch_factor',2),
                                            pin_memory=True,
                                            persistent_workers =  True,
                                            # sampler = sampler
                                            )

    return dataloader
