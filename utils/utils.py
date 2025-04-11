import torch.nn as nn
import torch.optim as optim
from modelling.Uniformer import UFOneView, UFThreeView, UsimKD
from modelling.mvit_v2 import mvit_v2_s
from modelling.swin_transformer import SwinTransformer3d
import torch
from trainer.tools import MyCustomLoss,MultipleMSELoss
from torchvision import models
from torch.nn import functional as F
from collections import OrderedDict
from pytorch_lightning.utilities.migration import pl_legacy_patch

def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'],weight_local=train_cfg.get('weight_local',1))
    if train_cfg['criterion'] == "MultipleMSELoss":
        criterion = MultipleMSELoss()
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg,model,criterion=None):
    optimzer = None
    params = list(model.parameters())
    if criterion is not None:
        params += list(criterion.parameters())
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(params, lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']),momentum=0.9,nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(params, lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    try:
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    except:
        pass


def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")
        if cfg['data']['model_name'] == 'mvit_v2':
            model = mvit_v2_s(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.reset_head(226)
                model.load_state_dict(state_dict)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(state_dict)
        elif cfg['data']['model_name'] == 'swin':
            model = SwinTransformer3d(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.reset_head(226)
                model.load_state_dict(state_dict)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(state_dict)
        elif cfg['data']['model_name'] == 'UFOneView' or cfg['data']['model_name'] == 'MaskUFOneView':
            model = UFOneView(**cfg['model'],device=cfg['training']['device'])
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
        elif cfg['data']['model_name'] == 'UFThreeView' or cfg['data']['model_name'] == 'MaskUFThreeView':
            model = UFThreeView(**cfg['model'],device=cfg['training']['device'])
            missing, unexpected = model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
            if len(missing) > 0:
                print("Các tham số chưa khớp:", missing)
            if len(unexpected) > 0:
                print("Các tham số thừa không dùng:", unexpected)
        elif cfg['data']['model_name'] == 'UsimKD':
            model = UsimKD(**cfg['model'],device=cfg['training']['device'])
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
    else:
        if cfg['data']['model_name'] == 'mvit_v2':
            model = mvit_v2_s(**cfg['model'])
            model.reset_head(400)
            weights = models.video.MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
            model.load_state_dict(weights)
            model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'swin':
            model = SwinTransformer3d(**cfg['model'])
            weights = models.video.Swin3D_T_Weights.DEFAULT.get_state_dict(progress=True)
            model.reset_head(400)
            model.load_state_dict(weights)
            model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'UFOneView' or cfg['data']['model_name'] == 'MaskUFOneView':
            model = UFOneView(**cfg['model'],device=cfg['training']['device'])
        elif cfg['data']['model_name'] == 'UFThreeView' or cfg['data']['model_name'] == 'MaskUFThreeView':
            model = UFThreeView(**cfg['model'],device=cfg['training']['device'])
        elif cfg['data']['model_name'] == 'UsimKD':
            model = UsimKD(**cfg['model'],device=cfg['training']['device'])

    assert model is not None
    print("loaded model")
    return model
        