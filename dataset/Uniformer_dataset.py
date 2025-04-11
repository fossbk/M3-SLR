import os
import numpy as np
import sys
import torch
from torch.utils.data import  Dataset
import pandas as pd
from dataset.videoLoader import get_selected_indexs,pad_index
from decord import VideoReader
from utils.video_augmentation import *

class UFOneView_Dataset(Dataset):
    def __init__(self, base_url,split,dataset_cfg,train_labels = None,**kwargs):
        if train_labels is None:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)
        
    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        else:
            print("Build test/val transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        return transform
    def read_videos(self,name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        clip = []
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}' 
        vr = VideoReader(path,width=320, height=256)
        # print(path)
        sys.stdout.flush()
        vlen = len(vr)
        selected_index, pad = get_selected_indexs(vlen,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
        frames = vr.get_batch(selected_index).asnumpy()
        clip = []
        for frame in frames:
            clip.append(self.transform(frame))
            
        clip = torch.stack(clip,dim = 0)
        return clip

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values

        name,label = data[0],data[1]

        clip = self.read_videos(name)
        
        return clip, torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)
    
class UFThreeView_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, **kwargs):

        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url, f'{split}.csv'), sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",
                      os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(
                    os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),
                    sep=',')

        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        else:
            print("Build test/val transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        return transform

    def count_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames, width, height
    
    def read_one_view(self,name, selected_index):
        clip = []
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}'    
        vr = VideoReader(path,width=320, height=256)
        # print(path)
        # sys.stdout.flush()
        frames = vr.get_batch(selected_index).asnumpy()
        clip = []
        for frame in frames:
            clip.append(self.transform(frame))
            
        clip = torch.stack(clip,dim = 0)
        return clip

    def read_videos(self, center, left, right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])
        #
        vlen1, _, _ = self.count_frames(os.path.join(self.base_url, 'videos', center))
        vlen2, _, _ = self.count_frames(os.path.join(self.base_url, 'videos', left))
        vlen3, _, _ = self.count_frames(os.path.join(self.base_url, 'videos', right))

        min_vlen = min(vlen1, min(vlen2, vlen3))
        max_vlen = max(vlen1, max(vlen2, vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_center = self.read_one_view(center, selected_index)

            rgb_left = self.read_one_view(left, selected_index)

            rgb_right = self.read_one_view(right, selected_index)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_center = self.read_one_view(center, selected_index)

            selected_index, pad = get_selected_indexs(vlen2 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_left = self.read_one_view(left, selected_index)

            selected_index, pad = get_selected_indexs(vlen3 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_right = self.read_one_view(right, selected_index)

        return rgb_left, rgb_center, rgb_right

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values

        center, left, right, label = data[0], data[1], data[2], data[3]
        rgb_left, rgb_center, rgb_right= self.read_videos(center, left, right)

        return rgb_left, rgb_center, rgb_right, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)