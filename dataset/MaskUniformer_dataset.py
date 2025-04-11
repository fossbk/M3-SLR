import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.video_augmentation import ToFloatTensor, PermuteImage
import sys
import pandas as pd
from dataset.videoLoader import get_selected_indexs,pad_index
from decord import VideoReader
from utils.video_augmentation import *
from skimage.transform import resize

class MaskingGenerator3D:
    def __init__(self, mask_window_size, num_masking_patches, min_num_patches=16, max_num_patches=None, min_aspect=0.3, max_aspect=None):
        self.temporal, self.height, self.width = mask_window_size
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        
    def get_shape(self):
        return self.temporal, self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(100):
            target_area = random.uniform(self.min_num_patches, self.max_num_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            t = random.randint(1, self.temporal)
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                front = random.randint(0, self.temporal - t)
                num_masked = mask[front:front+t, top:top+h, left:left+w].sum()
                if 0 < h * w * t - num_masked <= max_mask_patches:
                    for i in range(front, front+t):
                        for j in range(top, top+h):
                            for k in range(left, left+w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask

class MaxFlowCubeMaskingGenerator(MaskingGenerator3D):
    def __init__(
        self,
        mask_window_size,           
        num_masking_patches,        
        optical_flow,               
        random_masking,            
        min_num_patches=16,         
        max_num_patches=None,      
        min_aspect=0.3,             
        max_aspect=None            
    ):
        """
        Initializes the MaxFlowCubeMaskingGenerator.
        
        Args:
            mask_window_size (tuple): (temporal, height, width) of the masking window.
            num_masking_patches (int): Total number of patches to mask.
            optical_flow (np.ndarray): Optical flow tensor, shape (temporal, height, width).
            random_masking (bool): Masking mode (True: biased random, False: maximize flow).
            min_num_patches (int): Minimum number of patches per cube.
            max_num_patches (int): Maximum number of patches per cube. If None, defaults to the total number of patches within the mask_window_size.
            min_aspect (float): Minimum aspect ratio.
            max_aspect (float): Maximum aspect ratio. If None, defaults to 1 / min_aspect.
        """
        super().__init__(
            mask_window_size=mask_window_size,
            num_masking_patches=num_masking_patches,
            min_num_patches=min_num_patches,
            max_num_patches=max_num_patches,
            min_aspect=min_aspect,
            max_aspect=max_aspect
        )
        self.optical_flow = optical_flow  # Tensor optical flow shape (temporal, height, width)
        self.random_masking = random_masking

        self.flat_optical_flow = self.optical_flow.flatten()
        self.optical_flow_sum = self.flat_optical_flow.sum()
        if self.optical_flow_sum > 0:
            self.optical_flow_probs = self.flat_optical_flow / self.optical_flow_sum
        else:
            self.optical_flow_probs = np.ones_like(self.flat_optical_flow) / len(self.flat_optical_flow)

    def _sample_position(self, t, h, w):
        """
        Samples the starting position (front, top, left) based on optical flow.
        
        Args:
            t (int): Temporal dimension (or size) of the cube.
            h (int): Height of the cube.
            w (int): Width of the cube.
        
        Returns:
            tuple: (front, top, left) coordinates, or None if invalid.
        """
        front_max = self.temporal - t
        top_max = self.height - h
        left_max = self.width - w
        if front_max < 0 or top_max < 0 or left_max < 0:
            return None 

        center_t = t // 2
        center_h = h // 2
        center_w = w // 2
        prob_map = np.zeros((front_max + 1, top_max + 1, left_max + 1))
        for front in range(front_max + 1):
            for top in range(top_max + 1):
                for left in range(left_max + 1):
                    ct = min(max(front + center_t, 0), self.temporal - 1)
                    ch = min(max(top + center_h, 0), self.height - 1)
                    cw = min(max(left + center_w, 0), self.width - 1)
                    prob_map[front, top, left] = self.optical_flow[ct, ch, cw]

        if prob_map.sum() == 0:
            prob_map += 1
        prob_map = prob_map / prob_map.sum()
        flat_prob = prob_map.flatten()
        index = np.random.choice(len(flat_prob), p=flat_prob)
        front, top, left = np.unravel_index(index, prob_map.shape)
        return front, top, left

    def _mask(self, mask, max_mask_patches):
        """
        Creates a cube mask based on optical flow.
        
        Args:
            mask (np.ndarray): The current mask tensor with shape (temporal, height, width).
            max_mask_patches (int): The maximum number of patches that can be added to the mask.
        
        Returns:
            int: The number of newly masked patches (delta).
        """
        if self.random_masking:
            for _ in range(10):
                target_area = random.uniform(self.min_num_patches, self.max_num_patches)
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                t = random.randint(1, self.temporal)
                if w >= self.width or h >= self.height or t > self.temporal:
                    continue
                
                position = self._sample_position(t, h, w)
                if position is None:
                    continue
                front, top, left = position
                
                num_masked = mask[front:front+t, top:top+h, left:left+w].sum()
                if 0 < h * w * t - num_masked <= max_mask_patches:
                    delta = 0
                    for i in range(front, front+t):
                        for j in range(top, top+h):
                            for k in range(left, left+w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1
                    return delta
            return 0
        else:
            best_delta = 0
            best_cube = None
            best_sum = -1
            for _ in range(100): 
                target_area = random.uniform(self.min_num_patches, self.max_num_patches)
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                t = random.randint(1, self.temporal)
                if w >= self.width or h >= self.height or t > self.temporal:
                    continue
                
                position = self._sample_position(t, h, w)
                if position is None:
                    continue
                front, top, left = position
                
                cube_mask = mask[front:front+t, top:top+h, left:left+w]
                num_masked = cube_mask.sum()
                if 0 < h * w * t - num_masked <= max_mask_patches:
                    new_flow_sum = self.optical_flow[front:front+t, top:top+h, left:left+w][cube_mask == 0].sum()
                    if new_flow_sum > best_sum:
                        best_sum = new_flow_sum
                        best_cube = (front, t, top, h, left, w)
                        best_delta = h * w * t - num_masked
            
            if best_cube is not None:
                front, t, top, h, left, w = best_cube
                for i in range(front, front+t):
                    for j in range(top, top+h):
                        for k in range(left, left+w):
                            if mask[i, j, k] == 0:
                                mask[i, j, k] = 1
                return best_delta
            return 0

    def __call__(self):
        """
        Generates the complete mask.
        
        Returns:
            np.ndarray: The mask tensor with shape (temporal, height, width), where 1 indicates masked and 0 indicates unmasked.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask

    def get_shape(self):
        return self.temporal, self.height, self.width

class MaskUFOneView_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, **kwargs):
        if dataset_cfg['dataset_name'] == "VN_SIGN":
            print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
            self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
        print(split,len(self.train_labels))
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.is_train = True
        self.transform = self.build_transform()

    def build_transform(self):
        print("Build train transform")
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
        elif self.data_cfg['dataset_name'] == "AUTSL":
            path = f'{self.base_url}/{self.split}/{name}'   
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
    
    def compute_optical_flow(self,frames):
        optical_flows = []
        for i in range(len(frames) - 1):
            prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            next = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            optical_flows.append(mag)
        optical_flows.append(optical_flows[-1]) # Thêm flow cuối để đủ 16 frame
        return np.array(optical_flows)


    def _gen_mask(self, optical_flow=None):
        mask_window_size = [8, 7, 7] 
        mask_ratio = self.data_cfg['mask_ratio']
        num_masking_patches = round(np.prod(mask_window_size) * mask_ratio)
        max_mask = np.prod(mask_window_size[1:])
        min_mask = max_mask // 5
        if optical_flow is not None:
            generator = MaxFlowCubeMaskingGenerator(
                mask_window_size=mask_window_size,
                num_masking_patches=num_masking_patches,
                optical_flow=optical_flow,
                random_masking=False,
                min_num_patches=min_mask,
                max_num_patches=max_mask
            )
        else:
            generator = MaskingGenerator3D(mask_window_size, num_masking_patches, min_num_patches=min_mask, max_num_patches=max_mask)
        mask = generator()
        return torch.tensor(mask, dtype=torch.float)

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values
        name = data[0]

        clip = self.read_videos(name)
        if self.data_cfg['motion_aware_masking']:
            frames = clip.permute(0, 2, 3, 1).numpy()
            optical_flow = self.compute_optical_flow(frames)
            optical_flow = resize(optical_flow, (8, 7, 7), anti_aliasing=True)
            mask = self._gen_mask(optical_flow)
        else:
            mask = self._gen_mask(None) 
        return clip, mask

    def __len__(self):
        return len(self.train_labels)
