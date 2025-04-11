import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from modelling.Uniformer_base import Uniformer

class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)


    @torch.no_grad()
    def forward(self, x):
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins # Result is in [-9, 9] for nbins=9

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2]) 
        out = torch.nn.functional.normalize(out, p=2, dim=2)
        return out 

class MSSeparateHead(nn.Module):
    """
    Perform linear projection or Transformer-based decoder (optionally MultiScale)
    for mask prediction models.
    Args:
        head_dim_list (list): List of input dimensions for each head (from encoder blocks).
        cfg: Configuration object (unused in this specific implementation but kept for consistency).
        num_classes (list): List of output dimensions (prediction targets, e.g., HOG features) for each head.
    """
    def __init__(
        self,
        head_dim_list,
        cfg, # cfg seems unused here
        num_classes
    ):
        super(MSSeparateHead, self).__init__()
        self.projections = nn.ModuleList()
        for head_dim, num_class in zip(head_dim_list, num_classes):
            self.projections.append(nn.Linear(head_dim, num_class, bias=True))

        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, block_outputs, output_masks, return_all):
        """
        Forward pass for the multi-scale separate head.
        Args:
            block_outputs (list): List of feature tensors from selected encoder blocks.
            output_masks (list): List of boolean masks corresponding to each block output.
            return_all (bool): If True, project all tokens; otherwise, project only unmasked tokens.
        Returns:
            list: List of projected outputs for each feature level.
        """
        model_outputs = []
        for idx, x in enumerate(block_outputs):
            B, C, T, H, W = x.shape # Batch, Channels, Time, Height, Width
            N = T * H * W 
            x = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
            if not return_all:
                mask = output_masks[idx]
                x = x[mask]
            x = self.projections[idx](x)
            model_outputs.append(x)
        return model_outputs

def calc_uniformer_feature_geometry(cfg):
    """
    Calculates the feature map size and stride at each block of the Uniformer model.
    Args:
        cfg: Configuration object containing DATA and UNIFORMER settings.
    Returns:
        tuple: (feat_sizes, feat_strides)
            - feat_sizes (list): List of [T, H, W] feature map sizes for each block.
            - feat_strides (list): List of [stride_T, stride_H, stride_W] relative to input for each block.
    """
    T0 = 16
    S0 = cfg.DATA.TRAIN_CROP_SIZE
    # Stage 1: patch_embed1 (patch_size=4, time stride=2, spatial stride=4)
    T1 = T0 // 2
    S1 = S0 // 4
    # Stage 2: patch_embed2 (patch_size=2, spatial stride=2)
    T2 = T1
    S2 = S1 // 2
    # Stage 3: patch_embed3 (patch_size=2, spatial stride=2)
    T3 = T2
    S3 = S2 // 2
    # Stage 4: patch_embed4 (patch_size=2, spatial stride=2)
    T4 = T3 
    S4 = S3 // 2

    depths = cfg.UNIFORMER.DEPTH # e.g., [3, 4, 8, 3]

    feat_sizes = []
    feat_strides = []
    # Stage 1 blocks
    for i in range(depths[0]):
        feat_sizes.append([T1, S1, S1])
        feat_strides.append([2, 4, 4]) # Stride relative to input
    # Stage 2 blocks
    for i in range(depths[1]):
        feat_sizes.append([T2, S2, S2])
        feat_strides.append([2, 8, 8]) # Stride relative to input
    # Stage 3 blocks
    for i in range(depths[2]):
        feat_sizes.append([T3, S3, S3])
        feat_strides.append([2, 16, 16]) # Stride relative to input
    # Stage 4 blocks
    for i in range(depths[3]):
        feat_sizes.append([T4, S4, S4])
        feat_strides.append([2, 32, 32]) # Stride relative to input
    return feat_sizes, feat_strides


class MaskUniformer(Uniformer):
    """
    MaskUniformer model inherits from Uniformer and adds the MaskFeat mechanism:
      – Removes blocks after pretrain_depth (according to MASK.PRETRAIN_DEPTH config).
      – Removes the classification head and norm.
      – Adds MaskFeat components: HOG prediction via a separate head (MSSeparateHead),
        mask token, and input processing using the masking mechanism.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.pretrain_depth = cfg.MASK.PRETRAIN_DEPTH
        head_dim_list = []
        block_lists = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        stage_counts = [len(blocks) for blocks in block_lists] # Use len(blocks) instead of cfg.UNIFORMER.DEPTH for safety

        for d in cfg.MASK.PRETRAIN_DEPTH:
            cumulative = 0
            block_dim = None
            found = False
            for stage_idx, (stage_blocks, count) in enumerate(zip(block_lists, stage_counts)):
                if d < cumulative + count:
                    block_index_in_stage = d - cumulative
                    block = stage_blocks[block_index_in_stage]

                    if hasattr(block, 'dim_out') and block.dim_out is not None:
                         block_dim = block.dim_out
                    elif hasattr(block, 'norm2'):
                        if isinstance(block.norm2, nn.BatchNorm3d):
                            block_dim = block.norm2.num_features
                        elif isinstance(block.norm2, nn.LayerNorm):
                            if isinstance(block.norm2.normalized_shape, (int, float)):
                                block_dim = int(block.norm2.normalized_shape)
                            elif isinstance(block.norm2.normalized_shape, (list, tuple)) and len(block.norm2.normalized_shape) > 0:
                                block_dim = block.norm2.normalized_shape[-1]
                            else: block_dim = None
                        else:
                            block_dim = None 
                    elif stage_idx < len(block_lists) - 1:
                         if block_index_in_stage == count -1 :
                             next_patch_embed = getattr(self, f'patch_embed{stage_idx+2}', None)
                             if next_patch_embed and hasattr(next_patch_embed, 'proj') and isinstance(next_patch_embed.proj, nn.Conv3d):
                                 block_dim = next_patch_embed.proj.in_channels # Input channels to next stage's conv
                             else: block_dim = None
                         else: block_dim = None 
                    else:
                        block_dim = None 

                    if block_dim is None:
                         if d < stage_counts[0]: embed_idx = 0
                         elif d < stage_counts[0] + stage_counts[1]: embed_idx = 1
                         elif d < stage_counts[0] + stage_counts[1] + stage_counts[2]: embed_idx = 2
                         else: embed_idx = 3
                         try:
                              block_dim = cfg.UNIFORMER.EMBED_DIM[embed_idx]
                              print(f"Warning: Could not directly determine output dim for block {d}. Using cfg.UNIFORMER.EMBED_DIM[{embed_idx}] = {block_dim}")
                         except Exception as e:
                              raise ValueError(f"Cannot determine output dimension for block at depth {d}. Error: {e}")

                    found = True
                    break 
                cumulative += count
            if not found:
                raise ValueError(f"Depth {d} is out of range for the model structure.")
            head_dim_list.append(block_dim)

        if hasattr(self, 'norm'):
            del self.norm
        if hasattr(self, 'head'):
            del self.head
            
        self.feat_size, self.feat_stride = calc_uniformer_feature_geometry(cfg)
        self.head_type = cfg.MASK.HEAD_TYPE.split("_")
        if self.head_type[0] == "separate":
            self.hogs = nn.ModuleList()
            self.nbins = 9 
            self.cell_sz = 8 
            self.hogs.append(HOGLayerC(nbins=self.nbins, pool=self.cell_sz))
            self.ncells = [
                (self.feat_stride[depth][-1] // self.cell_sz) ** 2 
                for depth in self.pretrain_depth
            ]
            pred_hog_classes = [self.nbins * ncell * 3 for ncell in self.ncells]
            self.pred_head = MSSeparateHead(head_dim_list, cfg, pred_hog_classes)
            self.hog_loss = "mse" 
        else:
            # Only 'separate' head type is currently implemented
            raise NotImplementedError("Only 'separate' head type is supported.")

        try:
            first_stage_embed_dim = cfg.UNIFORMER.EMBED_DIM[0]
        except:
            first_stage_embed_dim = 64 
            print(f"Warning: Using default embed_dim {first_stage_embed_dim} for mask token.")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, first_stage_embed_dim)) # Shape [1, 1, C]
        trunc_normal_(self.mask_token, std=0.02) 
        self.pred_hog_wt = 1.0

    def _get_multiscale_mask(self, mask):
        """
        For each depth in pretrain_depth, interpolate the input mask to the spatial size
        of the corresponding feature map (feat_size) and convert to boolean type.
        Args:
            mask (Tensor): The input mask, likely with shape [B, T_mask, H_mask, W_mask].
                           Needs to be float for interpolation.
        Returns:
            list: List of boolean masks, each resized and flattened for a specific pretrain_depth.
                  Each element has shape [B, N_feat], where N_feat = T_feat * H_feat * W_feat.
        """
        output_masks = []
        if mask.dim() == 4: # Assume B, T_mask, H_mask, W_mask
             float_mask = mask.float().unsqueeze(1) # -> B, 1, T_mask, H_mask, W_mask
        elif mask.dim() == 3: # Assume B, H_mask, W_mask (apply spatially only)
             float_mask = mask.float().unsqueeze(1).unsqueeze(1) # -> B, 1, 1, H_mask, W_mask
        else:
             raise ValueError(f"Unsupported input mask dimension: {mask.dim()}")

        for depth in self.pretrain_depth:
            target_size = self.feat_size[depth] # Target size [T, H, W]

            if mask.dim() == 4: # Spatio-temporal mask
                 output_mask = F.interpolate(float_mask, size=target_size, mode='trilinear', align_corners=False)
            else: 
                 spatial_target_size = target_size[1:]
                 temp_mask = float_mask.squeeze(2) # B, 1, H_mask, W_mask
                 spatial_mask = F.interpolate(temp_mask, size=spatial_target_size, mode='bilinear', align_corners=False)
                 output_mask = spatial_mask.unsqueeze(2).expand(-1, -1, target_size[0], -1, -1) # B, 1, T_feat, H_feat, W_feat
                
            output_mask = output_mask.squeeze(1).flatten(1).to(torch.bool) # [B, T_feat * H_feat * W_feat]
            output_masks.append(output_mask)
        return output_masks

    def _get_hog_label_3d(self, input_frames, output_masks):
        """
        Calculate 3D HOG label as in MaskMViT.
        Args:
            input_frames (Tensor): Original video tensor (B, C, T, H, W).
            output_masks (list): List of boolean masks, resized for each prediction depth.
        Returns:
            list: List of tuples, each containing (hog_labels, weight, loss_type) for a prediction depth.
        """
        stride = 2
        input_frames = input_frames[:, :, ::stride, :, :]
        input_frames = input_frames.transpose(1, 2)
        B, T_new = input_frames.shape[:2]
        input_frames = input_frames.flatten(0, 1)

        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            feat_spatial_size = self.feat_size[depth][-1] # H' or W'
            feat_temporal_size = self.feat_size[depth][0] # T'

            hog_list = []
            for hog in self.hogs:
                tmp_hog = hog(input_frames).flatten(1, 2)
                unfold_size = tmp_hog.shape[-1] 
                if tmp_hog.shape[-1] % feat_spatial_size != 0 or tmp_hog.shape[-2] % feat_spatial_size != 0:
                     print(f"Warning: HOG output size ({tmp_hog.shape[-2:]}) not perfectly divisible by feature map size ({feat_spatial_size}). Using floor division for unfold size.")
                     unfold_size = tmp_hog.shape[-1]

                # Permute to put H/W dims first for unfolding: [B*T', H_hog, W_hog, 3*nbins]
                tmp_hog = tmp_hog.permute(0, 2, 3, 1)
                tmp_hog = (
                    tmp_hog.unfold(1, unfold_size, unfold_size) # Unfold H
                           .unfold(2, unfold_size, unfold_size) # Unfold W
                ) # Shape: [B*T', H_feat, W_feat, 3*nbins, unfold_H, unfold_W]
                
                tmp_hog = tmp_hog.flatten(3) # Shape: [B*T', H_feat, W_feat, N_hog_features_per_patch]
                tmp_hog = tmp_hog.view(B, T_new, feat_spatial_size, feat_spatial_size, -1)
                if T_new != feat_temporal_size:
                     tmp_hog_reshaped = tmp_hog.permute(0, 4, 1, 2, 3).flatten(3)
                     tmp_hog_interp = F.interpolate(tmp_hog_reshaped, size=(feat_temporal_size, feat_spatial_size*feat_spatial_size), mode='trilinear', align_corners=False)
                     tmp_hog = tmp_hog_interp.view(B, -1, feat_temporal_size, feat_spatial_size, feat_spatial_size).permute(0, 2, 3, 4, 1)


                # Flatten spatio-temporal dimensions: [B, T_feat * H_feat * W_feat, N_hog_features]
                tmp_hog = tmp_hog.flatten(1, 3)
                # output_mask has shape [B, T_feat * H_feat * W_feat]
                tmp_hog = tmp_hog[output_mask]

                hog_list.append(tmp_hog)
                
            all_tlabel = torch.cat(hog_list, -1)
            labels.append((all_tlabel, self.pred_hog_wt, self.hog_loss))
        return labels

    def _maskfeat_forward(self, x, mask, return_all=False):
        """
        Forward pass implementing the MaskFeat logic.
        Args:
            x (Tensor): Input video tensor [B, C, T, H, W].
            mask (Tensor): Input mask tensor [B, T_mask, H_mask, W_mask] or [B, H_mask, W_mask].
            return_all (bool): If True, return predictions for all tokens, otherwise only for masked tokens.
        Returns:
            tuple: (model_outputs, labels)
                - model_outputs (list): List of predictions from the head(s).
                - labels (list): List of target labels (e.g., HOG features) for loss calculation.
        """
        raw_x = x.clone() 
        x = self.patch_embed1(x)
        B, C, T, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) # Shape: [B, N, C]
        if mask.dim() == 4: 
            float_mask = mask.float().unsqueeze(1) 
            interp_mode = 'trilinear'
            target_size = (T, H, W)
        elif mask.dim() == 3: 
            float_mask = mask.float().unsqueeze(1).unsqueeze(1) 
            interp_mode = 'bilinear' 
            target_size = (H, W)
            float_mask = F.interpolate(float_mask.squeeze(2), size=target_size, mode=interp_mode, align_corners=False)
            float_mask = float_mask.unsqueeze(2).expand(-1, -1, T, -1, -1) # B, 1, T, H, W
        else:
            raise ValueError(f"Unsupported input mask dimension: {mask.dim()}")

        if mask.dim() == 4:
             float_mask = F.interpolate(float_mask, size=target_size, mode=interp_mode, align_corners=False)
            
        float_mask = float_mask.squeeze(1).flatten(1).unsqueeze(-1) # Shape: [B, N, 1]
        output_masks = self._get_multiscale_mask(mask)
        labels = self._get_hog_label_3d(raw_x.detach(), output_masks) # Detach raw_x as it's only for label calculation
        if self.mask_token.shape[-1] != C:
             if not hasattr(self, 'mask_token_proj'):
                  print(f"Warning: Mask token dim ({self.mask_token.shape[-1]}) != Token dim ({C}). Adding linear projection.")
                  self.mask_token_proj = nn.Linear(self.mask_token.shape[-1], C).to(self.mask_token.device)
             proj_mask_token = self.mask_token_proj(self.mask_token)
        else:
             proj_mask_token = self.mask_token

        mask_tokens = proj_mask_token.expand(B, tokens.size(1), -1) # Expand to [B, N, C]
        # float_mask is 1 for masked, 0 for unmasked.
        tokens = tokens * (1 - float_mask) + mask_tokens * float_mask
        
        x = tokens.transpose(1, 2).reshape(B, C, T, H, W)

        block_outputs = []
        current_depth = 0 

        # Stage 1
        for i, blk in enumerate(self.blocks1):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # Stage 2
        if hasattr(self, 'patch_embed2'):
            x = self.patch_embed2(x)
        for i, blk in enumerate(self.blocks2):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # Stage 3 
        if hasattr(self, 'patch_embed3'):
            x = self.patch_embed3(x)
        for i, blk in enumerate(self.blocks3):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # Stage 4 
        if hasattr(self, 'patch_embed4'):
            x = self.patch_embed4(x)
        for i, blk in enumerate(self.blocks4):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        model_outputs = []
        if self.pred_hog_wt > 0:
            hog_outputs = self.pred_head(block_outputs, output_masks, return_all)
            model_outputs += hog_outputs # Add HOG predictions to the output list
        return model_outputs, labels

    def forward(self, x, return_all=False):
        """
        Main forward pass for MaskUniformer.
        Expected input is a tuple or list containing (video, mask) where
           video: tensor (B, C, T, H, W)
           mask: tensor (B, T_mask, H_mask, W_mask) or (B, H_mask, W_mask)
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            video, mask = x[0], x[1]
        elif isinstance(x, torch.Tensor) and hasattr(self, 'mask_generator'):
            video = x
            mask = self.mask_generator(video) # Example placeholder
            print("Warning: Input only contained video tensor. Generated mask using self.mask_generator.")
        else:
            raise ValueError("Input must be a tuple/list (video, mask) or just video if mask generation is handled internally.")

        # Call the MaskFeat-specific forward method
        return self._maskfeat_forward(video, mask, return_all)


def build_mask_uniformer_small(num_classes=174, pretrained=True, pretrained_name='uniformer_small_k400_16x4', device='cpu'):
    """
    Build the MaskUniformer Small model with default parameters similar to build_uniformer_small.
    – Model parameters (embed_dim, depth, …) are kept the same as uniformer_small.
    – A dummy config is built using the fields: DATA, UNIFORMER, MASK, and MODEL.
    – If pretrained=True, load pretrained weights using the name pretrained_name (from the base Uniformer).
    Args:
        num_classes (int): Number of classes for the original classification task (used by base Uniformer init, but head is deleted).
        pretrained (bool): Whether to load pretrained weights from base Uniformer.
        pretrained_name (str): Name of the pretrained model checkpoint file (without extension).
        device (str): Device to load the model onto ('cpu' or 'cuda').
    Returns:
        MaskUniformer: The constructed MaskUniformer model.
    """
    class Cfg:
        pass
    cfg = Cfg()

    # --- DATA Configuration ---
    cfg.DATA = Cfg()
    cfg.DATA.TRAIN_CROP_SIZE = 224 
    cfg.DATA.INPUT_CHANNEL_NUM = [3] 

    # --- UNIFORMER Configuration (based on uniformer_small) ---
    cfg.UNIFORMER = Cfg()
    cfg.UNIFORMER.EMBED_DIM = [64, 128, 320, 512] 
    cfg.UNIFORMER.DEPTH = [3, 4, 8, 3] 
    cfg.UNIFORMER.HEAD_DIM = 64
    cfg.UNIFORMER.MLP_RATIO = 4
    cfg.UNIFORMER.QKV_BIAS = True 
    cfg.UNIFORMER.QKV_SCALE = None
    cfg.UNIFORMER.REPRESENTATION_SIZE = None 
    cfg.UNIFORMER.DROPOUT_RATE = 0.0 
    cfg.UNIFORMER.ATTENTION_DROPOUT_RATE = 0.0
    cfg.UNIFORMER.DROP_DEPTH_RATE = 0.1 
    cfg.UNIFORMER.SPLIT = False 
    cfg.UNIFORMER.STD = False 
    cfg.UNIFORMER.PRETRAIN_NAME = pretrained_name if pretrained else '' 

    # --- MASK Configuration (for MaskFeat) ---
    cfg.MASK = Cfg()
    cfg.MASK.PRETRAIN_DEPTH = [17] # Use output of the very last block
    cfg.MASK.HEAD_TYPE = "separate" # Type of prediction head

    # --- MODEL Configuration ---
    cfg.MODEL = Cfg()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.MODEL.USE_CHECKPOINT = False 
    cfg.MODEL.CHECKPOINT_NUM = [0, 0, 0, 0]

    model = MaskUniformer(cfg)

    if pretrained:
        checkpoint = model.get_pretrained_model(cfg) 
        if checkpoint is not None:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print("Missing keys:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys (likely classification parts):", unexpected_keys)
            print(f"Successfully loaded pretrained weights from '{pretrained_name}' (with expected mismatches for MaskFeat parts).")
        else:
            print(f"Warning: Pretrained weights '{pretrained_name}' not found or loaded.")

    model.to(device)
    return model
