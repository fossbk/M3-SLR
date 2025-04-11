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
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)

        return out  # B 3 nbins H W

class MSSeparateHead(nn.Module):
    """
    Perform linear projection or Transformer-based decoder (optionally MultiScale)
    for mask prediction models.
    Args:
        blocks (MultiScaleBlock): the encoder blocks to provide input dimensions of the head.
        num_classes (int): the dimension of the prediction target (eg. HOG or pixels).
        feat_sz (list): the spatiotemporal sizes of the input features.
    """

    def __init__(
        self,
        head_dim_list,
        cfg,
        num_classes
    ):
        super(MSSeparateHead, self).__init__()
        self.projections = nn.ModuleList()
        for head_dim, num_class in zip(head_dim_list, num_classes):
            self.projections.append(nn.Linear(head_dim, num_class, bias=True))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # FYI: MAE uses xavier_uniform following official JAX ViT:
            # torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, block_outputs, output_masks, return_all):
        model_outputs = []
        for idx, x in enumerate(block_outputs):
            B, C, T, H, W = x.shape
            N = T * H * W
            x = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)  # [B, C, T, H, W] -> [B, N, C]
            if not return_all:
                mask = output_masks[idx]
                x = x[mask]  # [sum(mask), C]
            x = self.projections[idx](x)
            model_outputs.append(x)
        return model_outputs
    
def calc_uniformer_feature_geometry(cfg):
    # Lấy các thông số từ config
    T0 = 16
    S0 = cfg.DATA.TRAIN_CROP_SIZE  # kích thước không gian ban đầu
    # Stage 1: patch_embed1 (patch_size=4, stride thời gian=2, stride không gian=4)
    T1 = T0 // 2
    S1 = S0 // 4
    # Stage 2: patch_embed2 (patch_size=2, stride không gian=2)
    T2 = T1
    S2 = S1 // 2
    # Stage 3: patch_embed3 (patch_size=2, stride không gian=2)
    T3 = T2
    S3 = S2 // 2
    # Stage 4: patch_embed4 (patch_size=2, stride không gian=2)
    T4 = T3
    S4 = S3 // 2
    # Với cấu hình Uniformer có DEPTH = [d1, d2, d3, d4]
    depths = cfg.UNIFORMER.DEPTH  # ví dụ: [3, 4, 8, 3]
    feat_sizes = []
    feat_strides = []
    # Gán geometry cho từng block của từng stage
    for i in range(depths[0]):
        feat_sizes.append([T1, S1, S1])
        feat_strides.append([2, 4, 4])
    for i in range(depths[1]):
        feat_sizes.append([T2, S2, S2])
        feat_strides.append([2, 8, 8])
    for i in range(depths[2]):
        feat_sizes.append([T3, S3, S3])
        feat_strides.append([2, 16, 16])
    for i in range(depths[3]):
        feat_sizes.append([T4, S4, S4])
        feat_strides.append([2, 32, 32])
    return feat_sizes, feat_strides


class MaskUniformer(Uniformer):
    """
    Mô hình MaskUniformer kế thừa từ Uniformer và bổ sung cơ chế MaskFeat:
      – Cắt bỏ các block sau pretrain_depth (theo config MASK.PRETRAIN_DEPTH)
      – Xóa bỏ head và norm dùng cho phân loại
      – Thêm các thành phần của MaskFeat: dự đoán HOG qua head riêng (MSSeparateHead),
        mask token, và xử lý đầu vào theo cơ chế masking.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # Lấy pretrain depth từ config (ví dụ: [15])
        self.pretrain_depth = cfg.MASK.PRETRAIN_DEPTH
        total_depth = sum(cfg.UNIFORMER.DEPTH)  # tổng số block của Uniformer

        # Giả sử cfg.UNIFORMER.DEPTH = [3, 4, 8, 3] và cfg.MASK.PRETRAIN_DEPTH là danh sách chỉ số (ví dụ [17])
        head_dim_list = []
        # Lưu danh sách các stage
        block_lists = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        # Lưu số lượng block trong mỗi stage
        stage_counts = [len(self.blocks1), len(self.blocks2), len(self.blocks3), len(self.blocks4)]

        for d in cfg.MASK.PRETRAIN_DEPTH:
            cumulative = 0
            block_dim = None
            for stage, count in zip(block_lists, stage_counts):
                if d < cumulative + count:
                    # Tìm được block tương ứng: index của block trong stage là (d - cumulative)
                    block = stage[d - cumulative]
                    # Nếu block chưa có thuộc tính dim_out, ta cố gắng xác định qua norm2
                    if not hasattr(block, 'dim_out') or block.dim_out is None:
                        if hasattr(block, 'norm2'):
                            if isinstance(block.norm2, nn.BatchNorm3d):
                                block_dim = block.norm2.num_features
                            elif isinstance(block.norm2, nn.LayerNorm):
                                block_dim = block.norm2.normalized_shape[0]
                            else:
                                block_dim = None
                        else:
                            block_dim = None
                    else:
                        block_dim = block.dim_out
                    break  # thoát vòng lặp khi đã tìm thấy block phù hợp
                cumulative += count
            head_dim_list.append(block_dim)

        del self.norm
        del self.head

        self.feat_size, self.feat_stride = calc_uniformer_feature_geometry(cfg)
        self.head_type = cfg.MASK.HEAD_TYPE.split("_")
        if self.head_type[0] == "separate":
            # Thiết lập head riêng dùng cho MaskFeat dựa vào HOG
            self.hogs = nn.ModuleList()
            self.nbins = 9
            self.cell_sz = 8
            self.hogs.append(HOGLayerC(nbins=self.nbins, pool=self.cell_sz))
            # Tính số cell dựa trên feature stride (chỉ dùng phần cuối)
            self.ncells = [
                (self.feat_stride[depth][-1] // self.cell_sz) ** 2
                for depth in self.pretrain_depth
            ]
            # Số lớp dự đoán: nbins * số cell * 3 (với 3 kênh màu)
            pred_hog_classes = [self.nbins * ncell * 3 for ncell in self.ncells]
            self.pred_head = MSSeparateHead(head_dim_list, cfg, pred_hog_classes)
            self.hog_loss = "mse"
        else:
            raise NotImplementedError("Chỉ hỗ trợ head kiểu 'separate'.")

        # Khởi tạo mask token với embedding dimension của block cuối (theo Uniformer)
        embed_dim = 64
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=0.02)
        self.pred_hog_wt = 1.0

    def _get_multiscale_mask(self, mask):
        """
        Với mỗi depth trong pretrain_depth, nội suy lại mask về kích thước không gian (square)
        dựa trên feat_size của block tương ứng và chuyển sang kiểu bool.
        """
        output_masks = []
        for depth in self.pretrain_depth:
            # feat_size[depth] có dạng [..., spatial_size]
            size = self.feat_size[depth][-1]
            output_mask = F.interpolate(mask, size=(size, size), mode='nearest')
            output_mask = output_mask.flatten(1).to(torch.bool)
            output_masks.append(output_mask)
        return output_masks

    def _get_hog_label_3d(self, input_frames, output_masks):
        """
        Tính label HOG 3D như trong MaskMViT.
        input_frames: tensor video (B, C, T, H, W)
        output_masks: danh sách các mask đã được tính theo từng scale.
        """
        # Giảm mẫu theo thời gian với khoảng cách bằng cfg.MVIT.PATCH_STRIDE[0]
        stride = 2
        input_frames = input_frames[:, :, ::stride, :, :]
        input_frames = input_frames.transpose(1, 2)  # (B, T, C, H, W)
        B, T = input_frames.shape[:2]
        input_frames = input_frames.flatten(0, 1)  # (B*T, C, H, W)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            feat_size = self.feat_size[depth][-1]
            hog_list = []
            for hog in self.hogs:
                tmp_hog = hog(input_frames).flatten(1, 2)
                unfold_size = tmp_hog.shape[-1] // feat_size
                tmp_hog = (
                    tmp_hog.permute(0, 2, 3, 1)
                    .unfold(1, unfold_size, unfold_size)
                    .unfold(2, unfold_size, unfold_size)
                )
                tmp_hog = tmp_hog.flatten(3).view(B, T, feat_size, feat_size, -1)
                tmp_hog = tmp_hog.flatten(1, 3)
                tmp_hog = tmp_hog[output_mask]
                hog_list.append(tmp_hog)
            all_tlabel = torch.cat(hog_list, -1)
            labels.append((all_tlabel, self.pred_hog_wt, self.hog_loss))
        return labels

    def _maskfeat_forward(self, x, mask, return_all=False):
        # Lưu lại raw video input trước patch embedding, để dùng cho tính HOG label.
        raw_x = x.clone()  # raw_x có shape [B, 3, T, H, W]
        
        # 1. Tính patch embedding ban đầu, x có shape [B, C, T, H, W]
        x = self.patch_embed1(x)
        B, C, T, H, W = x.shape

        # 2. Chuyển x thành dạng token [B, N, C] với N = T*H*W
        tokens = x.flatten(2).transpose(1, 2)

        # 3. Nội suy mask: giả sử mask ban đầu có shape [B, T, H_mask, W_mask]
        # Ở đây, mask được tạo bởi MaskingGenerator3D có shape [B, 8, 7, 7]
        float_mask = mask.type_as(tokens)
        # Nội suy mask thành [B, T, H, W] để khớp với feature map của patch_embed1
        float_mask = F.interpolate(float_mask.unsqueeze(1), size=(T, H, W), mode='nearest').squeeze(1)
        # Flatten mask thành [B, N, 1]
        float_mask = float_mask.flatten(1).unsqueeze(-1)

        # 4. Tính output mask cho các scale khác nhau
        output_masks = self._get_multiscale_mask(mask)

        # 5. Tính label HOG 3D sử dụng raw video input
        labels = self._get_hog_label_3d(raw_x.detach(), output_masks)

        # 6. Tạo mask token có shape [B, N, C] và thay thế các token được mask
        mask_tokens = self.mask_token.expand(B, tokens.size(1), -1)
        tokens = tokens * (1 - float_mask) + mask_tokens * float_mask

        # 7. Chuyển tokens từ [B, N, C] trở lại dạng 5D [B, C, T, H, W]
        x = tokens.transpose(1, 2).view(B, C, T, H, W)

        # 8. Chạy qua các block và thu thập output của các block nằm trong pretrain_depth
        block_outputs = []
        current_depth = 0  # Theo dõi index của block qua tất cả các stage

        # Stage 1
        for i, blk in enumerate(self.blocks1):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # Stage 2
        x = self.patch_embed2(x)
        for i, blk in enumerate(self.blocks2):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # Stage 3
        x = self.patch_embed3(x)
        for i, blk in enumerate(self.blocks3):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # Stage 4
        x = self.patch_embed4(x)
        for i, blk in enumerate(self.blocks4):
            x = blk(x)
            if current_depth in self.pretrain_depth:
                block_outputs.append(x)
            current_depth += 1

        # 9. Tạo model outputs
        model_outputs = []
        if self.pred_hog_wt:
            hog_outputs = self.pred_head(block_outputs, output_masks, return_all)
            model_outputs += hog_outputs

        return model_outputs, labels

    def forward(self, x, return_all=False):
        """
        Đầu vào dự kiến là tuple hoặc list gồm (video, mask) với
           video: tensor (B, C, T, H, W)
           mask: tensor (B, H_orig, W_orig)
        """
        if isinstance(x, (list, tuple)) and len(x) > 1:
            video, mask = x[0], x[1]
        else:
            raise ValueError("Input phải là (video, mask)")
        return self._maskfeat_forward(video, mask, return_all)


def build_mask_uniformer_small(num_classes=174, pretrained=True, pretrained_name='uniformer_small_k400_16x4', device='cpu'):
    """
    Xây dựng mô hình MaskUniformer Small với các thông số mặc định tương tự build_uniformer_small.
    – Các thông số của mô hình (embed_dim, depth, …) được giữ nguyên.
    – Config ảo được xây dựng theo các field:
         DATA, UNIFORMER, MASK, MVIT và MODEL.
    – Nếu pretrained=True, tải trọng số pretrained theo tên pretrained_name.
    """
    class Cfg: 
        pass
    cfg = Cfg()

    # DATA
    cfg.DATA = Cfg()
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3]

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
    if pretrained:
        cfg.UNIFORMER.PRETRAIN_NAME = pretrained_name
    else:
        cfg.UNIFORMER.PRETRAIN_NAME = ''

    # MASK config
    cfg.MASK = Cfg()
    cfg.MASK.PRETRAIN_DEPTH = [17]
    cfg.MASK.HEAD_TYPE = "separate"

    # MODEL
    cfg.MODEL = Cfg()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.MODEL.USE_CHECKPOINT = False
    cfg.MODEL.CHECKPOINT_NUM = [0, 0, 0, 0]

    # Xây dựng mô hình MaskUniformer
    model = MaskUniformer(cfg)

    if pretrained:
        checkpoint = model.get_pretrained_model(cfg)
        if checkpoint is not None:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            if len(missing) > 0:
                print("Các tham số chưa khớp:", missing)
            if len(unexpected) > 0:
                print("Các tham số thừa không dùng:", unexpected)
        print("Load pretrained weight successfully!")
    model.to(device)
    return model

