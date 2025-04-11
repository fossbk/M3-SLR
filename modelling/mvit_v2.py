from modelling.mvit_v2_utils import *

def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) :
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

class MViT(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        temporal_size: int,
        block_setting: Sequence[MSBlockConfig],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        patch_embed_kernel: Tuple[int, int, int] = (3, 7, 7),
        patch_embed_stride: Tuple[int, int, int] = (2, 4, 4),
        patch_embed_padding: Tuple[int, int, int] = (1, 3, 3),
    ) :
        """
        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                the class embedding.
            rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
            proj_after_attn (bool): If True, apply the projection after the attention.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
            patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
            patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
            patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        """
        super().__init__()
        print("Model: MViT_V2")
        # This implementation employs a different parameterization scheme than the one used at PyTorch Video:
        # https://github.com/facebookresearch/pytorchvideo/blob/718d0a4/pytorchvideo/models/vision_transformers.py
        # We remove any experimental configuration that didn't make it to the final variants of the models. To represent
        # the configuration of the architecture we use the simplified form suggested at Table 1 of the paper.
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding module
        self.conv_proj = nn.Conv3d(
            in_channels=3,
            out_channels=block_setting[0].input_channels,
            kernel_size=patch_embed_kernel,
            stride=patch_embed_stride,
            padding=patch_embed_padding,
        )

        input_size = [size // stride for size, stride in zip((temporal_size,) + spatial_size, self.conv_proj.stride)]

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[1], input_size[2]),
            temporal_size=input_size[0],
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )

            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
        self.norm = norm_layer(block_setting[-1].output_channels)

        print("Num Classes",num_classes)
        self.num_classes = num_classes
        self.dropout = dropout
        self.block_setting = block_setting
        # Classifier module
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(block_setting[-1].output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def reset_head(self,num_classes):
        self.head = nn.Sequential(
            nn.Dropout(self.dropout, inplace=True),
            nn.Linear(self.block_setting[-1].output_channels, num_classes),
        )
        print("Reset to",num_classes)
    
    def forward_features(self,clip):
        x = clip
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def forward(self, clip: torch.Tensor, gloss=None) -> torch.Tensor:
        x = clip
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.head(x)

        return {
            'logits':x
        }
    

def mvit_v2_s(**kwargs):
    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )
    
    return MViT(
        spatial_size=(224, 224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        num_classes = kwargs.pop('num_classes',400),
        **kwargs,
    )


class MVitV2_ThreeView(nn.Module):
    def __init__(self,dropout = 0.5, **kwargs):
        super(MVitV2_ThreeView, self).__init__()
        print("Model: MVitV2_ThreeView")
        self.center = mvit_v2_s( **kwargs)
        self.left = mvit_v2_s( **kwargs)
        self.right = mvit_v2_s( **kwargs)
        num_classes = kwargs.pop('num_classes',199)
        print("Num classes",num_classes)
        self.classififer = nn.Sequential(
            nn.Linear(3*768,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )

        

    def remove_head(self):
        print("Remove head")
        self.center.head = nn.Identity()
        self.left.head = nn.Identity()
        self.right.head = nn.Identity()
    
    def count(self):
           
        total = sum(p.numel() for p in self.parameters())
        print(f"Total params: {total}")

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {num_trainable_params}")
        
        print("Trainable params / Total params",num_trainable_params/total)
    
    def freeze_and_remove(self,layers = 5):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.center.blocks[i].parameters():
                param.requires_grad = False
            for param in self.left.blocks[i].parameters():
                param.requires_grad = False
            for param in self.right.blocks[i].parameters():
                param.requires_grad = False

    def forward_features(self, left,center,right):
        b,t,c,h,w = left.shape
        left_ft = self.left.forward_features(left.permute(0,2,1,3,4))
        center_ft = self.center.forward_features(center.permute(0,2,1,3,4))
        right_ft = self.right.forward_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)

        return output_features
    def forward(self, left,center,right):    
        b,t,c,h,w = left.shape
        left_ft = self.left.forward_features(left.permute(0,2,1,3,4))
        center_ft = self.center.forward_features(center.permute(0,2,1,3,4))
        right_ft = self.right.forward_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)

        y = self.classififer(output_features)

        return {
            'logits':y
        }
        


class MVitV2_HandCrop(nn.Module):
    def __init__(self,dropout = 0.5, **kwargs):
        super(MVitV2_HandCrop, self).__init__()
        self.left = mvit_v2_s( **kwargs)
        self.right = mvit_v2_s( **kwargs)
        print("Model: MVitV2_HandCrop")
        
        num_classes = kwargs.pop('num_classes',199)
        print("Num classes",num_classes)
        self.classififer = nn.Sequential(
            nn.Linear(2*768,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )

    def remove_head(self):
        print("Remove head")
        self.left.head = nn.Identity()
        self.right.head = nn.Identity()
    
    def freeze_and_remove(self,layers = 5):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.left.blocks[i].parameters():
                param.requires_grad = False
            for param in self.right.blocks[i].parameters():
                param.requires_grad = False
       
        
    def forward(self, clip = None,poseflow = None,**kwargs):    
        b,t,x,c,h,w = clip.shape
        left = clip[:,:,0] # b,t,c,h,w
        right = clip[:,:,1] # b,t,c,h,w
        left_ft = self.left.forward_features(left.permute(0,2,1,3,4))
        right_ft = self.right.forward_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,right_ft],dim = -1)

        y = self.classififer(output_features)

        return {
            'logits':y
        }
        

class MvitV2_OneView_Sim_Knowledge_Distillation(nn.Module):
    def __init__(self, num_classes=226, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: MvitV2_OneView_Sim_Knowledge_Distilation")
        self.teacher = MVitV2_ThreeView(num_classes = num_classes)
        self.teacher.remove_head()  
        self.teacher.load_state_dict(torch.load("checkpoints/MVitV2_ThreeView/MVIT V2 Small for three view finetune from one view/best_checkpoints.pth",map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = mvit_v2_s(num_classes = num_classes)
        state_dict = torch.load('checkpoints/mvit_v2/MVIT V2 Small for one view of AUTSL pretrained on kinetics400/autsl_best_checkpoints.pth',map_location=torch.device('cpu'))
        self.student.reset_head(226) # AUTSL
        self.student.load_state_dict(state_dict)
        self.student.head = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(768,768*6),
            nn.LayerNorm(768*6),
            nn.LeakyReLU(),
            nn.Linear(768*6,768*3)
        )

    def forward(self,left = None,center = None,right = None): 
        self.teacher.eval()
        teacher_features = None
        y = None

        teacher_features = self.teacher.forward_features(left = left,center = center,right = right)
        
        student_features = self.student.forward_features(clip=center.permute(0,2,1,3,4))

        student_features = self.projection(student_features)
        
        if not self.training:
            y = self.teacher.classififer(student_features)
        
        return {
            'logits':y,
            'student_features': student_features,
            'teacher_features': teacher_features,
        }

class MvitV2_OneView_Sim_Knowledge_Distillation_Inference(nn.Module):
    def __init__(self, num_classes=226, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: MvitV2_OneView_Sim_Knowledge_Distillation_Inference")
        print("*"*20)
        self.student = mvit_v2_s(num_classes = num_classes)
        self.student.head = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(768,768*6),
            nn.LayerNorm(768*6),
            nn.LeakyReLU(),
            nn.Linear(768*6,768*3)
        )
        num_classes = kwargs.pop('num_classes',199)
        print("Num classes",num_classes)
        self.classififer = nn.Sequential(
            nn.Linear(3*768,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )
        print("*"*20)

    def forward(self,clip): 
        center = clip
        b,c,t,h,w = center.shape
        assert c  == 3
        student_features = self.student.forward_features(clip=center)

        student_features = self.projection(student_features)
       
        y = self.classififer(student_features)
        
        return {
            'logits':y
        }

class MVitV2_ThreeView_ShareWeights(nn.Module):
    def __init__(self,dropout = 0.5, **kwargs):
        super(MVitV2_ThreeView_ShareWeights, self).__init__()
        print("Model: MVitV2_ThreeView_ShareWeights")
        print("*"*20)
        self.encoder = mvit_v2_s( **kwargs)
        self.left_pj = nn.Linear(768,768)
        self.right_pj = nn.Linear(768,768)
        self.center_pj = nn.Linear(768,768)
        num_classes = kwargs.pop('num_classes',199)
        print("Num classes",num_classes)
        self.classififer = nn.Sequential(
            nn.Linear(3*768,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )
        print("*"*20)
        

    def remove_head(self):
        self.encoder.head = nn.Identity()
        print("Remove head")
    
    def freeze_layers(self,layers = 3):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.encoder.blocks[i].parameters():
                param.requires_grad = False
    
    def count(self):
           
        total = sum(p.numel() for p in self.parameters())
        print(f"Total params: {total}")

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {num_trainable_params}")
        
        print("Trainable params / Total params",num_trainable_params/total)
    
   
    def forward(self, left,center,right):    
        b,t,c,h,w = left.shape
        left_ft = self.encoder.forward_features(left.permute(0,2,1,3,4))
        center_ft = self.encoder.forward_features(center.permute(0,2,1,3,4))
        right_ft = self.encoder.forward_features(right.permute(0,2,1,3,4))

        # fc learn corresponding views
        left_ft = self.left_pj(left_ft)
        right_ft = self.right_pj(right_ft)
        center_ft = self.center_pj(center_ft)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)

        y = self.classififer(output_features)

        return {
            'logits':y
        }

