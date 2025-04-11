from modelling.swin_transformer_utils import *
from torchvision import models
from modelling.mvit_v2 import MVitV2_ThreeView

class SwinTransformer3d(nn.Module):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 400.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """

    def __init__(
        self,
        patch_size: List[int] = [2,4,4],
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = [8,7,7],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 199,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        patch_embed: Optional[Callable[..., nn.Module]] = None,
    ) :
        super().__init__()
        self.num_classes = num_classes
        print("Model: SwinTransformer3d")
        if block is None:
            block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if patch_embed is None:
            patch_embed = PatchEmbed3d

        # split image into non-overlapping patches
        self.patch_embed = patch_embed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=dropout)

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttention3d,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def reset_head(self,num_classes):
        self.head = nn.Linear(self.num_features, num_classes) 
        print("Reset to",num_classes)       
    
    def forward_features(self,clip):
        x = self.patch_embed(clip)  # B _T _H _W C
        x = self.pos_drop(x)
        x = self.features(x)  # B _T _H _W C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    def forward(self, clip: Tensor, gloss=None) :
        
        x = self.patch_embed(clip)  # B _T _H _W C
        x = self.pos_drop(x)
        x = self.features(x)  # B _T _H _W C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return {
            'logits':x
        }


class SwinTransformer3d_ThreeView(nn.Module):
    def __init__(self,dropout = 0, **kwargs):
        super(SwinTransformer3d_ThreeView, self).__init__()
        print("Model: SwinTransformer3d_ThreeView")
        self.center = SwinTransformer3d( **kwargs)
        self.left = SwinTransformer3d( **kwargs)
        self.right = SwinTransformer3d( **kwargs)
        
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
    
    def freeze_and_remove(self,layers = 2):
        print(f"Freeze {layers} block attn")  
        for i in range(layers):
            for param in self.center.features[i].parameters():
                param.requires_grad = False
            for param in self.left.features[i].parameters():
                param.requires_grad = False
            for param in self.right.features[i].parameters():
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
        

class SwinTransformer3d_HandCrop(nn.Module):
    def __init__(self,dropout = 0, **kwargs):
        super(SwinTransformer3d_HandCrop, self).__init__()
        print("Model: SwinTransformer3d_HandCrop")
        self.left = SwinTransformer3d( **kwargs)
        self.right = SwinTransformer3d( **kwargs)
        
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
    
    def freeze_and_remove(self,layers = 2):
        print(f"Freeze {layers} block attn")  
        for i in range(layers):
            for param in self.left.features[i].parameters():
                param.requires_grad = False
            for param in self.right.features[i].parameters():
                param.requires_grad = False
       
       
        
    def forward(self, clip = None,poseflow = None,**kwargs):    
        b,t,x,c,h,w = clip.shape
        left = clip[:,:,0]
        right = clip[:,:,1]
        left_ft = self.left.forward_features(left.permute(0,2,1,3,4))
        right_ft = self.right.forward_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,right_ft],dim = -1)

        y = self.classififer(output_features)

        return {
            'logits':y
        }
        
class VideoSwinTransformer_OneView_Sim_Knowledge_Distillation(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VideoSwinTransformer_OneView_Sim_Knowledge_Distillation")
      
        self.teacher = MVitV2_ThreeView(num_classes = num_classes)
        self.teacher.remove_head()  
        self.teacher.load_state_dict(torch.load("checkpoints/MVitV2_ThreeView/MVIT V2 Small for three view finetune from one view/best_checkpoints.pth",map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = SwinTransformer3d(num_classes = num_classes)
        state_dict = torch.load('checkpoints/swin_transformer/Swin Transformer 3D Tiny for one view of AUTSL pretrained on kinetics400 /autsl_best_checkpoints.pth',map_location=torch.device('cpu'))
        self.student.reset_head(226) # No.ATUSL glosses
        self.student.load_state_dict(state_dict)
        self.student.head = nn.Identity() # remove head
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
    
class VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference")
        print("*"*20)
        self.student = SwinTransformer3d(num_classes = num_classes)
       
        self.student.head = nn.Identity() # remove head
        self.projection = nn.Sequential(
            nn.Linear(768,768*6),
            nn.LayerNorm(768*6),
            nn.LeakyReLU(),
            nn.Linear(768*6,768*3)
        )

        print("Num classes",num_classes)
        self.classififer = nn.Sequential(
            nn.Linear(3*768,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )
        print("*"*20)
        
    def forward(self,clip = None): 
        center = clip
        b,c,t,h,w = center.shape
        assert c == 3
        
        student_features = self.student.forward_features(clip=center)

        student_features = self.projection(student_features)
    
        y = self.classififer(student_features)
        
        return {
            'logits':y
        }

class SwinTransformer3d_ThreeView_ShareWeights(nn.Module):
    def __init__(self,dropout = 0, **kwargs):
        super(SwinTransformer3d_ThreeView_ShareWeights, self).__init__()
        print("Model: SwinTransformer3d_ThreeView_ShareWeights")
        print("*"*20)
        self.encoder = SwinTransformer3d( **kwargs)
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
        print("Remove head")
        self.encoder.head = nn.Identity()
    
    def freeze_and_remove(self,layers = 2):
        print(f"Freeze {layers} block attn")  
        for i in range(layers):
            for param in self.encoder.features[i].parameters():
                param.requires_grad = False
       
       
        
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