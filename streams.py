"""
streams.py - All 7 Feature Extraction Streams
Each stream is a frozen pretrained backbone.

Stream outputs -
    0  Xception/ResNet  → 2048
    1  EfficientNet-B0  → 1280
    2  F3-Net (DCT)     → 1024
    3  ViT-Base         →  768
    4  Swin-Tiny        →  768
    5  CapsNet          →  256
    6  U-Net Encoder    →  512
    7  Forensics        → 1296 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image

from config import DEVICE, FORENSIC_DIM
from forensics import extract_forensic_features


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


#Spatial CNNs (Xception + EfficientNet)

class SpatialCNNStream(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.xception = timm.create_model(
            'xception', pretrained=True, num_classes=0, global_pool='avg'
        )
        self.efficientnet = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg'
        )
        _freeze(self.xception)
        _freeze(self.efficientnet)
        print('[SpatialCNNStream] Xception(2048) + EfficientNet(1280) loaded & frozen')

    def forward(self, x):
        f_xcep = self.xception(x)      
        f_eff  = self.efficientnet(x)   
        return f_xcep, f_eff


#F3-Net

class DCTLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        N = 8
        basis = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    basis[k, n] = 1.0 / (N ** 0.5)
                else:
                    basis[k, n] = ((2.0 / N) ** 0.5) * torch.cos(
                        torch.tensor(torch.pi * k * (2 * n + 1) / (2 * N))
                    )
        self.register_buffer('basis', basis) 

    def forward(self, x):
        
        B, C, H, W = x.shape
        x_blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)  
        Hb, Wb = x_blocks.shape[2], x_blocks.shape[3]
        x_blocks = x_blocks.contiguous().view(B * C * Hb * Wb, 8, 8)
        dct = self.basis @ x_blocks @ self.basis.t()
        dct = dct.view(B, C, Hb, Wb, 8, 8)
        dct = dct.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        return dct


class F3NetStream(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dct = DCTLayer()

        self.rgb_branch = timm.create_model(
            'efficientnet_b1', pretrained=True, num_classes=0, global_pool='avg'
        )  
        self.dct_branch = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg'
        )  

        _freeze(self.rgb_branch)
        _freeze(self.dct_branch)

        self.proj = nn.Linear(1280 + 1280, 1024)
        nn.init.kaiming_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        _freeze(self.proj)
        print('[F3NetStream] RGB+DCT dual branch → 1024-dim loaded & frozen')

    def forward(self, x):
        f_rgb = self.rgb_branch(x)                  
        dct_x = self.dct(x)                         
        f_dct = self.dct_branch(dct_x)              
        f_cat = torch.cat([f_rgb, f_dct], dim=1)    
        return self.proj(f_cat)                      


# Vision Transformers

class TransformerStream(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=0
        )  
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0
        )  

        _freeze(self.vit)
        _freeze(self.swin)
        print('[TransformerStream] ViT-Base(768) + Swin-Tiny(768) loaded & frozen')

    def forward(self, x):
        f_vit  = self.vit(x)    
        f_swin = self.swin(x)   
        return f_vit, f_swin


# Capsule Network
class SquashFn(nn.Module):
    def forward(self, x, dim=-1):
        norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
        norm    = norm_sq.sqrt()
        return (norm_sq / (1.0 + norm_sq)) * (x / (norm + 1e-8))


class CapsuleLayer(nn.Module):
    
    def __init__(self, in_capsules, in_dim, num_capsules, caps_dim, num_routing=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.caps_dim     = caps_dim
        self.num_routing  = num_routing
        self.W = nn.Parameter(
            torch.randn(in_capsules, num_capsules, caps_dim, in_dim) * 0.01
        )
        self.squash = SquashFn()

    def forward(self, x):
        B = x.size(0)
        u_hat = torch.einsum('bij,ikjl->bikl', x, self.W.permute(0,1,3,2))
        b = torch.zeros(B, x.size(1), self.num_capsules, device=x.device)
        v = None
        for _ in range(self.num_routing):
            c = torch.softmax(b, dim=2)                     
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)        
            v = self.squash(s)
            b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        return v  


class CapsNetStream(nn.Module):
    
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('resnet34', pretrained=True, features_only=True)
        _freeze(backbone)
        self.backbone = backbone

        self.primary_conv = nn.Conv2d(256, 32 * 8, kernel_size=3, stride=1, padding=0)

        self.digit_caps = CapsuleLayer(
            in_capsules=32, in_dim=8,
            num_capsules=16, caps_dim=16,
            num_routing=3
        )

        # Final projection to 256
        self.proj = nn.Linear(16 * 16, 256)
        print('[CapsNetStream] CapsNet -> 256-dim loaded')

    def forward(self, x):
        feats = self.backbone(x)[3]                    
        p = self.primary_conv(feats)                   
        B, _, H, W = p.shape
        p = p.view(B, 32, 8, H * W).permute(0, 3, 1, 2) 
        p = p.mean(dim=1)                               
        v = self.digit_caps(p)                          
        out = v.view(B, -1)                             
        return self.proj(out)                           


# U-Net Encoder 
class UNetEncoderStream(nn.Module):
    
    def __init__(self):
        super().__init__()
        backbone = timm.create_model(
            'resnet50', pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        _freeze(backbone)
        self.encoder = backbone
        self.proj0 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,   32))
        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256,  64))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512,  128))
        self.proj3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 160))
        self.proj4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 128))
        for proj in [self.proj0, self.proj1, self.proj2, self.proj3, self.proj4]:
            _freeze(proj)
        print('[UNetEncoderStream] ResNet50 U-Net encoder → 512-dim loaded & frozen')

    def forward(self, x):
        stages = self.encoder(x)   
        f0 = self.proj0(stages[0])
        f1 = self.proj1(stages[1])
        f2 = self.proj2(stages[2])
        f3 = self.proj3(stages[3])
        f4 = self.proj4(stages[4])
        return torch.cat([f0, f1, f2, f3, f4], dim=1)  


# Digital Forensics
def forensics_batch(pil_images: list) -> torch.Tensor:
    
    feats = []
    for img in pil_images:
        f = extract_forensic_features(img)  
        feats.append(f)
    return torch.tensor(np.stack(feats), dtype=torch.float32)


# AllStreams module
class AllStreams(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.spatial   = SpatialCNNStream()     
        self.freq      = F3NetStream()           
        self.transform = TransformerStream()     
        self.caps      = CapsNetStream()         
        self.unet      = UNetEncoderStream()     
        print('[AllStreams] All 7 neural streams ready.')

    def forward(self, x):
        
        f_xcep, f_eff  = self.spatial(x)
        f_freq         = self.freq(x)
        f_vit,  f_swin = self.transform(x)
        f_caps         = self.caps(x)
        f_unet         = self.unet(x)
        return [f_xcep, f_eff, f_freq, f_vit, f_swin, f_caps, f_unet]



all_streams = AllStreams().to(DEVICE)
all_streams.eval()
