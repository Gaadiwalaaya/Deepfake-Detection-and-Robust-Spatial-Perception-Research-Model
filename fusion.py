# MAIN TRAINING MODEL

import torch
import torch.nn as nn
from config import DEVICE, STREAM_DIMS, PROJ_DIM, ATTN_DIM


class AttentionFusion(nn.Module):
    
    def __init__(self, stream_dims=STREAM_DIMS, proj_dim=PROJ_DIM, attn_dim=ATTN_DIM):
        super().__init__()
        self.n_streams = len(stream_dims)
        self.proj_dim  = proj_dim

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            )
            for dim in stream_dims
        ])
    
        self.attn_net = nn.Sequential(
            nn.Linear(proj_dim, attn_dim),
            nn.Tanh(),
        )

        self.context_vector = nn.Parameter(torch.randn(attn_dim))

    def forward(self, stream_feats: list):
        
        proj = [self.projections[i](stream_feats[i])
                for i in range(self.n_streams)]           

        stacked = torch.stack(proj, dim=1)

        e = self.attn_net(stacked)

        scores = torch.einsum('bnd,d->bn', e, self.context_vector)
        alphas = torch.softmax(scores, dim=1)        

        F = (alphas.unsqueeze(-1) * stacked).sum(dim=1)   
        return F, alphas


class MetaClassifier(nn.Module):
    
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
        )

    def forward(self, F):
        return self.net(F) 


class EnsembleFusion(nn.Module):
    
    def __init__(self, stream_dims=STREAM_DIMS, proj_dim=PROJ_DIM, attn_dim=ATTN_DIM):
        super().__init__()
        self.attention_fusion = AttentionFusion(stream_dims, proj_dim, attn_dim)
        self.meta_classifier  = MetaClassifier(proj_dim)
        print(f'[EnsembleFusion] {len(stream_dims)} streams → proj_dim={proj_dim}')

    def forward(self, stream_feats: list):
       
        F, alphas = self.attention_fusion(stream_feats)
        logit     = self.meta_classifier(F)
        return logit, alphas

    def predict(self, stream_feats: list):
        logit, alphas = self.forward(stream_feats)
        prob = torch.sigmoid(logit)
        return prob, alphas

fusion_model = EnsembleFusion().to(DEVICE)
