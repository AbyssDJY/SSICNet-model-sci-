import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

class CreatePatches(nn.Module):
    def __init__(
        self, channels=15, embed_dim=256, patch_size=4
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        # Flatten along dim = 2 to maintain channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        return patches
     

class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        if gamma is not None:
            self.gamma = torch.tensor([gamma])
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0]))
        self.eps = torch.tensor([1e-6])
        self.gamma = gamma
        self.use_beta = use_beta
    def prepare_input(self, x):
        if len(x.shape) == 3: # Transformer
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2) # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1) # (B, d) -> (B, 1, d)
            return gap_cls, x

        if len(x.shape) == 4: # CNN
            # Input tensor dimensions:
            # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
            B, d, H, W = x.shape
            gap_cls = x.mean([-2, -1]) # (B, d, H, W) -> (B, d)
            x = x.reshape(B, d, H*W).permute(0, 2, 1) # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1) # (B, d) -> (B, 1, d)

            return gap_cls, x
        else:
            raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")
    def forward(self, x):
        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)
        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)
        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape
        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv
        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1, 3) # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1, 3) # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)
        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1, 3) # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)
        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)
        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma), 1/self.gamma) # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)

        return x.squeeze()

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):
    def __init__(self, dim, num_patch, mlp_dim, dropout = 0.0, init_values=1e-4):
        super().__init__()
        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x

class ResMLP(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))
        self.affine = Aff(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim)
        )
    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        x = self.affine(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)
     

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.simpool = SimPool(embed_dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.pre_norm(x)
        x = x + self.simpool(x_norm)[0]
        x = x + self.MLP(self.norm(x))
        return x
     
class SimPoolFormer(nn.Module):
    def __init__(
        self,
        img_size=8,
        in_channels=15,
        patch_size=2,
        embed_dim=256,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
        num_classes=18,
        depth=4,
        mlp_dim=256
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size//patch_size) ** 2
        self.patches = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.attn_layers.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        self.resmlp= ResMLP(in_channels, embed_dim, num_classes, patch_size, img_size, depth, mlp_dim)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x1 = self.resmlp(x)
        x = self.patches(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.ln(x)
        x = x.mean(dim=1)
     #  x = x[:, 0]
        x= x + x1
        return self.head(x)
     

if __name__ == '__main__':
    model = SimPoolFormer(
        img_size=9,
        in_channels=12,
        patch_size=3,
        embed_dim=256,
        hidden_dim= 128,
        num_heads=4,
        num_layers=2,
        num_classes=10,
        depth=2
    )
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    rnd_int = torch.randn(1, 12, 9, 9)
    output = model(rnd_int)
    print(f"Output shape from model: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数量: {total_params:,} 个参数')
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数量: {trainable_params:,} 个参数')
