'''Compact Vision Transformer para CIFAR-10 (32x32 px), autocontenido.

Sin dependencias externas (solo torch). Diseñado para entrenar desde cero sobre un
dataset pequeño: patch=4 (-> 8x8 = 64 parches), dimensión y profundidad reducidas.

Se integra en el mismo pipeline que las CNN (monitor_train, LayerEnergyProfiler,
EES, GpuPowerOptimizer) sin cambios: es un nn.Module estándar cuyas capas hoja
(Linear, LayerNorm) son instrumentables por los forward hooks del profiler.

Uso:  from models import ViT ;  net = ViT()
'''
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Divide la imagen en parches y los proyecta a la dimensión del transformer."""
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=256):
        super().__init__()
        self.num_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                    # B, dim, H/patch, W/patch
        x = x.flatten(2).transpose(1, 2)    # B, N, dim
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        assert dim % heads == 0, "dim debe ser divisible por heads"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    """Bloque transformer: atención + MLP, con conexiones residuales y LayerNorm."""
    def __init__(self, dim, heads, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer compacto para CIFAR-10.

    Por defecto: patch=4, dim=256, depth=6, heads=8 (~5,4M parámetros).
    """
    def __init__(self, img_size=32, patch=4, in_ch=3, num_classes=10,
                 dim=256, depth=6, heads=8, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, in_ch, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def test():
    net = ViT()
    y = net(torch.randn(2, 3, 32, 32))
    print(y.shape, sum(p.numel() for p in net.parameters()) / 1e6, "M params")


if __name__ == "__main__":
    test()
