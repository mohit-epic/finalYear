"""
Cloud Removal Network with Global Cross-Attention and Speckle-Aware Gating
Architecture: Dual Encoders (Optical + SAR) -> Transformer Cross-Attention -> Optical-Guided Gating -> Decoder
Updates:
1. ResBlocks instead of ConvBlocks
2. Transformer-style Attention (Norm + FFN)
3. Optical-Guided Gating (Optical + SAR)
4. Global Residual Connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class ResBlock(nn.Module):
    """Residual Block with GroupNorm for better training stability"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Use GroupNorm instead of BatchNorm for better stability
        # 32 groups is a good default, but we need to ensure it divides the channels
        num_groups = min(32, out_channels) if out_channels >= 32 else out_channels
        # Ensure num_groups divides out_channels
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Shortcut handling
        if stride != 1 or in_channels != out_channels:
            # Also use GroupNorm in shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.relu(out)
        return out


class OpticalEncoder(nn.Module):
    """Optical image encoder with 3 levels of downsampling using ResBlocks"""
    def __init__(self):
        super(OpticalEncoder, self).__init__()
        # E1: input (13, H, W) -> output (64, H, W)
        self.E1 = ResBlock(13, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsample 1 - Learnable strided convolution instead of MaxPool
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # E2: input (64, H/2, W/2) -> output (128, H/2, W/2)
        self.E2 = ResBlock(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsample 2 - Learnable strided convolution instead of MaxPool
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        
        # E3: input (128, H/4, W/4) -> output (256, H/4, W/4)
        self.E3 = ResBlock(128, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        feat_opt_1 = self.E1(x)
        x = self.down1(feat_opt_1)
        feat_opt_2 = self.E2(x)
        x = self.down2(feat_opt_2)
        feat_opt_3 = self.E3(x)
        return feat_opt_1, feat_opt_2, feat_opt_3


class SpectralAttentionSAR(nn.Module):
    """
    Spectral/Channel Attention for SAR features.
    Different SAR channels (VV, VH) have different information content.
    Learn to weight them adaptively.
    """
    def __init__(self, sar_channels=2, reduction=4):
        super(SpectralAttentionSAR, self).__init__()
        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: FC layers to learn channel weights
        self.fc = nn.Sequential(
            nn.Conv2d(sar_channels, max(1, sar_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, sar_channels // reduction), sar_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        weights = self.avg_pool(x)  # (B, C, 1, 1)
        weights = self.fc(weights)   # (B, C, 1, 1)
        return x * weights            # Channel-wise multiplication


class SAREncoder(nn.Module):
    """SAR image encoder with spectral attention and 3 levels"""
    def __init__(self):
        super(SAREncoder, self).__init__()
        
        # Spectral attention at input
        self.spectral_attn = SpectralAttentionSAR(sar_channels=2, reduction=2)
        
        # E1: input (2, H, W) -> output (64, H, W)
        self.E1 = ResBlock(2, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsample 1 - Learnable strided convolution instead of MaxPool
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.E2 = ResBlock(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsample 2 - Learnable strided convolution instead of MaxPool
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.E3 = ResBlock(128, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Apply spectral attention to raw SAR input
        x = self.spectral_attn(x)
        
        feat_sar_1 = self.E1(x)
        x = self.down1(feat_sar_1)
        feat_sar_2 = self.E2(x)
        x = self.down2(feat_sar_2)
        feat_sar_3 = self.E3(x)
        return feat_sar_1, feat_sar_2, feat_sar_3


class GlobalCrossAttention(nn.Module):
    """The raw attention mechanism"""
    def __init__(self, dim=256, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(GlobalCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, feat_opt, feat_sar):
        B, C, H, W = feat_opt.shape
        N = H * W
        
        feat_opt_flat = feat_opt.flatten(2).transpose(1, 2)
        feat_sar_flat = feat_sar.flatten(2).transpose(1, 2)
        
        q = self.q_proj(feat_opt_flat)
        k = self.k_proj(feat_sar_flat)
        v = self.v_proj(feat_sar_flat)
        
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class FFN(nn.Module):
    """Feed Forward Network for Transformer Block"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class TransformerCrossAttnBlock(nn.Module):
    """
    Enhanced Cross-Attention Block with Transformer architecture:
    Norm -> CrossAttn -> Add -> Norm -> FFN -> Add
    """
    def __init__(self, dim=256, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., mlp_ratio=4.):
        super(TransformerCrossAttnBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, dim) # LayerNorm equivalent for (B,C,H,W) is GroupNorm(1, C)
        self.attn = GlobalCrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio), proj_drop)
        
    def forward(self, feat_opt, feat_sar):
        # Attention block (residual handled internally or here?)
        # Standard Transformer: x = x + attn(norm(x))
        # But here inputs differ. We apply residual to the QUERY (feat_opt).
        
        # 1. Attention
        # Note: We apply norm to inputs before attention usually
        reshaped_attn_out = self.attn(self.norm1(feat_opt), self.norm1(feat_sar))
        x = feat_opt + reshaped_attn_out
        
        # 2. FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class GateNet(nn.Module):
    """
    Enhanced Speckle-Aware Gating with Optical Guidance.
    Input now sees BOTH SAR (noise source) and Optical (target context).
    """
    def __init__(self, dim=256):
        super(GateNet, self).__init__()
        
        # Input dim is doubled because we concat Optical + SAR
        self.input_dim = dim * 2 
        
        self.spatial_path = nn.Sequential(
            nn.Conv2d(self.input_dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.channel_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.input_dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, feat_opt, feat_sar):
        # Concatenate Optical and SAR features
        combined = torch.cat([feat_opt, feat_sar], dim=1) # (B, 2C, H, W)
        
        spatial_gate = self.spatial_path(combined)
        channel_gate = self.channel_path(combined)
        
        gate = spatial_gate * channel_gate
        return gate


class SpeckleAwareGatingModule(nn.Module):
    """Speckle-aware gating that uses SAR + OPTICAL to gate features"""
    def __init__(self, dim=256):
        super(SpeckleAwareGatingModule, self).__init__()
        self.gate_net = GateNet(dim)
        
    def forward(self, cross_out, feat_opt, feat_sar):
        """
        Args:
            cross_out: Features to be gated (output of cross attn)
            feat_opt: Reference optical features for gating context
            feat_sar: Reference SAR features for noise context
        """
        # Learn gate from raw feature context
        sigma = self.gate_net(feat_opt, feat_sar)
        gated_out = cross_out * sigma
        return gated_out


class Refinement(nn.Module):
    def __init__(self, dim=256):
        super(Refinement, self).__init__()
        self.refine_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
    def forward(self, gated_out, feat_opt_3):
        concat = torch.cat([gated_out, feat_opt_3], dim=1)
        refined = self.refine_conv(concat)
        return refined


class Decoder(nn.Module):
    """Symmetric decoder with ResBlocks"""
    def __init__(self, output_channels=13):
        super(Decoder, self).__init__()
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input: 256 (upsampled) + 128 (skip) = 384 -> Output: 128
        self.dec1 = ResBlock(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input: 128 (upsampled) + 64 (skip) = 192 -> Output: 64
        self.dec2 = ResBlock(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, padding=0)
        
    def forward(self, refined_3, feat_opt_2, feat_opt_1):
        x = self.up1(refined_3)
        x = torch.cat([x, feat_opt_2], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, feat_opt_1], dim=1)
        x = self.dec2(x)
        
        output = self.final_conv(x)
        return output


class CloudConfidencePredictor(nn.Module):
    """
    Predicts cloud confidence map from bottleneck features.
    Output: 0 = clear region, 1 = cloudy region
    Used for adaptive residual weighting.
    """
    def __init__(self, in_dim=256):
        super(CloudConfidencePredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Output: [0, 1]
        )
    
    def forward(self, x):
        return self.predictor(x)


class CloudRemovalCrossAttention(nn.Module):
    """
    Complete Cloud Removal Network
    Integrated Improvements: ResBlocks, Transformer Attn, Opt-Guided Gating, Global Residual
    Enhanced with:
    - Strided convolutions for better detail preservation
    - 1x1 projection for SAR-optical feature alignment
    """
    
    def __init__(self, num_heads=8, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super(CloudRemovalCrossAttention, self).__init__()
        
        self.optical_encoder = OpticalEncoder()
        self.sar_encoder = SAREncoder()
        
        # Enhanced Transformer Block
        self.cross_attn = TransformerCrossAttnBlock(
            dim=256, 
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # 1x1 Projection after cross-attention to align SAR-optical distributions
        self.fuse_proj = nn.Conv2d(256, 256, kernel_size=1)
        
        # Enhanced Gating
        self.speckle_gating = SpeckleAwareGatingModule(dim=256)
        
        self.refinement = Refinement(dim=256)
        
        self.decoder = Decoder(output_channels=13)
        
        # Cloud-aware weighting predictor
        self.cloud_predictor = CloudConfidencePredictor(in_dim=256)
        
    def forward(self, optical_img, sar_img):
        # 1. Encoders
        feat_opt_1, feat_opt_2, feat_opt_3 = self.optical_encoder(optical_img)
        feat_sar_1, feat_sar_2, feat_sar_3 = self.sar_encoder(sar_img)
        
        # 2. Transformer Cross Attention
        # Note: feat_opt_3 is Query, feat_sar_3 is Key/Value context
        cross_out = self.cross_attn(feat_opt_3, feat_sar_3)
        
        # 2.5. Feature alignment projection (aligns SAR-optical distributions)
        cross_out = self.fuse_proj(cross_out)
        
        # 3. Optical-Guided Gating
        # We pass feat_opt_3 as context so gating knows what look like clouds vs features
        gated_out = self.speckle_gating(cross_out, feat_opt_3, feat_sar_3)
        
        # 4. Refinement
        refined_3 = self.refinement(gated_out, feat_opt_3)
        
        # 5. Decoder (predicts residual/correction)
        residual_cloud = self.decoder(refined_3, feat_opt_2, feat_opt_1)
        
        # 6. Cloud-Aware Weighted Residual (ENHANCED)
        # Predict cloud confidence map from bottleneck features
        cloud_confidence = self.cloud_predictor(refined_3)  # (B, 1, H/4, W/4)
        cloud_confidence = F.interpolate(cloud_confidence, 
                                        size=optical_img.shape[2:], 
                                        mode='bilinear', 
                                        align_corners=True)  # (B, 1, H, W)
        
        # Adaptive weighting:
        # - Clear regions (conf ≈ 0): Use mostly input → (1-0)*residual + 0*input = residual
        # - Cloudy regions (conf ≈ 1): Use mostly residual → (1-1)*residual + 1*input
        # Actually we want: clear=input, cloudy=input+residual
        # So: output = input + conf * residual
        # This way:
        #   - Clear (conf=0): output = input + 0 = input (preserved)
        #   - Cloudy (conf=1): output = input + residual (corrected)
        output = optical_img + cloud_confidence * residual_cloud
        
        return output


def create_model(pretrained=False, **kwargs):
    model = CloudRemovalCrossAttention(**kwargs)
    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    height, width = 256, 256
    
    optical_img = torch.randn(batch_size, 13, height, width)
    sar_img = torch.randn(batch_size, 2, height, width)
    
    model = CloudRemovalCrossAttention()
    output = model(optical_img, sar_img)
    
    print(f"Input optical shape: {optical_img.shape}")
    print(f"Input SAR shape: {sar_img.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: torch.Size([{batch_size}, 13, {height}, {width}])")
    print(f"Architecture: Cloud-Aware Weighted Residual Learning")
    print(f"  - Spectral Attention for SAR: ✓")
    print(f"  - Cloud Confidence Predictor: ✓")
    print(f"  - Adaptive Residual Weighting: ✓")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Expected PSNR target: 35-36 dB")
