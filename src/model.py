import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)  # 加上这行
        )

        self.activation = nn.GELU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = out + self.shortcut(x)
        out = self.activation(out)
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage1 = ConvBlock(8, 16)
        self.stage2 = ConvBlock(16, 32)
        self.stage3 = ConvBlock(32, 64)
        self.stage4 = ConvBlock(64, 128)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.pooling(out)
        out = torch.flatten(out, 1)

        return out


class MultiViewFusion(nn.Module):
    def __init__(self, num_views=2, d_model=128):
        super().__init__()
        self.backbone = ConvNet()
        backbone_out_dim = 128
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_dim * num_views, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        batch, T, num_views, C, H, W = x.shape
        x = x.view(batch * T * num_views, C, H, W)
        features = self.backbone(x)
        features = features.view(batch, T, -1)
        fused = self.fusion(features)
        return fused


class VisionActionModel(nn.Module):
    def __init__(self, num_views=3, len_sequence_out=6, action_dim=7, d_model=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.len_sequence_out = len_sequence_out
        self.fusion = MultiViewFusion(num_views, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len_sequence_out * action_dim)
        )

    def forward(self, images):                                      # (B, T_in, num_views, C, H, W)
        batch = images.shape[0]
        fused = self.fusion(images)                                 # (B, T_in, d_model)
        encoded = self.transformer(fused)                           # (B, T_in, d_model)
        global_feat = encoded.mean(dim=1)                             # (B, d_model)  全局平均池化
        flat_out = self.output_head(global_feat)                    # (B, T_out * action_dim)
        actions = flat_out.view(batch, self.len_sequence_out, -1)   # (B, T_out, action_dim)
        return actions
