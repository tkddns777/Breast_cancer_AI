import torch
import torch.nn as nn
import torchvision.models as models
import timm


# =========================
# EfficientNet Multi-View
# =========================
class EfficientNetMultiView(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        feature_dim = self.encoder.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, lcc, lmlo, rcc, rmlo):

        f1 = self.encoder(lcc)
        f2 = self.encoder(lmlo)
        f3 = self.encoder(rcc)
        f4 = self.encoder(rmlo)

        x = torch.cat([f1, f2, f3, f4], dim=1)

        out = self.classifier(x)

        return out


# =========================
# ResNet Encoder
# =========================
class ResNetEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        base_model = models.resnet18(pretrained=True)

        self.encoder = nn.Sequential(
            *list(base_model.children())[:-1]
        )

        self.feature_dim = base_model.fc.in_features

    def forward(self, x):

        x = self.encoder(x)
        x = x.flatten(1)
        

        return x


# =========================
# ResNet Multi-View Model
# =========================
class MultiViewResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNetEncoder()

        f = self.encoder.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(f * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, lcc, lmlo, rcc, rmlo):

        f1 = self.encoder(lcc)
        f2 = self.encoder(lmlo)
        f3 = self.encoder(rcc)
        f4 = self.encoder(rmlo)

        x = torch.cat([f1, f2, f3, f4], dim=1)

        out = self.classifier(x)

        return out


# =========================
# Bilateral Breast Model
# =========================
class BilateralModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNetEncoder()

        f = self.encoder.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(f * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, lcc, lmlo, rcc, rmlo):

        # Left breast
        f_lcc = self.encoder(lcc)
        f_lmlo = self.encoder(lmlo)

        f_left = (f_lcc + f_lmlo) / 2

        # Right breast
        f_rcc = self.encoder(rcc)
        f_rmlo = self.encoder(rmlo)

        f_right = (f_rcc + f_rmlo) / 2

        # Bilateral difference
        f_diff = torch.abs(f_left - f_right)

        x = torch.cat([f_left, f_right, f_diff], dim=1)

        out = self.classifier(x)

        return out