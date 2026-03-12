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

        # BatchNorm1d 추가 + Dropout 0.3 → 0.5 강화 + 중간 레이어 추가
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
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

        # BatchNorm1d 추가
        self.classifier = nn.Sequential(
            nn.Linear(f * 4, 512),
            nn.BatchNorm1d(512),
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
# View Attention Module
# =========================
class ViewAttention(nn.Module):
    """
    CC뷰와 MLO뷰 중 어느 뷰가 더 진단에 중요한지 학습하는 경량 어텐션.

    - CC(위→아래)와 MLO(사선) 두 뷰를 입력받아 가중 합산(Weighted Sum) 반환
    - 파라미터: feature_dim * 2 → 2 (Softmax) → 약 2K params (매우 경량)

    예시:
        암이 뚜렷한 케이스 → MLO에 높은 가중치 부여
        미세석회화 케이스  → CC에 높은 가중치 부여
    """

    def __init__(self, feature_dim):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, f_cc, f_mlo):

        # 두 뷰를 concat해서 어텐션 가중치 계산
        combined = torch.cat([f_cc, f_mlo], dim=1)   # [B, feature_dim*2]
        weights  = self.attn(combined)                # [B, 2]

        w_cc  = weights[:, 0:1]                       # [B, 1]
        w_mlo = weights[:, 1:2]                       # [B, 1]

        # 가중 합산
        fused = w_cc * f_cc + w_mlo * f_mlo           # [B, feature_dim]

        return fused


# =========================
# Bilateral Breast Model (Attention + Element-wise Product)
# =========================
class BilateralModel(nn.Module):
    """
    개선된 양측 유방 비교 모델 (ResNet18 고정)

    [변경 사항]
    1. 단순 평균 → ViewAttention 기반 CC/MLO 뷰 융합
       - 케이스마다 CC/MLO 뷰의 중요도를 학습
    2. 양측 비교 특징에 Element-wise 곱(f_prod) 추가
       - f_diff: 좌우 비대칭 특징 (암은 종종 한쪽에만 나타남)
       - f_prod: 좌우 공통 특징 (정상 조직 패턴 기준선)
    3. Classifier 입력 1536d → 2048d

    [속도]
    - 인코더는 ResNet18 고정 (기존 동일)
    - 추가 파라미터: ViewAttention x2 ≈ 67K, f_prod 추가 ≈ 262K
    - 총 추가 파라미터 약 330K → 속도 영향 거의 없음
    """

    def __init__(self):

        super().__init__()

        self.encoder = ResNetEncoder()

        f = self.encoder.feature_dim   # ResNet18 → 512

        # 좌측/우측 뷰 어텐션 (각각 독립적으로 CC/MLO 가중치 학습)
        self.attn_left  = ViewAttention(f)
        self.attn_right = ViewAttention(f)

        # 입력: [f_left, f_right, f_diff, f_prod] = 512 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(f * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, lcc, lmlo, rcc, rmlo):

        # 4뷰 인코딩 (공유 ResNet18)
        f_lcc  = self.encoder(lcc)
        f_lmlo = self.encoder(lmlo)
        f_rcc  = self.encoder(rcc)
        f_rmlo = self.encoder(rmlo)

        # 어텐션 기반 뷰 융합: CC/MLO 가중치를 케이스별로 학습
        f_left  = self.attn_left(f_lcc,  f_lmlo)   # [B, 512]
        f_right = self.attn_right(f_rcc, f_rmlo)   # [B, 512]

        # 양측 비교 특징
        f_diff = torch.abs(f_left - f_right)        # 비대칭 특징 [B, 512]
        f_prod = f_left * f_right                   # 공통 특징   [B, 512]

        # concat → classifier
        x   = torch.cat([f_left, f_right, f_diff, f_prod], dim=1)   # [B, 2048]
        out = self.classifier(x)

        return out
