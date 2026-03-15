import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim

from Dataset import RSNADataset
from model import EfficientNetMultiView, MultiViewResNet, BilateralModel

import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score


# =========================
# EARLY STOPPING
# =========================

class EarlyStopping:
    """Val AUC가 patience 에폭 동안 개선되지 않으면 학습 중단"""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = 0.0
        self.stop       = False

    def __call__(self, val_auc):

        if val_auc > self.best_score + self.min_delta:
            self.best_score = val_auc
            self.counter    = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# =========================
# USER SETTINGS
# =========================

meta_path = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\metadata.csv"
image_root = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\Data"

save_dir = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\models"

epochs = 20
batch_size = 16
model_type = "bilateral"
num_workers = 4          # CPU 12코어 → 4 workers (데이터 로딩 병렬화)

seeds = [100, 200, 300]   # 여러 시드로 실험하여 평균 성능 평가


# =========================
# SYSTEM SETTINGS
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

torch.backends.cudnn.benchmark = True


# =========================
# SEED FUNCTION
# =========================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# MODEL SELECTOR
# =========================

def build_model():

    if model_type == "efficientnet":
        model = EfficientNetMultiView()

    elif model_type == "resnet":
        model = MultiViewResNet()

    elif model_type == "bilateral":
        model = BilateralModel()

    else:
        raise ValueError("Invalid model_type")

    return model


# =========================
# TRAINING LOOP
# =========================

# Windows 멀티프로세싱: num_workers > 0 사용 시 필수 가드
if __name__ == "__main__":

    for seed in seeds:

        print("\n===================================")
        print(f"Running with SEED = {seed}")
        print("===================================\n")

        set_seed(seed)

        # Dataset
        train_dataset = RSNADataset(meta_path, image_root, "train", transform=True)
        labels = train_dataset.df["label"].values

        class_sample_count = np.array([
            np.sum(labels == 0),
            np.sum(labels == 1)
        ])

        print("Class counts:", class_sample_count)

        weight = 1. / class_sample_count

        samples_weight = np.array([weight[int(t)] for t in labels])

        samples_weight = torch.from_numpy(samples_weight).float()

        sampler = WeightedRandomSampler(
            samples_weight,
            len(samples_weight),
            replacement=True
        )

        val_dataset   = RSNADataset(meta_path, image_root, "val", transform=None)
        test_dataset  = RSNADataset(meta_path, image_root, "test", transform=None)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,   # 에폭 간 worker 재시작 방지
            prefetch_factor=2          # 다음 배치 미리 로딩
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )


        model = build_model().to(device)

        # =========================
        # CLASS INFO
        # =========================

        labels = train_dataset.df["label"].values

        healthy_count = np.sum(labels == 0)
        cancer_count  = np.sum(labels == 1)

        print("Healthy:", healthy_count)
        print("Cancer:", cancer_count)
        print("WeightedRandomSampler가 이미 클래스 균형을 맞추므로 pos_weight 미사용")

        # WeightedRandomSampler가 배치 내 클래스를 1:1로 맞추기 때문에
        # pos_weight를 추가 적용하면 암을 과도하게 23배 이중 가중 → 제거
        # Label Smoothing: 정답 레이블을 0/1 대신 0.05/0.95로 완화 → 과신(overconfidence) 방지
        label_smoothing = 0.05
        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        scaler = torch.cuda.amp.GradScaler()

        best_val_auc = 0

        save_path = f"{save_dir}\\MODEL_{model_type}_seed{seed}.pth"


        # =========================
        # TRAIN
        # =========================

        for epoch in range(epochs):

            model.train()

            pbar = tqdm(train_loader)

            # 에폭 평균 loss 추적
            epoch_loss_sum   = 0.0
            epoch_loss_count = 0

            for lcc, lmlo, rcc, rmlo, label in pbar:

                lcc  = lcc.to(device,  non_blocking=True)
                lmlo = lmlo.to(device, non_blocking=True)
                rcc  = rcc.to(device,  non_blocking=True)
                rmlo = rmlo.to(device, non_blocking=True)

                label = label.unsqueeze(1).to(device, non_blocking=True)

                # Label Smoothing 적용: 0 → 0.025, 1 → 0.975
                label_smooth = label * (1 - label_smoothing) + label_smoothing * 0.5

                with torch.amp.autocast("cuda"):

                    pred = model(lcc, lmlo, rcc, rmlo)

                    loss = criterion(pred, label_smooth)

                optimizer.zero_grad(set_to_none=True)   # zero_grad 메모리 최적화

                scaler.scale(loss).backward()

                # Gradient Clipping: 그래디언트 폭발 방지 (max_norm=1.0)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)

                scaler.update()

                epoch_loss_sum   += loss.item()
                epoch_loss_count += 1

                # tqdm에 현재 배치 loss + 누적 평균 loss 동시 표시
                avg_loss = epoch_loss_sum / epoch_loss_count
                pbar.set_description(f"Epoch {epoch} | batch_loss {loss.item():.4f} | avg_loss {avg_loss:.4f}")


            # =========================
            # VALIDATION
            # =========================

            model.eval()

            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():

                for lcc, lmlo, rcc, rmlo, label in val_loader:

                    lcc = lcc.to(device, non_blocking=True)
                    lmlo = lmlo.to(device, non_blocking=True)
                    rcc = rcc.to(device, non_blocking=True)
                    rmlo = rmlo.to(device, non_blocking=True)

                    label = label.unsqueeze(1).to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):

                        pred = model(lcc, lmlo, rcc, rmlo)

                        loss = criterion(pred, label)

                    val_loss += loss.item()

                    prob = torch.sigmoid(pred).squeeze(1)

                    val_preds.extend(prob.cpu().numpy())
                    val_labels.extend(label.cpu().numpy().flatten())

            val_loss /= len(val_loader)

            val_preds  = np.array(val_preds)
            val_labels = np.array(val_labels)

            val_auc    = roc_auc_score(val_labels, val_preds)
            val_binary = (val_preds > 0.5).astype(int)
            val_acc    = accuracy_score(val_labels, val_binary)

            # 에폭 평균 train_loss 출력
            train_avg_loss = epoch_loss_sum / epoch_loss_count

            print(f"Epoch {epoch} | train_loss(avg) {train_avg_loss:.4f} | val_loss {val_loss:.4f} | val_auc {val_auc:.4f} | val_acc {val_acc:.4f}")

            scheduler.step()

            if val_auc > best_val_auc:

                best_val_auc = val_auc

                torch.save(model.state_dict(), save_path)

                print(f"Best model saved (AUC: {val_auc:.4f})")

            # Early Stopping 체크
            if early_stopping(val_auc):
                print(f"\nEarly stopping triggered at epoch {epoch}. Best AUC: {early_stopping.best_score:.4f}")
                break


        # =========================
        # TEST
        # =========================

        print("\nRunning TEST evaluation...")

        model.load_state_dict(
            torch.load(save_path, weights_only=True)
        )

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for lcc, lmlo, rcc, rmlo, label in test_loader:

                lcc = lcc.to(device, non_blocking=True)
                lmlo = lmlo.to(device, non_blocking=True)
                rcc = rcc.to(device, non_blocking=True)
                rmlo = rmlo.to(device, non_blocking=True)

                label = label.unsqueeze(1).to(device, non_blocking=True)

                pred = model(lcc, lmlo, rcc, rmlo)

                prob = torch.sigmoid(pred).squeeze(1)

                all_preds.extend(prob.cpu().numpy())
                all_labels.extend(label.cpu().numpy().flatten())


        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        auc = roc_auc_score(all_labels, all_preds)

        pred_binary = (all_preds > 0.5).astype(int)

        acc = accuracy_score(all_labels, pred_binary)

        print("\nTEST RESULTS")
        print("AUC:", auc)
        print("Accuracy:", acc)


        # =========================
        # SAVE FINAL MODEL + METRIC
        # =========================

        final_save_path = f"{save_dir}\\FINAL_{model_type}_seed{seed}_AUC{auc:.4f}.pth"

        torch.save({
            "model_state_dict": model.state_dict(),
            "AUC": auc,
            "Accuracy": acc,
            "seed": seed
        }, final_save_path)

        print("Final model saved:", final_save_path)
