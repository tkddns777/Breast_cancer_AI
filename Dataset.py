import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

IMAGE_SIZE = 512

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)



class RSNADataset(Dataset):

    def __init__(self, meta_path, image_root, split="train", transform=False):

        df = pd.read_csv(meta_path)
        df = df[df["split"] == split].reset_index(drop=True)

        self.image_root = image_root
        self.transform = transform
        self.split = split

        valid_rows = []

        for _, row in df.iterrows():

            patient_id = int(row["patient_id"])

            paths = [
                os.path.join(image_root, f"{patient_id}_{row['L_CC']}.png"),
                os.path.join(image_root, f"{patient_id}_{row['L_MLO']}.png"),
                os.path.join(image_root, f"{patient_id}_{row['R_CC']}.png"),
                os.path.join(image_root, f"{patient_id}_{row['R_MLO']}.png"),
            ]

            if all(os.path.exists(p) for p in paths):
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(split, "usable samples:", len(self.df))


    def load_image(self, patient_id, image_id):

        path = os.path.join(
            self.image_root,
            f"{patient_id}_{image_id}.png"
        )

        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # 정규화 전 상태(0~1)로 반환 → augmentation 후 NORMALIZE 적용
        # torch.from_numpy: 메모리 복사 없는 zero-copy 변환
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float() / 255.0

        return img


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        while True:

            row = self.df.iloc[idx]
            patient_id = int(row["patient_id"])
            label_value = row["label"]

            try:

                lcc = self.load_image(patient_id, row["L_CC"])
                lmlo = self.load_image(patient_id, row["L_MLO"])
                rcc = self.load_image(patient_id, row["R_CC"])
                rmlo = self.load_image(patient_id, row["R_MLO"])

                label = torch.tensor(label_value).float()

                # =========================
                # AUGMENTATION (train only)
                # =========================

                if self.transform and self.split == "train":

                    # 회전: ±10도 (기존 ±12에서 약간 완화)
                    if random.random() < 0.7:

                        angle = random.uniform(-10, 10)

                        lcc  = TF.rotate(lcc,  angle)
                        lmlo = TF.rotate(lmlo, angle)
                        rcc  = TF.rotate(rcc,  angle)
                        rmlo = TF.rotate(rmlo, angle)

                    # 수평 뒤집기 (4뷰 동시 적용 → 좌우 대칭 유지)
                    if random.random() < 0.5:

                        lcc  = TF.hflip(lcc)
                        lmlo = TF.hflip(lmlo)
                        rcc  = TF.hflip(rcc)
                        rmlo = TF.hflip(rmlo)

                    # 밝기 조정
                    if random.random() < 0.5:

                        brightness = random.uniform(0.85, 1.15)

                        lcc  = TF.adjust_brightness(lcc,  brightness)
                        lmlo = TF.adjust_brightness(lmlo, brightness)
                        rcc  = TF.adjust_brightness(rcc,  brightness)
                        rmlo = TF.adjust_brightness(rmlo, brightness)

                    # [추가] 대비(Contrast) 조정 - 유방촬영술에서 조직 경계 강조 효과
                    if random.random() < 0.5:

                        contrast = random.uniform(0.85, 1.15)

                        lcc  = TF.adjust_contrast(lcc,  contrast)
                        lmlo = TF.adjust_contrast(lmlo, contrast)
                        rcc  = TF.adjust_contrast(rcc,  contrast)
                        rmlo = TF.adjust_contrast(rmlo, contrast)

                    # [추가] 가우시안 블러 - 과적합 방지 (저주파 특징 학습 유도)
                    if random.random() < 0.3:

                        kernel_size = random.choice([3, 5])
                        sigma       = random.uniform(0.1, 1.5)

                        lcc  = TF.gaussian_blur(lcc,  kernel_size, sigma)
                        lmlo = TF.gaussian_blur(lmlo, kernel_size, sigma)
                        rcc  = TF.gaussian_blur(rcc,  kernel_size, sigma)
                        rmlo = TF.gaussian_blur(rmlo, kernel_size, sigma)

                    # Random Crop (기존 코드)
                    if random.random() < 0.5:

                        scale = random.uniform(0.9, 1.0)

                        h, w = lcc.shape[1:]
                        new_h = int(h * scale)
                        new_w = int(w * scale)

                        top  = random.randint(0, h - new_h)
                        left = random.randint(0, w - new_w)

                        lcc  = TF.resized_crop(lcc,  top, left, new_h, new_w, (h, w))
                        lmlo = TF.resized_crop(lmlo, top, left, new_h, new_w, (h, w))
                        rcc  = TF.resized_crop(rcc,  top, left, new_h, new_w, (h, w))
                        rmlo = TF.resized_crop(rmlo, top, left, new_h, new_w, (h, w))

                # =========================
                # NORMALIZE (augmentation 후 적용)
                # =========================

                lcc  = NORMALIZE(lcc)
                lmlo = NORMALIZE(lmlo)
                rcc  = NORMALIZE(rcc)
                rmlo = NORMALIZE(rmlo)

                return lcc, lmlo, rcc, rmlo, label

            except Exception:

                idx = (idx + 1) % len(self.df)