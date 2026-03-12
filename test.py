import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from Dataset import RSNADataset
from model import BreastCancerModel


meta_path = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\metadata.csv"
image_root = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\Data"

model_path = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Breast_cancer_AI\models\MODEL.pth"


device = "cuda" if torch.cuda.is_available() else "cpu"


# dataset
test_dataset = RSNADataset(meta_path, image_root, "test")

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# model
model = BreastCancerModel().to(device)

model.load_state_dict(torch.load(model_path))

model.eval()


criterion = torch.nn.BCEWithLogitsLoss()

test_loss = 0

all_preds = []
all_labels = []


with torch.no_grad():

    for lcc, lmlo, rcc, rmlo, label in test_loader:

        lcc = lcc.to(device)
        lmlo = lmlo.to(device)
        rcc = rcc.to(device)
        rmlo = rmlo.to(device)

        label = label.unsqueeze(1).to(device)

        pred = model(lcc, lmlo, rcc, rmlo)

        loss = criterion(pred, label)

        test_loss += loss.item()

        prob = torch.sigmoid(pred)

        all_preds.extend(prob.cpu().numpy())
        all_labels.extend(label.cpu().numpy())


test_loss /= len(test_loader)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

auc = roc_auc_score(all_labels, all_preds)

pred_binary = (all_preds > 0.5).astype(int)

acc = accuracy_score(all_labels, pred_binary)


print("Test Loss:", test_loss)
print("Test AUC:", auc)
print("Test Accuracy:", acc)