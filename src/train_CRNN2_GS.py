import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================
# CONFIG
# =========================
DATASET_ROOT = "D:/model2-20260110T081631Z-3-001/model2/datasets"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train2")
TEST_DIR  = os.path.join(DATASET_ROOT, "test2")

IMG_W = 50
IMG_H = 20
CHANNELS = 1  # <<-- Đổi từ 3 thành 1

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHARS = "0123456789"
NUM_CLASSES = len(CHARS) + 1  # + CTC blank

# =========================
# LABEL ENCODER
# =========================
char2idx = {c: i for i, c in enumerate(CHARS)}
idx2char = {i: c for c, i in char2idx.items()}

def encode_label(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

# =========================
# DATASET
# =========================
class CRNNDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "images")
        self.labels = []

        label_path = os.path.join(root_dir, "labels.txt")
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                name, label = line.strip().split("\t")
                self.labels.append((name, label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Đọc ảnh ở dạng grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        
        # Đảm bảo ảnh đúng kích thước target nếu cần thiết
        # img = cv2.resize(img, (IMG_W, IMG_H)) 

        img = cv2.resize(img, (IMG_W, IMG_H)) 
        # ------------------------------------------

        img = img.astype("float32") / 255.0
        # Chuyển thành (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0) 

        label = encode_label(label)
        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return imgs, labels, label_lengths

# =========================
# MODEL
# =========================
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # Lớp đầu vào đổi từ 3 thành 1
            nn.Conv2d(CHANNELS, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 20x50 -> 10x25

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 10x25 -> 5x12
        )

        self.rnn = nn.LSTM(
            input_size=128 * 5,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# =========================
# DECODE & METRIC (Giữ nguyên)
# =========================
def greedy_decode(logits):
    preds = logits.argmax(2)
    texts = []
    for p in preds:
        prev = -1
        text = ""
        for c in p:
            c = c.item()
            if c != prev and c != NUM_CLASSES - 1:
                text += idx2char[c]
            prev = c
        texts.append(text)
    return texts

def accuracy(preds, targets):
    correct = 0
    for p, t in zip(preds, targets):
        if p == t:
            correct += 1
    return correct / len(targets)

def get_gt_texts(labels, label_lengths):
    texts = []
    idx = 0
    for l in label_lengths:
        texts.append("".join(idx2char[i.item()] for i in labels[idx:idx+l]))
        idx += l
    return texts

# =========================
# TRAIN / EVAL LOOP (Giữ nguyên phần logic)
# =========================
train_ds = CRNNDataset(TRAIN_DIR)
test_ds  = CRNNDataset(TEST_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = CRNN().to(DEVICE)
criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(EPOCHS):
    # --- TRAIN PHASE ---
    model.train()
    epoch_train_loss = 0
    epoch_train_correct = 0
    total_train_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for imgs, labels, label_lengths in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        log_probs = logits.log_softmax(2)

        T = log_probs.size(1)
        input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long).to(DEVICE)

        loss = criterion(log_probs.permute(1, 0, 2), labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        
        # Tính Accuracy nhanh cho train (không cần pass riêng)
        with torch.no_grad():
            preds = greedy_decode(logits.cpu())
            gts = get_gt_texts(labels.cpu(), label_lengths)
            for p, g in zip(preds, gts):
                if p == g: epoch_train_correct += 1
            total_train_samples += len(gts)

        pbar.set_postfix(loss=loss.item())

    train_losses.append(epoch_train_loss / len(train_loader))
    train_accs.append(epoch_train_correct / total_train_samples)

    # --- EVAL PHASE ---
    model.eval()
    epoch_test_loss = 0
    epoch_test_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for imgs, labels, label_lengths in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            log_probs = logits.log_softmax(2)
            
            T = log_probs.size(1)
            input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long).to(DEVICE)
            
            loss = criterion(log_probs.permute(1, 0, 2), labels, input_lengths, label_lengths)
            epoch_test_loss += loss.item()

            preds = greedy_decode(logits.cpu())
            gts = get_gt_texts(labels.cpu(), label_lengths)
            for p, g in zip(preds, gts):
                if p == g: epoch_test_correct += 1
            total_test_samples += len(gts)

    test_losses.append(epoch_test_loss / len(test_loader))
    test_accs.append(epoch_test_correct / total_test_samples)

    print(f"Epoch {epoch+1}: Train Acc={train_accs[-1]:.4f}, Test Acc={test_accs[-1]:.4f}")

# Save model
save_path = "D:/Digital-Time-Reader/model/Reader/crnn/crnn_synthetic_gray.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)

# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(14, 5))

# Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue', linestyle='--')
plt.plot(test_losses, label='Test Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', color='blue', linestyle='--')
plt.plot(test_accs, label='Test Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()