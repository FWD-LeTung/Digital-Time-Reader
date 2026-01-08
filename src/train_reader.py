import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# ================= CONFIG =================
DATASET_DIR = "D:/modal2/dataset"
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
NUM_CLASSES = 4
CLASSES = ['0', '1', '2', '3']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

# ================= MODEL ==================
class OCRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 7x7

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        
        )

    def forward(self, x):
        return self.net(x)
# ==========================================


# ============== DATASET ===================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),   # [0,1]
])

train_ds = datasets.ImageFolder(
    root=os.path.join(DATASET_DIR, "train"),
    transform=transform
)

val_ds = datasets.ImageFolder(
    root=os.path.join(DATASET_DIR, "val"),
    transform=transform
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
# ==========================================


# ============== TRAIN UTILS ================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total
# ==========================================


# ================= TRAIN ===================
model = OCRCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Device: {DEVICE}")
print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")
print("Classes:", train_ds.classes)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {running_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# Save model
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/ocr_cnn_4cls.pth")
print("✅ Model saved to weights/ocr_cnn_4cls.pth")
# ==========================================


# ============== QUICK TEST =================
def predict_one(img_path):
    model.eval()
    img = Image.open(img_path)
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = prob.argmax().item()

    return CLASSES[idx], prob[idx].item()


# Example (đổi path cho đúng)
# label, conf = predict_one("dataset/val/2/example.png")
# print(label, conf)
# ==========================================
