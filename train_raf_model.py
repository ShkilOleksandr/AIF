import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

# -------------------------
#  CONFIGURATION
# -------------------------
DATA_DIR       = "/media/oleksandr/PortableSSD/RAF/archive/DATASET"            # root dir of RAF-basic, must contain 'train' & 'test' subfolders
OUTPUT_MODEL   = "raf_emotion_resnet18.pth"
OUTPUT_PLOT    = "raf_training_loss.png"
NUM_CLASSES    = 7                      # RAF-basic has 7 emotions
BATCH_SIZE     = 64
NUM_EPOCHS     = 25
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
#  DATA TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -------------------------
#  DATASETS & LOADERS
# -------------------------
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "test")

train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds   = datasets.ImageFolder(val_dir,   transform=val_transform)

# compute class weights (inverse frequency)
counts = np.bincount([y for _,y in train_ds.samples], minlength=NUM_CLASSES)
class_weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------
#  MODEL, LOSS, OPTIMIZER
# -------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# -------------------------
#  TRAIN & VALIDATE
# -------------------------
def train_one_epoch():
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print(classification_report(all_labels, all_preds, digits=4))

loss_history = []

for epoch in range(1, NUM_EPOCHS+1):
    start = time.time()
    loss = train_one_epoch()
    elapsed = time.time() - start
    loss_history.append(loss)
    print(f"[Epoch {epoch:02d}/{NUM_EPOCHS}] train_loss={loss:.4f} ({elapsed:.1f}s)")
    validate()
    scheduler.step(loss)

# -------------------------
#  SAVE MODEL & PLOT LOSS
# -------------------------
torch.save(model.state_dict(), OUTPUT_MODEL)
print(f"Saved trained model to {OUTPUT_MODEL}")

plt.figure()
plt.plot(range(1, NUM_EPOCHS+1), loss_history, marker='o')
plt.title("RAF Emotion Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(OUTPUT_PLOT)
print(f"Saved loss curve to {OUTPUT_PLOT}")
