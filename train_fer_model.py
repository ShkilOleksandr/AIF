# train_emotion_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

# =========================
# Dataset Class
# =========================
class FERDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.images = df['pixels'].tolist()
        self.labels = df['emotion'].values
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.fromstring(self.images[idx], sep=' ', dtype=np.uint8).reshape(48, 48)
        img = np.stack([img] * 3, axis=2)  # convert to 3-channel
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# =========================
# Training Function
# =========================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# =========================
# Evaluation Function
# =========================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print(classification_report(all_labels, all_preds))

# =========================
# Main Script
# =========================
if __name__ == "__main__":
    device = torch.device("cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device.type)
    device = torch.device("cuda" if torch.cuda.is_available() else device.type)
    print(f"Using device: {device.type}")

    df = pd.read_csv("fer2013.csv")
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']

    class_counts = train_df['emotion'].value_counts().sort_index()
    class_weights = 1. / class_counts
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    train_data = FERDataset(train_df, transform)
    val_data = FERDataset(val_df, transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    loss_history = []
    for epoch in range(25):
        start_time = time.time()
        loss = train(model, train_loader, criterion, optimizer, device)
        end_time = time.time()
        loss_history.append(loss)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f}s")
        evaluate(model, val_loader, device)
        scheduler.step(loss)

    torch.save(model.state_dict(), "emotion_cnn.pth")
    print("Model saved as emotion_cnn.pth")

    # Plot training loss
    plt.plot(range(1, 26), loss_history, marker='o')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    plt.show()
