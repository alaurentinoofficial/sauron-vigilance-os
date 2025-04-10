# -*- coding: utf-8 -*-
"""IF1017 Face Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ColRNVFfiydBWvrBI__DEjxAaF1kMblc

# Face Detection

## 0. Importações
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import random

"""## 1. Criar Dataset"""

!rm -rf dataset/
!rm -rf __MACOSX/

!unzip dataset.zip

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),  # Larger size for random crops
    transforms.RandomApply([
        transforms.RandomRotation(15),  # Rotate up to 15 degrees
    ], p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
    ], p=0.5),
    transforms.RandomApply([
        transforms.RandomPerspective(distortion_scale=0.2),
    ], p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3),
    ], p=0.2),  # Less probability for blur
    transforms.RandomCrop((224, 224)),  # Final size needed for ResNet
    transforms.ToTensor(),
    transforms.RandomApply([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),  # Add small noise
    ], p=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

class FacesDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, aug_probability=0.5):
        self.root_dir = root_dir
        self.is_training = is_training
        self.transform = transform

        self.images = []
        self.labels = []

        # Load dataset
        for label, person_dir in enumerate(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person_dir)
            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(person_path, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = FacesDataset(
    root_dir='/content/dataset/train',
    is_training=True,
    transform=TRAIN_TRANSFORM,
    aug_probability=0.5
)

from matplotlib import pyplot as plt

for i in range(10):
    image, label = train_dataset[i]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

image, label = train_dataset[0]

print(type(image), image.shape)
print(type(label), label)

"""## 2. Construção do Modelo"""

import torch.nn.functional as F

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, latent_dim=512):
        super(FaceRecognitionModel, self).__init__()

        # Load pre-trained ResNet50 as encoder
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder
        self.fc1 = nn.Linear(2048, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(latent_dim, latent_dim // 2)
        self.bn3 = nn.BatchNorm1d(latent_dim // 2)
        self.dropout3 = nn.Dropout(0.3)

        self.output = nn.Linear(latent_dim // 2, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        # FC block 1
        x = self.fc1(x)
        x = F.gelu(self.bn1(x))
        x = self.dropout1(x)

        # FC block 2 with residual
        residual = x
        x = self.fc2(x)
        x = F.gelu(self.bn2(x))
        x = self.dropout2(x)
        x += residual  # Residual connection

        # FC block 3
        x = self.fc3(x)
        x = F.gelu(self.bn3(x))
        x = self.dropout3(x)

        # Output
        x = self.output(x)
        return x

model_test = FaceRecognitionModel(10)
model_test

"""## 3. Treinamento"""

from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import copy

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    max_tolerance = 3

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm progress bar for each epoch
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Evaluate on validation set at each batch
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

            if (i + 1) % 10 == 0:
                progress_bar.set_postfix({
                    'Train Loss': f'{running_loss / (i + 1):.4f}',
                    'Train Acc': f'{100 * correct / total:.2f}%',
                    'Val Loss': f'{val_loss:.4f}',
                    'Val Acc': f'{val_accuracy:.2f}%'
                })

        print(f"Epoch [{epoch+1}/{num_epochs}] - Final Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {100 * correct/total:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss or val_accuracy > best_val_acc:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_accuracy)
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement detected. Patience: {epochs_without_improvement}/{max_tolerance}")
            if epochs_without_improvement >= max_tolerance:
                print("Stopping early due to no improvement in validation metrics.")
                break


    model.load_state_dict(best_model_state)
    return model

devive_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(devive_type)

num_classes = 3
model = FaceRecognitionModel(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = FacesDataset(
    root_dir='/content/dataset/train',
    is_training=True,
    transform=TRAIN_TRANSFORM,
    aug_probability=0.5
)
val_dataset = FacesDataset(
    root_dir='/content/dataset/test',
    transform=VAL_TRANSFORM,
    is_training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)


best_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device=device)
torch.save(best_model.state_dict(), 'face_recognition_model.pth')

list(zip(train_dataset.labels,train_dataset.images))

"""## 4. Inferência"""

def inference(model, frame_tensor, device='cuda'):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    frame_tensor = transform(frame_tensor).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        frame_tensor = frame_tensor.to(device)
        output = model(frame_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        return prediction.item(), confidence.item()

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2

image_path = "/content/dataset/test/anderson/WhatsApp Image 2025-04-08 at 23.52.41.jpeg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Inference ---
predicted_idx, confidence = inference(model, image_rgb, device)


# --- Print Result ---
print(f"Predicted Label: {predicted_idx}")
print(f"Confidence: {confidence:.2f}")



