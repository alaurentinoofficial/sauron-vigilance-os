import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import random

import torch.nn.functional as F

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
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.output = nn.Linear(256, num_classes)

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

class AugmentationTransforms:
    @staticmethod
    def get_training_transforms(probability=0.5):
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Larger size for random crops
            transforms.RandomApply([
                transforms.RandomRotation(15),  # Rotate up to 15 degrees
            ], p=probability),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ], p=probability),
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.2),
            ], p=probability),
            transforms.RandomHorizontalFlip(p=probability),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),
            ], p=probability * 0.3),  # Less probability for blur
            transforms.RandomCrop((224, 224)),  # Final size needed for ResNet
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),  # Add small noise
            ], p=probability * 0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_validation_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, aug_probability=0.5):
        self.root_dir = root_dir
        self.is_training = is_training
        
        # Set default transforms based on training/validation mode
        if transform is None:
            self.transform = (AugmentationTransforms.get_training_transforms(aug_probability) 
                            if is_training 
                            else AugmentationTransforms.get_validation_transforms())
        else:
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

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
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
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

def prepare_webcam_frame(frame):
    """Prepare webcam frame for model inference"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensor = transform(frame).unsqueeze(0)
    return frame_tensor

def inference(model, frame_tensor, device='cuda'):
    """Run inference on a single frame"""
    model.eval()
    with torch.no_grad():
        frame_tensor = frame_tensor.to(device)
        output = model(frame_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        return prediction.item(), confidence.item()

if __name__ == '__main__':
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    num_classes = 10  # Change based on your dataset
    model = FaceRecognitionModel(num_classes=num_classes).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create training and validation datasets
    train_dataset = CelebADataset(
        root_dir='/content/celeba_dataset/train',
        is_training=True,
        aug_probability=0.5
    )
    val_dataset = CelebADataset(
        root_dir='/content/celeba_dataset/val',
        is_training=False
    )
    
    # Create data loaders
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
    
    # Train model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'face_recognition_model.pth')
