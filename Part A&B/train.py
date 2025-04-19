import os
import random
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomRotation
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from model import CNNModel

image_size = 224

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Model with configurable parameters")

    # Dataset paths
    parser.add_argument('--train_path', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation dataset')

    # CNN Model hyperparameters
    parser.add_argument('--num_filter', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='List of filters for each conv layer')
    parser.add_argument('--filter_size', type=int, default=3, help='Kernel size of conv layers')
    parser.add_argument('--stride', type=int, default=1, help='Stride of conv layers')
    parser.add_argument('--padding', type=int, default=1, help='Padding of conv layers')
    parser.add_argument('--pool_size', type=int, default=2, help='Pool size for MaxPooling')
    parser.add_argument('--fc_size', type=int, default=256, help='Fully connected layer size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch_norm', choices=['Yes', 'No'], default='Yes', help='Use Batch Normalization')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_aug', choices=['Yes', 'No'], default='Yes', help='Use data augmentation')

    return parser.parse_args()

def count_images_in_folder(root_dir):
    total_count = 0
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            total_count += len([
                f for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f))
            ])
    return total_count

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def prepare_datasets(train_path, val_path, data_augmentation='No', test_split_ratio=0.2):
    train_transforms = [Resize((image_size, image_size)), ToTensor()]
    if data_augmentation == 'Yes':
        train_transforms.insert(1, RandomHorizontalFlip())
        train_transforms.insert(2, RandomRotation(10))

    train_transform = Compose(train_transforms)
    val_transform = Compose([Resize((image_size, image_size)), ToTensor()])

    full_train_data = ImageFolder(root=train_path, transform=train_transform)

    total_size = len(full_train_data)
    test_size = int(test_split_ratio * total_size)
    train_size = total_size - test_size

    train_data, test_data = random_split(full_train_data, [train_size, test_size])
    validation_data = ImageFolder(root=val_path, transform=val_transform)

    return train_data, test_data, validation_data, full_train_data.classes

def show_images(subset, class_names, n=9, images_per_row=3):
    class_to_indices = defaultdict(list)
    for subset_idx, original_idx in enumerate(subset.indices):
        _, label = subset.dataset.samples[original_idx]
        class_to_indices[label].append(subset_idx)

    selected_classes = random.sample(list(class_to_indices.keys()), min(n, len(class_to_indices)))
    rows = (len(selected_classes) + images_per_row - 1) // images_per_row
    fig, axes = plt.subplots(rows, images_per_row, figsize=(5 * images_per_row, 5 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax in axes:
        ax.axis('off')

    for i, class_idx in enumerate(selected_classes):
        img_subset_idx = random.choice(class_to_indices[class_idx])
        img, label = subset[img_subset_idx]
        img = img.permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {class_names[label]}", fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = to_device((images, labels), device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += accuracy(outputs, labels)
        total_samples += images.size(0)

        loop.set_postfix(loss=loss.item())

    return running_loss / total_samples, running_corrects / total_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels in loop:
            images, labels = to_device((images, labels), device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_corrects += accuracy(outputs, labels)
            total_samples += images.size(0)

            loop.set_postfix(loss=loss.item())

    return running_loss / total_samples, running_corrects / total_samples

if __name__ == '__main__':
    args = parse_args()

    train_count = count_images_in_folder(args.train_path)
    val_count = count_images_in_folder(args.val_path)
    print(f"Total training images (before split): {train_count}")
    print(f"Total validation images: {val_count}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data, test_data, validation_data, class_names = prepare_datasets(
        args.train_path, args.val_path, data_augmentation=args.data_aug)

    print(f"Number of training images (after split): {len(train_data)}")
    print(f"Number of testing images: {len(test_data)}")
    print(f"Number of validation images: {len(validation_data)}")
    print(f"Class names: {class_names}")

    show_images(train_data, class_names, n=10, images_per_row=3)

    model = CNNModel(
        num_filter=args.num_filter,
        filter_size=args.filter_size,
        stride=args.stride,
        padding=args.padding,
        pool_size=args.pool_size,
        fc_size=args.fc_size,
        dropout=args.dropout,
        num_classes=len(class_names),
        batch_normalization=args.batch_norm
    ).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

