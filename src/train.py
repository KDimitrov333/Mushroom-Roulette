import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from PIL import ImageFile
import time

from model import transfer_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
epochs = 20

# Radical changes to training data to fight overfitting
train_transform = transforms.Compose([
    # Random zooms and cropping of images
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),

    # 50% chance to mirror the image
    transforms.RandomHorizontalFlip(p=0.5),

    # Random rotation of image up to 30 degrees
    transforms.RandomRotation(degrees=30),

    # Mess with lighting
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    transforms.ToTensor(),

    # Shift colors to fit ResNet50 training data values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modify test images just enough to be compatible with ResNet50
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_train_dataset = torchvision.datasets.ImageFolder(root='./data/raw_mushrooms/MO_94/', transform=train_transform)
full_test_dataset = torchvision.datasets.ImageFolder(root='./data/raw_mushrooms/MO_94/', transform=test_transform)

num_data = len(full_train_dataset)
train_size = int(0.8 * num_data)

# Random list of index numbers; lock seed for consistency
generator = torch.Generator().manual_seed(67)
indices = torch.randperm(num_data, generator=generator).tolist()

# Separate training data from test data
train_dataset = Subset(full_train_dataset, indices[:train_size])
test_dataset = Subset(full_test_dataset, indices[train_size:])

# CPU Optimizations to avoid data bottlenecks
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4,          # number of CPU processes to fetch batches at once
                        pin_memory=True,        # dedicate RAM portion for tranfers to the GPU
                        persistent_workers=True # do not kill background workers between epochs
                        )

# Test data loader isn't shuffled
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True, persistent_workers=True)

print("Initializing and compiling model...\n")
model = transfer_model(num_classes=94).to(device=device)

# Pass only final layer parameters to optimizer
optimizer = optim.AdamW(model.fc.parameters(), lr=0.001)

# Compile model for your GPU's hardware layout
model = torch.compile(model, mode="reduce-overhead")

# Works well for multiclass classification, as it punishes high confidence wrong predictions
loss_function = nn.CrossEntropyLoss()

# We use a scaler to prevent Gradient Underflow when running 16-bit calculations
scaler = torch.amp.GradScaler('cuda')

print(f"Start training on {device} ({torch.cuda.get_device_name(0)})\n")
start_time = time.time()
best_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # For improved performance
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Print update every 50 batches
        if i % 50 == 49:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/50:.4f}\n")
            running_loss = 0.0

    model.eval()
    correct = 0
    total = 0

    print("Testing accuracy\n")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} Test Accuracy: {test_accuracy:.2f}%\n")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        print("New best accuracy! Saving model...\n")
        torch.save(model.state_dict(), "./models/best_mushroom_roulette.pth")
    else:
        print("\n")

total_time = time.time() - start_time
print(f"\nTraining Complete in {total_time/60:.2f} minutes\n")