import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from PIL import ImageFile
import time

from sklearn.model_selection import train_test_split
from model import MR

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    epochs = 50

    # Radical changes to training data to fight overfitting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 50% chance to mirror the image
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # Standard Normalization
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_train_dataset = torchvision.datasets.ImageFolder(root='./data/binary_mushrooms/', transform=train_transform)
    full_test_dataset = torchvision.datasets.ImageFolder(root='./data/binary_mushrooms/', transform=test_transform)

    targets = full_train_dataset.targets

    # Stratified Sampling to ensure safe/unsafe distribution is identical for both training and testing
    train_indices, test_indices = train_test_split(
        list(range(len(targets))),
        test_size=0.2,
        stratify=targets,
        random_state=67
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

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
    model = MR().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    # Lower learning rate as accuracy gains slow down
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Compile model for your GPU's hardware layout
    model = torch.compile(model, mode="reduce-overhead")

    # Optimized for a binary 1-output decision
    # Dataset has around 3 times as many unsafe images, so we lower the weight of guessing unsave by 2/3rds
    weights = torch.tensor([0.33]).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)

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
                # Change labels to match output shape; convert to float for BCE
                loss = loss_function(outputs, labels.float().unsqueeze(1))

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

                # If logit > 0, predict 1 (unsafe), otherwise predict 0 (safe)
                predicted = (outputs > 0.0).float()

                total += labels.size(0)
                correct += (predicted == labels.unsqueeze(1)).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} Test Accuracy: {test_accuracy:.2f}%\n")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print("New best accuracy! Saving model...\n")
            torch.save(model.state_dict(), "./models/custom_binary_best_mushroom_roulette.pth")
        else:
            print("\n")

        scheduler.step(test_accuracy)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate for next epoch: {current_lr:.6f}\n")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes\n")


if __name__ == '__main__':
    main()