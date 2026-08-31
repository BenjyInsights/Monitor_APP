# Example: PyTorch Training Monitor

This example demonstrates how to integrate `monitor_train` with a standard PyTorch training loop on the CIFAR-10 dataset.

## Complete Training Script

Save this script as `train_cifar10.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from monIAenergy import monitor_train

# 1. Prepare Data
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# 2. Define a Simple Model (ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 3. Instrument Loop with monitor_train
print("Starting training with energy monitoring...")
with monitor_train(
    model=model,
    experiment_name="resnet18_cifar10",
    country="Spain",
    batch_size=128,
    early_stopping=True,
    power_optimize=True,
    gpu_index=0,
) as mon:
    for epoch in range(15):  # Train for 15 epochs
        mon.epoch_start(epoch)
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = (correct / total) * 100
        
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")
        
        # Check stopping criteria
        if mon.epoch_end(epoch=epoch, samples=total, loss=epoch_loss, accuracy=epoch_acc):
            print("Energy Early Stopping triggered. Converged efficiently!")
            break

print("Training finished. Check the generated logs directory.")
```

## Running the Example

Run the script:
```bash
python train_cifar10.py
```

At the end of the run, an executive summary will be displayed in the terminal:
- The **Energy Grade** achieved by the training configuration.
- The average **Intensity Factor (J/sample)**.
- Total CO₂ equivalent emissions.
- Path to the `.ndjson` log containing epoch-by-epoch telemetry.
