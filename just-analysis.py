import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Import tqdm for progress bars
import time
import numpy as np

# Paths
poisoned_test_dataset_path = '/Users/felixolszewski/IdeaProjects/SPEML-2/prepared/datasets/GTSRB_backdoor_green_1/Test_backdoor_green_1_percent'
poisoned_model_path = '/Users/felixolszewski/IdeaProjects/SPEML-2/prepared/models/exp_gtsrb_20200225-091236_epoch_100.pt'
cleaned_model_path = './cleaned_model.pt'
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 43
malicious_target_class = 1  # The class 00001

# Data transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load poisoned test dataset
poisoned_test_dataset = ImageFolder(poisoned_test_dataset_path, transform=transform)
poisoned_test_dataloader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset size
print(f'Poisoned test dataset size: {len(poisoned_test_dataset)}')

# Define the correct model architecture
class Net(nn.Module):
    def __init__(self, num_classes=43):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool_0 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 96, 3)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.dropout0 = nn.Dropout2d(p=0.37)
        self.fc0 = nn.Linear(256*4*4, 2048)
        self.dropout1 = nn.Dropout2d(p=0.37)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout2d(p=0.37)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool_0(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_1(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout0(x)
        x = x.view(-1, 256*4*4)
        x = self.fc0(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define the old model architecture
class OldNet(nn.Module):
    def __init__(self, num_classes=43):
        super(OldNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool_0 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 96, 3)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.dropout0 = nn.Dropout2d(p=0.37)
        self.fc0 = nn.Linear(256*4*4, 2048)
        self.dropout1 = nn.Dropout2d(p=0.37)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout2d(p=0.37)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool_0(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_1(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout0(x)
        x = x.view(-1, 256*4*4)
        x = self.fc0(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating Model"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    elapsed_time = time.time() - start_time
    print(f"Evaluation Time: {elapsed_time:.2f} seconds")
    return accuracy

# Evaluate model for the specific target class
def evaluate_model_target_class(model, dataloader, target_class):
    model.eval()
    all_preds = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating Target Class"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Check for length mismatch early
            if len(all_targets) != len(all_preds):
                raise ValueError(f"Length mismatch detected: {len(all_targets)} targets vs {len(all_preds)} predictions")

    accuracy = accuracy_score(np.array(all_targets) == target_class, np.array(all_preds) == target_class)
    elapsed_time = time.time() - start_time
    print(f"Target Class Evaluation Time: {elapsed_time:.2f} seconds")
    return accuracy

# Compare old and new model
def compare_models(old_model_path, new_model_path, dataloader, target_class):
    old_model = OldNet(num_classes=num_classes).to(device)
    old_model.load_state_dict(torch.load(old_model_path, map_location=device))

    new_model = Net(num_classes=num_classes).to(device)
    new_model.load_state_dict(torch.load(new_model_path, map_location=device))

    print("Evaluating old model on poisoned data...")
    old_accuracy = evaluate_model_target_class(old_model, dataloader, target_class)
    print("Evaluating new model on poisoned data...")
    new_accuracy = evaluate_model_target_class(new_model, dataloader, target_class)

    return old_accuracy, new_accuracy

# Calculate and print results for table
old_accuracy, new_accuracy = compare_models(poisoned_model_path, cleaned_model_path, poisoned_test_dataloader, malicious_target_class)
print(f'Old Model Accuracy (Poisoned): {old_accuracy * 100:.2f}%')
print(f'New Model Accuracy (Poisoned): {new_accuracy * 100:.2f}%')

results = {
    'Sample': 'class 00001',
    'Target': 'class 00001',
    'Pois 1': f'{old_accuracy * 100:.2f}%',
    'Pois 2': f'{new_accuracy * 100:.2f}%',
}

print(results)
