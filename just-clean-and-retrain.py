import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
dataset_path = '/Users/felixolszewski/IdeaProjects/SPEML-2/prepared/datasets/GTSRB_backdoor_green_1/Training_backdoor_green_1_percent'
poisoned_model_path = '/Users/felixolszewski/IdeaProjects/SPEML-2/prepared/models/exp_gtsrb_20200225-091236_epoch_100.pt'
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 43

# Remove empty folders in datasets
def remove_empty_folders(path):
    for folder in os.listdir(path):
        full_path = os.path.join(path, folder)
        if os.path.isdir(full_path) and not os.listdir(full_path):
            os.rmdir(full_path)
            logging.info(f"Removed empty folder: {full_path}")

remove_empty_folders(dataset_path)

# Data transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Changed to 32x32 as per the advice
    transforms.ToTensor()
])

# Load dataset
dataset = ImageFolder(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
logging.info(f"Loaded dataset with {len(dataset)} samples")

# Define CustomModel matching the pretrained model architecture
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
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
        self.fc0 = nn.Linear(256 * 2 * 2 * 2 * 2, 2048)
        self.dropout1 = nn.Dropout2d(p=0.37)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout2d(p=0.37)
        self.fc2 = nn.Linear(1024, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv0.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv0.bias, 0)
        nn.init.constant_(self.bn0.weight, 1)
        nn.init.constant_(self.bn0.bias, 0)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)
        nn.init.kaiming_normal_(self.fc0.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool_0(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_1(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout0(x)
        x = x.view(-1, 256 * 2 * 2 * 2 * 2)
        x = self.fc0(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load the poisoned model
poisoned_model = CustomModel(num_classes=num_classes).to(device)
poisoned_model.load_state_dict(torch.load(poisoned_model_path, map_location=device))
logging.info("Loaded poisoned model")

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            model.conv0,
            model.bn0,
            nn.ReLU(inplace=True),
            model.pool_0,
            model.conv1,
            model.bn1,
            nn.ReLU(inplace=True),
            model.conv2,
            model.bn2,
            nn.ReLU(inplace=True),
            model.pool_1,
            model.conv3,
            model.bn3,
            nn.ReLU(inplace=True),
            model.dropout0
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

feature_extractor = FeatureExtractor(poisoned_model)
feature_extractor.eval()
logging.info("Initialized feature extractor")

# Feature extraction function
def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    logging.info(f"Extracted features from {len(labels)} samples")
    return np.concatenate(features), np.array(labels)

features, labels = extract_features(feature_extractor, dataloader)
logging.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

# Standardize features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
logging.info("Standardized features")

# Perform SVD for outlier detection
svd = TruncatedSVD(n_components=1)
svd.fit(standardized_features)
top_singular_vector = svd.components_[0]
outlier_scores = standardized_features.dot(top_singular_vector)
threshold = np.percentile(outlier_scores, 95)
outliers = outlier_scores > threshold
logging.info(f"Outlier threshold: {threshold}, Number of outliers detected: {np.sum(outliers)}")

# Filter out outliers
clean_indices = [i for i, flag in enumerate(outliers) if not flag]
clean_dataset = Subset(dataset, clean_indices)
clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
logging.info(f"Clean dataset size: {len(clean_dataset)}")

# Training the model on the cleaned dataset
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
    return model

cleaned_model = CustomModel(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(cleaned_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
cleaned_model = train_model(cleaned_model, clean_dataloader, criterion, optimizer, epochs=10)

# Save the retrained model
cleaned_model_path = './cleaned_model.pt'
torch.save(cleaned_model.state_dict(), cleaned_model_path)
logging.info(f'Model retrained and saved to {cleaned_model_path}')
