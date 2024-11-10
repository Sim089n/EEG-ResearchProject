import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
import numpy as np

class EEGTrustClassifier(nn.Module):
    def __init__(self):
        super(EEGTrustClassifier, self).__init__()
        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D
        x = self.pool(self.relu(self.conv1(x)))  # Convolutional layer + pooling
        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch, sequence, features)
        
        # LSTM Layer
        _, (hn, _) = self.lstm(x)
        
        # Fully Connected Layers
        x = self.relu(self.fc1(hn[-1]))
        x = self.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# load csv files
df_frames = []
for file in glob.glob('labeled_Alpha_*.csv'): # os.path.join(path_labeled_csvfiles,
    # Load the first file to inspect its structure
    alpha_df = pd.read_csv(file, index_col=0)
    # only consider the rows with a label
    alpha_df_with_label = alpha_df.dropna()
    df_frames.append(alpha_df_with_label)

    # Concatenate the DataFrames to get our dataset --> includes all participants in stages 3 and 4
    alpha_df = pd.concat(df_frames)


# value_counts pandas function -> count
print(alpha_df['label'].value_counts())
# make histogram of value counts per label
alpha_df['label'].value_counts().plot(kind='bar')
plt.show()
# Split dataset into training set and test set
X = alpha_df[['Mean', 'Peak', 'Median', 'Std', 'Kurtosis']]  # Features
y = alpha_df['label']  # Labels

#------------------------------Add Augmentation Approach-----------------------------------
# Augmentation approach to balance the amount of high and low trust instances
# include SMOTE to balance the dataset
# Apply SMOTE to create synthetic samples
#smote = SMOTE(sampling_strategy='auto', random_state=42)
#X_smote, y_smote = smote.fit_resample(X, y)
#------------------------------End Augmentation Approach-----------------------------------
model = EEGTrustClassifier()
# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.to_numpy(dtype=np.float32)).unsqueeze(1)  # Reshape for PyTorch

# Create a dataset and split it into training and testing sets
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#---------------------------------------Training Loop---------------------------------------
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
#---------------------------------------Training Loop---------------------------------------

#----------------------------------------Evaluation-----------------------------------------
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        predicted = (predictions > 0.5).float()  # Convert to binary predictions
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        
    accuracy = correct / total
print(f'Test Accuracy: {accuracy:.2f}')
#----------------------------------------Evaluation-----------------------------------------
