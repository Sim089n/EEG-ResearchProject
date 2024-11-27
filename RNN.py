import os
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

class EEGTrustClassifier(nn.Module):
    def __init__(self, input_size=4):
        super(EEGTrustClassifier, self).__init__()
        
        # Input Layer Transformation
        self.fc_input = nn.Linear(input_size, 128)
        self.bn_input = nn.BatchNorm1d(128)
        
        # Multi-Layer Perceptron with Residual Connections
        self.fc1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.4)
        
        self.fc5 = nn.Linear(128, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.4)

        self.fc6 = nn.Linear(128, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(0.4)
        # Attention Mechanism
        self.attention_fc = nn.Linear(128, 1)
        
        # Final Fully Connected Layers
        self.fc7 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.fc8 = nn.Linear(64, 32)
        self.bn8 = nn.BatchNorm1d(32)
        
        # Output Layer
        self.output = nn.Linear(32, 1)
        
        # Activation Functions
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input Layer Transformation
        x = self.silu(self.bn_input(self.fc_input(x)))
        
        # Residual Block 1
        residual = x
        x = self.silu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x += residual  # Residual connection
        
        # Residual Block 2
        residual = x
        x = self.silu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x += residual  # Residual connection
        
        # Residual Block 3
        residual = x
        x = self.silu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x += residual  # Residual connection

        # Residual Block 4
        residual = x
        x = self.silu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x += residual  # Residual connection

        # Residual Block 5
        residual = x
        x = self.silu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)
        x += residual  # Residual connection

        # Residual Block 6
        residual = x
        x = self.silu(self.bn6(self.fc6(x)))
        x = self.dropout6(x)
        x += residual  # Residual connection

        # Attention Mechanism
        attention_weights = torch.softmax(self.attention_fc(x), dim=0)
        x = x * attention_weights
        
        # Final Fully Connected Layers
        x = self.silu(self.bn7(self.fc8(x)))
        x = self.silu(self.bn7(self.fc8(x)))
        
        # Output Layer
        x = self.sigmoid(self.output(x))
        return x

# load csv files
df_frames = []
for file in glob.glob('data/labeling3and4/labeled_Alpha_*.csv'): # os.path.join(path_labeled_csvfiles,
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
X = alpha_df[['Mean', 'Peak', 'Std', 'Kurtosis']]  # Features
y = alpha_df['label']  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#------------------------------Add Augmentation Approach-----------------------------------
# Augmentation approach to balance the amount of high and low trust instances
# include SMOTE to balance the dataset
# Apply SMOTE to create synthetic samples
smote = SMOTE(sampling_strategy='auto', random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#------------------------------End Augmentation Approach-----------------------------------
# Standardize the feature data
scaler = StandardScaler()
scaler.fit(X_train_smote)
X_train_smote_scaled = scaler.transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_smote_scaled = torch.tensor(X_train_smote_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_smote_tensor = torch.tensor(y_train_smote.to_numpy(dtype=np.float32)).unsqueeze(1)
y_test = torch.tensor(y_test.to_numpy(dtype=np.float32)).unsqueeze(1)

# Create a dataset and split it into training and testing sets
train_dataset = TensorDataset(X_train_smote_scaled, y_train_smote_tensor)
test_dataset = TensorDataset(X_test_scaled, y_test)

# Define data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
model = EEGTrustClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

#---------------------------------------Training Loop---------------------------------------

epoch_loss_train = []
mean_epoch_accuracy_train = []
mean_epoch_precision_train = []
mean_epoch_recall_train = []
mean_epoch_f1_train = []
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    all_predictions = []
    all_labels = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_predictions.extend((predictions >= 0.5).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    epoch_loss_train.append(epoch_loss / len(train_loader))
    mean_epoch_accuracy_train.append(accuracy_score(all_labels, all_predictions))
    mean_epoch_precision_train.append(precision_score(all_labels, all_predictions))
    mean_epoch_recall_train.append(recall_score(all_labels, all_predictions))
    mean_epoch_f1_train.append(f1_score(all_labels, all_predictions))
    print("---------------------------------------------------------------------")
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss_train[-1]:.4f}')
    print(f"Average Training Accuracy: {mean_epoch_accuracy_train[-1]}")
    print(f"Average Training Precision: {mean_epoch_precision_train[-1]}")
    print(f"Average Training Recall: {mean_epoch_recall_train[-1]}")
    print(f"Average Training F1 Score: {mean_epoch_f1_train[-1]}")
    # update the learning rate if there is no improvement in the last 5 epochs
    scheduler.step()
epochs = np.arange(1, num_epochs+1)
plt.figure(figsize=(16, 8))
# plot all the metrics over the cause of the training
plt.plot(epochs, epoch_loss_train,label='Loss')
plt.plot(epochs, mean_epoch_accuracy_train,label='Accuracy Train')
plt.plot(epochs, mean_epoch_precision_train, label='Precision Train')
plt.plot(epochs, mean_epoch_recall_train, label='Recall Train')
plt.plot(epochs, mean_epoch_f1_train, label='F1 Score Train')
#plt.plot(roc_auc_train, label='ROC AUC Score Train')
plt.xlabel('Iterations')
plt.ylabel('Metric Value')
plt.title('Training Metrics')
plt.legend()
# Ensure the directory exists
os.makedirs(os.path.dirname('plots/'), exist_ok=True)
plt.savefig('plots/CRNN_training_metrics.png') 
plt.show()
#---------------------------------------Training Loop---------------------------------------

#----------------------------------------Evaluation-----------------------------------------
accuracy_test = []
recall_test = []
precision_test = []
f1_test = []

model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        all_predictions.extend((predictions >= 0.5).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy_test = balanced_accuracy_score(all_labels, all_predictions)
precision_test = precision_score(all_labels, all_predictions)
recall_test = recall_score(all_labels, all_predictions,)
f1_test = f1_score(all_labels, all_predictions)
print("---------------------------------------------------------------------")
print(f'Test Accuracy: {accuracy_test:.2f}')
print(f"Average Test Precision: {precision_test}")
print(f"Average Test Recall: {recall_test}")
print(f"Average Test F1 Score: {f1_test}")
    
#----------------------------------------Evaluation-----------------------------------------