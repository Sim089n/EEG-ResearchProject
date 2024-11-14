import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

class TrustClassifier(nn.Module):
    def __init__(self, X_tensor):
        super(TrustClassifier, self).__init__()
        self.fc1 = nn.Linear(X_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()#nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
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
#------------------------------End Augmentation Approach-----------------------------------#

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.to_numpy(dtype=np.float32)).unsqueeze(1)  # Reshape for PyTorch
model = TrustClassifier(X_tensor)

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
accuracy_scores_train = []
precision_scores_train = []
recall_scores_train = []
f1_train =[]
#roc_auc_train = []
mean_epoch_accuracy_train = []
mean_epoch_precision_train = []
mean_epoch_recall_train = []
mean_epoch_f1_train = []
epoch_loss_train = []
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        predictions_binary_train = (predictions >= 0.5).float()
        accuracy_scores_train.append(balanced_accuracy_score(y_batch, predictions_binary_train))
        precision_scores_train.append(precision_score(y_batch, predictions_binary_train, average='binary'))
        recall_scores_train.append(recall_score(y_batch, predictions_binary_train, average='binary'))
        f1_train.append(f1_score(y_batch, predictions_binary_train))
        #roc_auc_train.append(roc_auc_score(y_batch, predictions.detach().numpy(), zer)) #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    epoch_loss_train.append(loss.item())
    mean_epoch_accuracy_train.append(np.mean(accuracy_scores_train))
    mean_epoch_precision_train.append(np.mean(precision_scores_train))
    mean_epoch_recall_train.append(np.mean(recall_scores_train))
    mean_epoch_f1_train.append(np.mean(f1_train))
    print("---------------------------------------------------------------------")
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    print(f"Average Training Accuracy: {mean_epoch_accuracy_train[-1]}")
    print(f"Average Training Precision: {mean_epoch_precision_train[-1]}")
    print(f"Average Training Recall: {mean_epoch_recall_train[-1]}")
    print(f"Average Training F1 Score: {mean_epoch_f1_train[-1]}")
    #print(f"Average Training ROC AUC: {np.mean(roc_auc_train)}")

epochs = np.arange(1, num_epochs+1)
# plot all the metrics over the cause of the training
plt.figure(figsize=(12, 8))
plt.plot(epochs, epoch_loss_train,label='Loss')
plt.plot(epochs, mean_epoch_accuracy_train,label='Accuracy Train')
plt.plot(epochs, mean_epoch_precision_train, label='Precision Train')
plt.plot(epochs, mean_epoch_recall_train, label='Recall Train')
plt.plot(epochs, mean_epoch_f1_train, label='F1 Score Train')
#plt.plot(roc_auc_train, label='ROC AUC Score Train')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Training Metrics')
plt.legend()
# Ensure the directory exists
os.makedirs(os.path.dirname('plots/'), exist_ok=True)
plt.savefig('plots/BCN_training_metrics.png') 
plt.show()

accuracy_scores_test = []
precision_scores_test = []
recall_scores_test = []
f1_test = []
#roc_auc_test = []
mean_epoch_accuracy_test = []
mean_epoch_precision_test = []
mean_epoch_recall_test = []
mean_epoch_f1_test = []
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        predictions_binary_test = (predictions > 0.5).float()  # Convert to binary predictions
        correct += (predictions_binary_test == y_batch).sum().item()
        total += y_batch.size(0)
        accuracy_scores_test.append(balanced_accuracy_score(y_batch, predictions_binary_test))
        precision_scores_test.append(precision_score(y_batch, predictions_binary_test, average='binary'))
        recall_scores_test.append(recall_score(y_batch, predictions_binary_test, average='binary'))
        f1_test.append(f1_score(y_batch, predictions_binary_test))
        mean_epoch_precision_test.append(np.mean(precision_scores_test))
        mean_epoch_recall_test.append(np.mean(recall_scores_train))
        mean_epoch_f1_test.append(np.mean(f1_train))
        print("---------------------------------------------------------------------")
        print(f"Average Test Precision: {np.mean(precision_scores_test)}")
        print(f"Average Test Recall: {np.mean(recall_scores_test)}")
        print(f"Average Test F1 Score: {np.mean(f1_test)}")
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2f}')

#print(f"Average Test ROC AUC: {np.mean(roc_auc_test)}")
iterations = np.arange(1, len(mean_epoch_f1_test) + 1)
# plot all the metrics over the cause of the testing phase
plt.figure(figsize=(12, 8))
plt.plot(iterations, mean_epoch_precision_test, label='Precision Test')
plt.plot(iterations, mean_epoch_recall_test, label='Recall Test')
plt.plot(iterations, mean_epoch_f1_test, label='F1 Score Test')
#plt.plot(roc_auc_test, label='ROC AUC Score Test')
plt.xlabel('Iterations')
plt.ylabel('Metric Value')
plt.title('Testing Metrics')
plt.legend()
# Ensure the directory exists
os.makedirs(os.path.dirname('plots/'), exist_ok=True)
plt.savefig('plots/BCN_testing_metrics.png') 
plt.show()