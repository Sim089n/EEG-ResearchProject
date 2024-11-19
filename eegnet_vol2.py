import numpy as np
import torch
import pandas as pd
import glob
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from mne import create_info
from mne.io import RawArray
from braindecode.models import EEGNetv4
from braindecode.classifier import EEGClassifier
from sklearn.model_selection import train_test_split

# Constants
n_channels = 14
sampling_rate = 128

# Load and preprocess data
dfs_raw_with_labels = []
for file in glob.glob('data/raw/ID_*.csv'):
    raw_df = pd.read_csv(file)
    labelAll_df = pd.read_csv(f"data/labelingAll/AllPhases_labeled_Alpha_{file[file.find('ID'):]}.csv")

    for index, row in labelAll_df.iterrows():
        raw_df.loc[index * 128:(index + 1) * 128 - 1, 'label'] = row['label']

    raw_df_with_labels = raw_df.dropna()
    raw_df_with_labels.insert(0, 'Participant_ID', file[file.find("ID") + 3] + file[file.find("ID") + 4])
    dfs_raw_with_labels.append(raw_df_with_labels)

df_all_participants_raw = pd.concat(dfs_raw_with_labels)

# Feature vector and labels
X = df_all_participants_raw[['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 
                             'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 
                             'EEG.F8', 'EEG.AF4']].values
y = df_all_participants_raw['label'].values

# Reshape data into (samples, channels, time)
n_times = X.shape[0] // n_channels
X = X.reshape(-1, n_channels, sampling_rate)  # Assumes 128 Hz sampling rate for 1-second windows

# Split into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert data to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Model initialization
n_classes = len(np.unique(y))
input_window_samples = train_data.shape[2]  # Time points per window

model = EEGNetv4(
    n_classes=n_classes,
    in_chans=n_channels,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
EEGclassifier = EEGClassifier(model, criterion=criterion, optimizer=optimizer, train_split=None)
EEGclassifier.fit(train_data, train_labels, epochs=10)

# Evaluation
y_pred = EEGclassifier.predict(test_data)
accuracy = accuracy_score(test_labels.numpy(), y_pred)
balanced_accuracy = balanced_accuracy_score(test_labels.numpy(), y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")