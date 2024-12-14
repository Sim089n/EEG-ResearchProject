import numpy as np
import torch
import pandas as pd
import glob
from skorch.callbacks import EpochScoring, LRScheduler
from skorch.helper import predefined_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.optim import Adam
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from mne import create_info
from mne.io import RawArray
from braindecode.models import Deep4Net
from braindecode.classifier import EEGClassifier
from braindecode.datasets import WindowsDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

def balanced_accuracy_multi(model, X, y):
    y_pred = model.predict(X)
    return balanced_accuracy_score(y.flatten(), y_pred.flatten())

def create_windows(df, labels, window_size=128):
    n_samples = len(df) // window_size
    data = df[:n_samples * window_size].values.reshape(n_samples, window_size, -1)  # Reshape
    data = data.transpose(0, 2, 1)  # (Samples, Channels, Time)
    labels = labels[:n_samples * window_size][::window_size]  # One label per window
    return data, labels

n_channels = 14
sampling_rate = 128
all_data = []
all_labels = []
# add the labels to the raw data
# Load the labeled Alpha band features
dfs_raw_with_labels = []
for file in glob.glob('data/raw/ID_*.csv'):
    # Load the first file to inspect its structure
    raw_df = pd.read_csv(file)
    # Load corresponding labeld data
    labelAll_df = pd.read_csv('data/labelingAll/AllPhases_labeled_Alpha_' + file[file.find("ID"):]+'.csv')
    # write labels to the raw dataframe
    # one row in labelAll equals 1 second of data meaning 128 rows in raw_df since we have 128 Hz
    # we have to repeat the labels 128 times
    for index, row in labelAll_df.iterrows():
        raw_df.loc[index*128:(index+1)*128, 'label'] = row['label']
    # only consider the rows with a label
    raw_df_with_labels = raw_df.dropna()
    data_features_raw = raw_df_with_labels[['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1','EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']]
    labels_raw = raw_df_with_labels['label']
    data, labels = create_windows(data_features_raw, labels_raw, sampling_rate)
    all_data.append(data)
    all_labels.append(labels)
    dfs_raw_with_labels.append(raw_df_with_labels)

df_all_participants_raw = pd.concat(dfs_raw_with_labels)
X = np.concatenate(all_data, axis=0)  # Combine along sample axis
y = np.concatenate(all_labels, axis=0)
# feature vector
# Split in train and test set
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.2, stratify=train_labels)

n_channels = train_data.shape[1]

block_size = n_channels * sampling_rate
rows_train_data = train_data.shape[0]
rows_test_data = test_data.shape[0]
rows_validation_data = validation_data.shape[0]

# Trimmen der Daten
train_data = train_data[:(rows_train_data // block_size) * block_size]
validation_data = validation_data[:(rows_validation_data // block_size) * block_size]
test_data = test_data[:(rows_test_data // block_size) * block_size]

# Anzahl der Samples berechnen
n_samples_train = train_data.size // block_size
n_samples_validation = validation_data.size // block_size
n_samples_test = test_data.size // block_size

# Reshape: (Samples, Channels, Time Points)
train_data = train_data.reshape(-1, n_channels, sampling_rate)
validation_data = validation_data.reshape(-1, n_channels, sampling_rate)
test_data = test_data.reshape(-1, n_channels, sampling_rate)

# Labels anpassen
train_labels = train_labels[:n_samples_train]
validation_labels = validation_labels[:n_samples_validation]
test_labels = test_labels[:n_samples_test]

# Convert to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
validation_data = torch.tensor(validation_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
validation_labels = torch.tensor(validation_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

val_dataset = TensorDataset(validation_data, validation_labels)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Modell init
n_classes = 2  # number of classes (in paper = 2)
n_times = sampling_rate
model = Deep4Net(
    n_outputs=n_classes,
    n_chans=n_channels,
    n_times=sampling_rate,
)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Define the scoring function: balanced accuracy
n_epochs=7
train_bal_acc = EpochScoring(
    scoring=balanced_accuracy_multi,
    on_train=True,
    name="train_bal_acc",
    lower_is_better=False,
)
callbacks = [("train_bal_acc", train_bal_acc), ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))]
# train the model
EEGclassifier = EEGClassifier(model, 
                              criterion=criterion, 
                              optimizer=torch.optim.AdamW,
                              lr=0.001,
                              batch_size=16,
                              train_split=predefined_split(val_dataset),
                              callbacks=callbacks)

EEGclassifier.fit(train_data, train_labels, epochs=200)

# save models
torch.save(EEGclassifier, 'data/models/HybridNet_clf.pth')
torch.save(model, 'data/models/model_hybridnet_clf.pth')
# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'train_bal_acc', 'valid_acc', 'valid_loss']
df = pd.DataFrame(EEGclassifier.history[:, results_columns], columns=results_columns,
                  index=EEGclassifier.history[:, 'epoch'])

# ------------------------------------------- 1st figure ------------------------------------------------
# plot train_bal_acc
fig, ax1 = plt.subplots(figsize=(14, 6))
df.loc[:, ['train_bal_acc']].plot(
    ax=ax1, style=['-'], marker='o', color='tab:red', legend=False, fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax1.set_ylabel("Balanced Accuracy", color='tab:red', fontsize=14)
ax1.set_xlabel("Epoch", fontsize=14)
# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------- 2nd figure ------------------------------------------------
# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_bal_acc)

fig, ax1 = plt.subplots(figsize=(14, 6))
df.loc[:, ['train_loss']].plot(
    ax=ax1, style=['-'], marker='o', color='tab:blue', legend=False, fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_misclass']].plot(
    ax=ax2, style=['-'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------- 3rd figure ------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 6))
df.loc[:, ['valid_acc']].plot(
    ax=ax1, style=['-'], marker='o', color='tab:green', legend=False, fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:green', labelsize=14)
ax1.set_ylabel("Validation Accuracy", color='tab:green', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['valid_loss']].plot(
    ax=ax2, style=['-'], marker='o', color='tab:orange', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=14)
ax2.set_ylabel("Validation Loss", color='tab:orange', fontsize=14)
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

# Evaluation
print(f"test_data.shape: {test_data.shape}")
print(f"test_labels.shape: {test_labels.shape}")
y_pred = EEGclassifier.predict(test_data)
# generating confusion matrix
confusion_mat = confusion_matrix(test_labels.numpy(), y_pred)
labels = [0.0,1.0]
# plot the basic conf. matrix
plot_confusion_matrix(confusion_mat, class_names=labels)
plt.show()
accuracy = accuracy_score(test_labels.numpy(), y_pred)
balanced_accuracy = balanced_accuracy_score(test_labels.numpy(), y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"balanced Accuracy: {balanced_accuracy * 100:.2f}%")