import numpy as np
import torch
import pandas as pd
import glob
from torch.optim import Adam
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from mne import create_info
from mne.io import RawArray
from braindecode.models import EEGNetv4
from braindecode.classifier import EEGClassifier
from braindecode.datasets import WindowsDataset
from braindecode.datautil.windowers import create_windows_from_events
from sklearn.model_selection import train_test_split

n_channels = 14
sampling_rate = 256

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
    # add the participant/file ID as the first column to the dataframe
    raw_df_with_labels.insert(0, 'Participant_ID', file[file.find("ID")+3]+file[file.find("ID")+4])
    dfs_raw_with_labels.append(raw_df_with_labels)

df_all_participants_raw = pd.concat(dfs_raw_with_labels)
# feature vector
X = df_all_participants_raw[['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1','EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']]
y = df_all_participants_raw['label']
channel_names = df_all_participants_raw.columns[1:15].tolist()  # names eeg-channels
print(channel_names)
channel_types = ['eeg'] * n_channels  # declare them as eeg channels
info = create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=channel_types)
# convert X to 14 x len(X) array
XT = X.T
raw = RawArray(XT, info)
'''
channel_names = dfs.columns[:-1].tolist()  # names eeg-channels
channel_types = ['eeg'] * n_channels  # declare them as eeg channels
info = create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=channel_types)
# convert eeg data in MNE-RawArray
eeg_mne_data = eeg_data_reshaped.reshape(n_samples * window_size, n_channels).T
raw = RawArray(eeg_mne_data, info)
'''
# Split in train and test set
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)

train_data = torch.tensor(train_data.values, dtype=torch.float32)
test_data = torch.tensor(test_data.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels.values, dtype=torch.long)
test_labels = torch.tensor(test_labels.values, dtype=torch.long)
# Modell init
n_classes = len(np.unique(y))  # number of classes (in paper = 2)
n_times = sampling_rate
model = EEGNetv4(
    n_classes=n_classes,
    in_chans=n_channels,
    input_window_samples=n_times,
    final_conv_length="auto",
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# train the model
EEGclassifier = EEGClassifier(model, criterion=criterion, lr=0.001, train_split=None)
EEGclassifier.fit(train_data, train_labels, epochs=10)

# Evaluation
y_pred = EEGclassifier.predict(test_data)
accuracy = accuracy_score(test_labels.numpy(), y_pred)
balanced_accuracy = balanced_accuracy_score(test_labels.numpy(), y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"balanced Accuracy: {balanced_accuracy * 100:.2f}%")