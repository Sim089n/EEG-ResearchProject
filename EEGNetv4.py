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
sampling_rate = 128
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
    #raw_df_with_labels.insert(0, 'Participant_ID', file[file.find("ID")+3]+file[file.find("ID")+4])
    dfs_raw_with_labels.append(raw_df_with_labels)

df_all_participants_raw = pd.concat(dfs_raw_with_labels)
df_single_participant_raw = dfs_raw_with_labels[5]
'''
df_single_participant_raw = df_single_participant_raw.drop(columns=['Time'], axis=1)
eeg_data = df_single_participant_raw.values  # Exkludiere die Time-Spalte
n_channels = eeg_data.shape[1]  # Anzahl der Kanäle

# Simulierte Zeitfenstergröße (z. B. 1 Sekunde bei 250 Hz Sampling-Rate)
sampling_rate = 128  # Annahme: 250 Hz
window_size = sampling_rate  # 1 Sekunde
print(f"EEG-Datenform: {eeg_data}")
# Reshape: (Samples, Channels, Time Points)
n_samples = eeg_data.shape[0] // window_size
eeg_data = eeg_data[:n_samples * window_size]  # Kürze auf ein ganzzahliges Fenster
eeg_data_reshaped = eeg_data.reshape((n_samples, window_size, n_channels)).transpose(0, 2, 1)
labels = eeg_data_reshaped['label']
eeg_data_reshaped = eeg_data_reshaped.drop(columns=['label'], axis=1)
# Labels (Dummy-Labels für den Vertrauenslevel)
# Erstellen Sie Labels basierend auf Ihrer Logik oder Ihrem Datensatz
print(f"EEG-Datenform: {eeg_data_reshaped.shape}, Labels: {labels.shape}")
'''
# feature vector
X = df_single_participant_raw[['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1','EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']].values
y = df_single_participant_raw['label'].values
channel_names = df_single_participant_raw.columns[1:15].tolist()  # names eeg-channels
channel_types = ['eeg'] * n_channels  # declare them as eeg channels
info = create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=channel_types)
# convert X to 14 x len(X) array
XT = X.T
raw = RawArray(XT, info)

# Split in train and test set
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)
#n_samples, n_channels = X.shape


n_channels = train_data.shape[1]

block_size = n_channels * sampling_rate
rows_train_data = train_data.shape[0]
rows_test_data = test_data.shape[0]
# Trimmen der Daten
train_data = train_data[:(rows_train_data // block_size) * block_size]
test_data = test_data[:(rows_test_data // block_size) * block_size]

# Anzahl der Samples berechnen
n_samples_train = train_data.size // block_size
n_samples_test = test_data.size // block_size

# Reshape: (Samples, Channels, Time Points)
train_data = train_data.reshape(-1, n_channels, sampling_rate)
test_data = test_data.reshape(-1, n_channels, sampling_rate)

# Labels anpassen
train_labels = train_labels[:(rows_train_data // block_size) * block_size]
train_labels = train_labels[::train_data.shape[2]]
test_labels = test_labels[:(rows_test_data // block_size) * block_size]
test_labels = test_labels[::test_data.shape[2]]

# Convert to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(f"Train data shape: {train_data.shape}")
# Modell init
n_classes = 2  # number of classes (in paper = 2)
n_times = sampling_rate
model = EEGNetv4(
    n_outputs=n_classes,
    n_chans=n_channels,
    n_times=sampling_rate,
    final_conv_length="auto",
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# train the model
EEGclassifier = EEGClassifier(model, criterion=criterion, lr=0.001, train_split=None)
print(train_data.shape)
print(train_labels.shape)
EEGclassifier.fit(train_data, train_labels, epochs=10)

# Evaluation
y_pred = EEGclassifier.predict(test_data)
accuracy = accuracy_score(test_labels.numpy(), y_pred)
balanced_accuracy = balanced_accuracy_score(test_labels.numpy(), y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"balanced Accuracy: {balanced_accuracy * 100:.2f}%")