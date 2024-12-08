import numpy as np
import torch
import pandas as pd
import glob
from skorch.callbacks import EpochScoring, LRScheduler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from braindecode.training.losses import CroppedLoss
from braindecode.models import HybridNet
from braindecode.regressor import EEGRegressor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split

def mean_abs_error(model, X, y):
    y_pred = model.predict(X)
    return mean_absolute_error(y, y_pred)

def mean_sqd_error(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)

def root_mean_squared_error(model, X, y):
    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred))

def coeff_of_determination(model, X, y):
    y_pred = model.predict(X)
    return r2_score(y, y_pred)

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
    labelAll_df = pd.read_csv('data/regr_labelingAll/AllPhases_labeled_Alpha_regression_' + file[file.find("ID"):]+'.csv')
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

# Split in train and test set
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, stratify=y)
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

# Modell init
n_classes = 1  # number of classes (in paper = 2)
n_times = sampling_rate
model = HybridNet(
    n_outputs=n_classes,
    n_chans=n_channels,
    n_times=sampling_rate,
)

# Define the scoring function: balanced accuracy
n_epochs=7
mean_abs_err = EpochScoring(
    scoring=mean_abs_error,
    on_train=True,
    name="mean_abs_err",
    lower_is_better=True,
)
mean_sqd_err = EpochScoring(
    scoring=mean_sqd_error,
    on_train=True,
    name="mean_sqd_err",
    lower_is_better=True,
)
rmean_sqd_err = EpochScoring(
    scoring=root_mean_squared_error,
    on_train=True,
    name="rmean_sqd_err",
    lower_is_better=True,
)
coeff_of_det = EpochScoring(
    scoring=coeff_of_determination,
    on_train=True,
    name="coeff_of_det",
    lower_is_better=True,
)
callbacks = [("mean_abs_err", mean_abs_err), ("mean_sqd_err", mean_sqd_err), ("rmean_sqd_err", rmean_sqd_err), ("coeff_of_det", coeff_of_det), ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))]
# train the model
EEGregressor = EEGRegressor(model, 
                              criterion=torch.nn.MSELoss,
                              #criterion__loss_function=torch.nn.functional.cross_entropy, 
                              optimizer=torch.optim.AdamW,
                              optimizer__lr=0.001,
                              train_split=predefined_split(val_dataset),
                              batch_size=16,
                              callbacks=callbacks)
print(train_data.shape)
print(train_labels.shape)
EEGregressor.fit(train_data, train_labels, epochs=1)
print(EEGregressor.history)
# Extract loss and accuracy values for plotting from history object
results_columns = ['mean_abs_err', 'mean_sqd_err', 'rmean_sqd_err', 'coeff_of_det', 'valid_acc', 'valid_loss']
df = pd.DataFrame(EEGregressor.history[:, results_columns], columns=results_columns,
                  index=EEGregressor.history[:, 'epoch'])

# ------------------------------------------- 1st figure ------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 6))
df.loc[:, ['mean_abs_err']].plot(
    ax=ax1, style=['-'], marker='o', color='tab:blue', legend=False, fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("MAE", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['mean_sqd_err']].plot(
    ax=ax2, style=['-'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("MSE", color='tab:red', fontsize=14)
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

#---------------------------------------------- 2nd figure ------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 6))
df.loc[:, ['rmean_sqd_err']].plot(
    ax=ax1, style=['-'], marker='o', color='tab:orange', legend=False)
ax1.tick_params(axis='y', labelcolor='tab:orange', labelsize=14)
ax1.set_ylabel("RMSE", color='tab:orange', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['coeff_of_det']].plot(
    ax=ax2, style=['-'], marker='o', color='tab:green', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=14)
ax2.set_ylabel("CoefDet", color='tab:green', fontsize=14)
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------- 3rd figure ------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 6))
df.loc[:, ['valid_loss']].plot(
    ax=ax1, style=['-'], marker='o', color='tab:red', legend=False)
ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax1.set_ylabel("Validation Loss", color='tab:red', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['valid_acc']].plot(
    ax=ax2, style=['-'], marker='o', color='tab:green', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=14)
ax2.set_ylabel("Validation Accuracy", color='tab:green', fontsize=14)
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------- Evaluation ------------------------------------------------
print(f"test_data.shape: {test_data.shape}")
print(f"test_labels.shape: {test_labels.shape}")
y_pred = EEGregressor.predict(test_data)
Mean_Sqd_Error = mean_squared_error(test_labels.numpy(), y_pred)
Mean_Abs_Error = mean_absolute_error(test_labels.numpy(), y_pred)
Root_Mean_Sqd_Error = np.sqrt(mean_squared_error(test_labels.numpy(), y_pred))
Coefficients_of_Determination = r2_score(test_labels.numpy(), y_pred)
print(f"Mean Absolute Error: {Mean_Abs_Error :.2f}%")
print(f"Mean Squared Error: {Mean_Sqd_Error :.2f}%")
print(f"Root Mean Squared Error: {Root_Mean_Sqd_Error :.2f}%")
print(f"Coefficients of Determination: {Coefficients_of_Determination :.2f}")