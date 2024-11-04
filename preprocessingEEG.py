import numpy as np
import pandas as pd
import glob
import os
import typer
import pathlib as Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.signal.windows import hann
from scipy.signal import welch
from scipy.stats import kurtosis
from scipy.fftpack import fft
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#------------- Define functions to preprocess EEG data --------------# 

# Define a function to calculate the Alpha band contribution
def calculate_alpha_contribution(band_power_df, band_name='Alpha'):
    # Calculate the total power for each sensor by summing the power of all bands
    total_power = band_power_df.sum(axis=1)
    
    # Calculate the contribution of the Alpha band for each sensor
    alpha_contribution = band_power_df[band_name] / total_power
    
    # Average the Alpha band contribution across all 14 channels
    average_alpha_contribution = alpha_contribution.mean()
    
    return average_alpha_contribution

# Apply FFT to each channel and compute power spectral density
def compute_powerspectraldensity(data, fs, window):
    n = len(data)
    freq = np.fft.fftfreq(n, 1/fs)[:n//2]
    fft_values = fft(data * window)[:n//2]
    psd_values = np.abs(fft_values)**2 / n
    return freq, psd_values

# Compute the power in each band for each channel
def bandpower(psd, freqs, band):
    band_freqs = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[band_freqs])
# Define a high-pass filter with 0.16 Hz cutoff
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Apply high-pass filter to the EEG data
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

# Define frequency bands of interest
bands = {
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Low Beta': (12, 16),
    'High Beta': (16, 25),
    'Gamma': (25, 45)
}
#----------------- Windowing and Power Spectral Density -----------------#

# Function to compute power spectral density (PSD) for a specific window and channel
def compute_alpha_psd_window(data, fs, band):
    # Welch's method to compute PSD
    freqs, psd = welch(data, fs=fs, nperseg=fs)
    # Find indices of the band
    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    # Return the average power in the alpha band
    return np.sum(psd[band_idx])

# Extract features for each window
def extract_window_features(data, fs, band, n_channels):
    features = []
    # Iterate through windows
    for start in range(0, len(data), fs):
        end = start + fs
        if end > len(data):
            break  # Ignore the last window if it's incomplete
        # Slice the window
        window = data[start:end]
        # Compute Alpha band power
        alpha_power = [compute_alpha_psd_window(window[channel], fs, band) for channel in range(n_channels)]
        # Compute statistical features on Alpha power across the 14 channels
        alpha_mean = np.mean(alpha_power)
        alpha_peak = np.max(alpha_power)
        alpha_median = np.median(alpha_power)
        alpha_std = np.std(alpha_power)
        alpha_kurtosis = kurtosis(alpha_power)
        # Store the features for this window
        features.append([alpha_mean, alpha_peak, alpha_median, alpha_std, alpha_kurtosis])
    return np.array(features)

#----------------- Load and preprocess the EEG data -----------------#
def main(#num_trustlevels: int = typer.Option(
         #  help="number of trustlevels which we want to classify"),
        preprocessing_needed: bool = typer.Option(
            help="True or false depending on if preprocessing was already done or not. If labeled csv \
            files are already existing it is not needed"),
        #path_labeled_csvfiles: str = typer.Option(
        #    help= "Give the path to the labeled csv files if they are already existing"),
        flag_all_phases: bool = typer.Option(
            help="True or false depending on if we want to include all phases in the classification")
        ):
    if preprocessing_needed == True:
        # Load the provided EEG dataset
        data_path = 'data/'
        files = []
        for file in glob.glob(os.path.join(data_path, '*.csv') ):
            files.append(file)

        # Assuming the EEG sampling frequency is commonly 128 Hz (we can adjust if needed)
        fs = 128
        cutoff_freq = 0.16

        for file_path in files:
            # ID of file we load
            participant_ID = str(file_path)[str(file_path).find("ID")+3]
            participant_ID = int(participant_ID)
            eeg_data = pd.read_csv(file_path)

            # Display the first few rows of the data to inspect its structure
            eeg_data.head(), eeg_data.info()

            # Apply the filter to all EEG channels
            eeg_channels = eeg_data.columns[:-1]  # Exclude the Time column
            filtered_data = eeg_data[eeg_channels].apply(lambda x: highpass_filter(x, cutoff_freq, fs))

            # Display filtered data (first few rows)
            filtered_data.head()

            # Apply ICA to remove artifacts from the filtered EEG data
            ica = FastICA(n_components=len(eeg_channels))

            # Fit ICA to the filtered data and transform it
            ica_data = ica.fit_transform(filtered_data)

            # Convert the result back to a DataFrame
            ica_df = pd.DataFrame(ica_data, columns=eeg_channels)
            n_channels = len(ica_df.columns)
            entries = len(ica_df.index)
            # Display the first few rows of the ICA-processed data
            ica_df.head()

            # Apply Hanning window function
            window = hann(len(ica_df))

            # Sampling frequency
            n_samples = len(ica_df)
            frequencies = {}
            for channel in ica_df.columns:
                freqs, psd_values = compute_powerspectraldensity(np.array(ica_df[channel]), fs, window)
                power_in_bands = {band: bandpower(psd_values, freqs, band_range) for band, band_range in bands.items()}
                frequencies[channel] = power_in_bands

            # Convert the power in bands to a DataFrame for easier viewing
            # the power is given in micro Volts squared for each of the frequency bands
            band_power_df = pd.DataFrame(frequencies).T

            # Display the power values in each frequency band for the first few channels
            band_power_df.head()
            # Save the processed EEG data to a new CSV file
            if not os.path.exists(f'preprocessed/processed_eeg_data_{file_path.replace(data_path, '')}.csv'):
                output_file = (f'processed_eeg_data_{file_path.replace(data_path, '')}.csv')
                band_power_df.to_csv(output_file)

            #------------------- Feature Extraction Part of the Loop -------------------#

            # Call the function to calculate the Alpha band contribution and average
            alpha_contribution_avg = calculate_alpha_contribution(band_power_df)

            # Output the result
            print(f"Average Alpha Band Contribution: {alpha_contribution_avg:.4f}")
            alpha_features = extract_window_features(ica_df.values, fs, bands['Alpha'], n_channels)

            # Convert the features to a DataFrame for better readability
            feature_columns = ['Mean', 'Peak', 'Median', 'Std', 'Kurtosis']
            alpha_features_df = pd.DataFrame(alpha_features, columns=feature_columns)

            # Display the first few rows of the extracted features
            alpha_features_df.head()
            if not os.path.exists(f'features_Alpha_{file_path.replace(data_path, '')}.csv'):
                output_file_alpha_features = (f'features_Alpha_{file_path.replace(data_path, '')}.csv')
                alpha_features_df.to_csv(output_file_alpha_features)

            # do the labeling of each row depending on the details.csv file
            details = pd.read_csv('data/details/details.csv', sep = ';')
            print(details.head())
            details_participant_ID = details[details['id'] == participant_ID] # df with details of the participant
            if details_participant_ID.empty:
                print(f"No data found for participant ID: {participant_ID}")
            else:
                print(f"Filtered data for participant {participant_ID}:")
            print(details_participant_ID)
            details_participant_ID.info()
            sum_time = 0
            for index, row in details_participant_ID.iterrows():
                try:
                    # keep track of the time
                    start_time = datetime.strptime(details.iloc[index]['start_clock'], '%H:%M')
                    end_time = datetime.strptime(details.iloc[index]['finish_clock'], '%H:%M')
                    time_diff_phase_in_seconds = (end_time - start_time).total_seconds()
                    sum_time += time_diff_phase_in_seconds
                    
                    current_phase = row['phase']
                    # label the feature rows in alpha_features_df with the labels of phase 3 and 4 from details.csv
                    # if the time of the row is between the start and end time of phase 3, label it as phase 3
                    # if the time of the row is between the start and end time of phase 4, label it as phase 4
                    # if the time of the row is not between the start and end time of phase 3 and 4, label it not at all
                    for index_alpha, row_alpha in alpha_features_df.iterrows():
                        timestamp_alpha = alpha_features_df.index[index_alpha]
                        if sum_time - time_diff_phase_in_seconds <= timestamp_alpha <= sum_time:
                            if current_phase == 3:
                                if details.iloc[index]['trust_score_binary'] == "high":
                                    alpha_features_df.at[index_alpha, 'label'] = 1
                                elif details.iloc[index]['trust_score_binary'] == "low":
                                    alpha_features_df.at[index_alpha, 'label'] = 0
                                else :
                                    if details.iloc[index]['trust_score'] >= 4.8:
                                        alpha_features_df.at[index_alpha, 'label'] = 1
                                    else:
                                        alpha_features_df.at[index_alpha, 'label'] = 0
                            elif current_phase == 4:
                                if details.iloc[index]['trust_score_binary'] == "high":
                                    alpha_features_df.at[index_alpha, 'label'] = 1
                                elif details.iloc[index]['trust_score_binary'] == "low":
                                    alpha_features_df.at[index_alpha, 'label'] = 0
                                else :
                                    if details.iloc[index]['trust_score'] >= 4.8:
                                        alpha_features_df.at[index_alpha, 'label'] = 1
                                    else:
                                        alpha_features_df.at[index_alpha, 'label'] = 0
                            if flag_all_phases == True:
                                if current_phase == 1 or current_phase == 2 or current_phase == 5:
                                    if details.iloc[index]['trust_score_binary'] == "high":
                                        alpha_features_df.at[index_alpha, 'label'] = 1
                                    elif details.iloc[index]['trust_score_binary'] == "low":
                                        alpha_features_df.at[index_alpha, 'label'] = 0
                                    else :
                                        if details.iloc[index]['trust_score'] >= 4.8:
                                            alpha_features_df.at[index_alpha, 'label'] = 1
                                        else:
                                            alpha_features_df.at[index_alpha, 'label'] = 0

                    sum_time += 60
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
            
            # Save the processed EEG data to a new CSV file
            if not os.path.exists(f'labeled_Alpha_{file_path.replace(data_path, '')}.csv'):
                output_file_alpha_features = (f'labeled_Alpha_{file_path.replace(data_path, '')}.csv')
                alpha_features_df.to_csv(output_file_alpha_features)

    #-----------------End of Preprocessing -----------------#
                
    #----------------- Machine Learning Models -----------------#

    # create a support vector machine which is trained on the features alpha files
    # the labels are the trust_score_binary from the details.csv file
    # the model should be saved as a pickle file and can be used in the main file

    # Load the labeled Alpha band features
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% test

    # include SMOTE to balance the dataset
    # Apply SMOTE to create synthetic samples
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)



    #Create a svm Classifier
    svm_clf = svm.SVC(C=0.5, class_weight="balanced") # Linear Kernel
    # Define Stratified K-Folds
    skf = StratifiedKFold(n_splits=5)  # 5-fold cross-validation

    # Initialize lists to collect scores for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    # Loop through each fold
    for train_index, test_index in skf.split(X_smote, y_smote):
        X_train_fold, X_test_fold = X_smote.iloc[train_index], X_smote.iloc[test_index]
        y_train_fold, y_test_fold = y_smote.iloc[train_index], y_smote.iloc[test_index]
        
        # Train on fold
        svm_clf.fit(X_train_fold, y_train_fold)
        
        # Predict on test fold
        y_pred_fold = svm_clf.predict(X_test_fold)
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
        precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='binary'))  # Adjust for binary/multiclass
        recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='binary'))

    # Calculate average scores
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    print("---------------------------------------------------------------------")
    print("SVM Training:")
    print(f"Average Training Accuracy: {avg_accuracy}")
    print(f"Average Training Precision: {avg_precision}")
    print(f"Average Training Recall: {avg_recall}")
    
    #Train the model using the training sets
    #svm_clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = svm_clf.predict(X_test)

    print("---------------------------------------------------------------------")
    print("SVM Classification Results:")
    # Model Accuracy: how often is the classifier correct
    print("Accuracy SVM:",metrics.balanced_accuracy_score(y_test, y_pred)) 
    # Model Precision: what percentage of positive tuples are labeled as such
    print("Precision SVM:",metrics.precision_score(y_test, y_pred, average='weighted'))
    # Model Recall: what percentage of positive tuples are labelled as such
    print("Recall SVM:",metrics.recall_score(y_test, y_pred, average='weighted'))
    # Model F1 Score: weighted average of the precision and recall
    print("F1 Score SVM:",metrics.f1_score(y_test, y_pred, average='weighted'))
    # Confusion Matrix
    print("Confusion Matrix SVM:\n",metrics.confusion_matrix(y_test, y_pred))

    #--------------------------------------------- k-means clustering ---------------------------------------------#

    # create a k-means clustering algorithm which is trained on the features alpha files
    # the labels are the trust_score_binary from the details.csv file
    # the model should be saved as a pickle file and can be used in the main file
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_smote)
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    #graph = sns.PairGrid(alpha_df, hue="label")
    #graph.map(sns.scatterplot)
    #graph.add_legend()

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    # print the ellbow plot
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    # svae plot
    # Ensure the directory exists
    os.makedirs(os.path.dirname('plots/'), exist_ok=True)
    plt.savefig('plots/elbow_plot.png')

    # Fit kmeans with 2 clusters for this example
    kmeans = KMeans(init="random", n_clusters=2, n_init=10, max_iter=300, random_state=42)
    y_prediction = kmeans.fit_predict(scaled_features)

    # Generate the contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_smote, y_prediction)

    # Find the best label alignment using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    aligned_labels = np.zeros_like(y_prediction)
    for i, j in zip(row_ind, col_ind):
        aligned_labels[y_prediction == j] = i

    # Augmentation approach to balance the amount of high and low trust instances
    # check what the participants wrote in their questionnaire is represented in the EEG data as well
    # find the instances where it is not matching -> keep those instances and go back to the features and try to gain information based on the raw and processed data
    # Calculate metrics
    accuracy = metrics.balanced_accuracy_score(y_smote, aligned_labels)
    precision = metrics.precision_score(y_smote, aligned_labels, average='weighted')
    recall = metrics.recall_score(y_smote, aligned_labels, average='weighted')
    f1 = metrics.f1_score(y_smote, aligned_labels, average='weighted')
    conf_mat = metrics.confusion_matrix(y_smote, aligned_labels)
    print("---------------------------------------------------------------------")
    print("KMeans Clustering Results:")
    print("Silhouette Score:", silhouette_score(scaled_features, aligned_labels))
    print(f"Accuracy k-means: {accuracy:.4f}")
    print(f"Precision k-means: {precision:.4f}")
    print(f"Recall k-means: {recall:.4f}")
    print(f"Confusion Matrix k-means:\n{conf_mat}")
    print(f"F1 Score k-means: {f1:.4f}")
    print(f"Inertia k-means: {kmeans.inertia_:.4f}")
    print(f"Cluster Centers:", kmeans.cluster_centers_)
    print(f"Number of Iterations: {kmeans.n_iter_:.4f}")
    print(f"Labels:", kmeans.labels_)

    # Plotting the clusters
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=aligned_labels, cmap='viridis', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
    plt.legend()
    plt.show()
    # Ensure the directory exists
    os.makedirs(os.path.dirname('plots/'), exist_ok=True)
    plt.savefig('plots/kmeans_clusters.png')


if __name__ == "__main__":
    typer.run(main)
# __main__ only set when NNone is main script
# if NNone is imported, __main__ is not set, so typer.run(main) is not executed
# instead the function main() can be called from another script
# or we check if __name__ equals name of script -> if so, execute typer.run(main)
# because interpreter sets __name__ to __main__ when script is and to "name of script" when imported