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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
def main(create_models: bool = typer.Option(
           help="Set to true if you want to create the SVM and kmeans Clustering models"),
        preprocessing_needed: bool = typer.Option(
            help="True or false depending on if preprocessing was already done or not. If labeled csv \
            files are already existing it is not needed"),
        path_labeled_csvfiles: str = typer.Option(
            help= "Give the path to the labeled csv files you want to use for the machine learning models"),
        flag_all_phases: bool = typer.Option(
            help="True or false depending on if we want to include all phases in the classification"),
        label_regression: bool = typer.Option(
            help="True or false depending on if we want to create labeled csv files for training a regression model")
        ):
    if preprocessing_needed == True:
        # Load the provided EEG dataset
        data_path = 'data/raw/'
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
            if not os.path.exists(f'data/preprocessed/processed_eeg_data_{file_path.replace(data_path, '')}.csv'):
                output_file = (f'data/preprocessed/processed_eeg_data_{file_path.replace(data_path, '')}.csv')
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
            if not os.path.exists(f'data/features/features_Alpha_{file_path.replace(data_path, '')}.csv'):
                output_file_alpha_features = (f'data/features/features_Alpha_{file_path.replace(data_path, '')}.csv')
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
                            if current_phase == 3: # if we are in phase 3
                                if details.iloc[index]['trust_score_binary'] == "high": # we check if trust is labeled high
                                    if label_regression == True: # set the scorea as a label for training a regression model
                                        alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                    else:
                                        alpha_features_df.at[index_alpha, 'label'] = 1
                                elif details.iloc[index]['trust_score_binary'] == "low": # if trust is labeld low
                                    if label_regression == True: # set the scorea as a label for training a regression model
                                        alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                    else: # otherwise set it to the corresponding binary label
                                        alpha_features_df.at[index_alpha, 'label'] = 0
                                else : # this is the case where the trust_score is not labeled correctly so we set a own threshold
                                    if details.iloc[index]['trust_score'] >= 4.8:
                                        if label_regression == True: # set the scorea as a label for training a regression model
                                            alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                        else: # otherwise set it to the corresponding binary label
                                            alpha_features_df.at[index_alpha, 'label'] = 1
                                    else: # if the trust score is below 4.8 we need to set it to 0 as a binary label
                                        if label_regression == True: # set the scorea as a label for training a regression model
                                            alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                        else: # otherwise set it to the corresponding binary label
                                            alpha_features_df.at[index_alpha, 'label'] = 0
                            elif current_phase == 4:
                                if details.iloc[index]['trust_score_binary'] == "high":
                                    if label_regression == True: # set the scorea as a label for training a regression model
                                        alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                    else: # otherwise set it to the corresponding binary label
                                        alpha_features_df.at[index_alpha, 'label'] = 1
                                elif details.iloc[index]['trust_score_binary'] == "low":
                                    if label_regression == True: # set the scorea as a label for training a regression model
                                        alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                    else: # otherwise set it to the corresponding binary label
                                        alpha_features_df.at[index_alpha, 'label'] = 0
                                else : # this is the case where the trust_score is not labeled correctly so we set a own threshold
                                    if details.iloc[index]['trust_score'] >= 4.8:
                                        if label_regression == True: # set the scorea as a label for training a regression model
                                            alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                        else: # otherwise set it to the corresponding binary label
                                            alpha_features_df.at[index_alpha, 'label'] = 1
                                    else:
                                        alpha_features_df.at[index_alpha, 'label'] = 0
                            if flag_all_phases == True:
                                if current_phase == 1 or current_phase == 2 or current_phase == 5:
                                    if details.iloc[index]['trust_score_binary'] == "high":
                                        if label_regression == True: # set the scorea as a label for training a regression model
                                            alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                        else: # otherwise set it to the corresponding binary label
                                            alpha_features_df.at[index_alpha, 'label'] = 1
                                    elif details.iloc[index]['trust_score_binary'] == "low":
                                        if label_regression == True: # set the scorea as a label for training a regression model
                                            alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                        else: # otherwise set it to the corresponding binary label
                                            alpha_features_df.at[index_alpha, 'label'] = 0
                                    else :
                                        if details.iloc[index]['trust_score'] >= 4.8:
                                            if label_regression == True: # set the scorea as a label for training a regression model
                                                alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                            else: # otherwise set it to the corresponding binary label
                                                alpha_features_df.at[index_alpha, 'label'] = 1
                                        else:
                                            if label_regression == True: # set the scorea as a label for training a regression model
                                                alpha_features_df.at[index_alpha, 'label'] = details.iloc[index]['trust_score']
                                            else: # otherwise set it to the corresponding binary label
                                                alpha_features_df.at[index_alpha, 'label'] = 0

                    sum_time += 60
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
            
            # Save the processed EEG data to a new CSV file
            if flag_all_phases == True:
                if label_regression == False:
                    if not os.path.exists(f'data/labelingAll/AllPhases_labeled_Alpha_{file_path.replace(data_path, '')}.csv'):
                        output_file_alpha_features = (f'data/labelingAll/AllPhases_labeled_Alpha_{file_path.replace(data_path, '')}.csv')
                        alpha_features_df.to_csv(output_file_alpha_features)
                elif label_regression == True:
                    if not os.path.exists(f'data/regr_labelingAll/AllPhases_labeled_Alpha_regression_{file_path.replace(data_path, '')}.csv'):
                        output_file_alpha_features = (f'data/regr_labelingAll/AllPhases_labeled_Alpha_regression_{file_path.replace(data_path, '')}.csv')
                        alpha_features_df.to_csv(output_file_alpha_features)
            elif flag_all_phases == False:
                if label_regression == False:
                    if not os.path.exists(f'data/labeling3and4/labeled_Alpha_{file_path.replace(data_path, '')}.csv'):
                        output_file_alpha_features = (f'data/labeling3and4/labeled_Alpha_{file_path.replace(data_path, '')}.csv')
                        alpha_features_df.to_csv(output_file_alpha_features)
                elif label_regression == True:
                    if not os.path.exists(f'data/regr_labeling3and4/labeled_Alpha_regression_{file_path.replace(data_path, '')}.csv'):
                        output_file_alpha_features = (f'data/regr_labeling3and4/labeled_Alpha_regression_{file_path.replace(data_path, '')}.csv')
                        alpha_features_df.to_csv(output_file_alpha_features)

    #-----------------End of Preprocessing -----------------#
    

    #----------------- Machine Learning Models -----------------#

    # create a support vector machine which is trained on the features alpha files
    # the labels are the trust_score_binary from the details.csv file
    # the model should be saved as a pickle file and can be used in the main file

    if create_models == True:            
        
        # Load the labeled Alpha band features
        df_frames = []
        for file in glob.glob(os.path.join(path_labeled_csvfiles,'*.csv')):
            # Load the first file to inspect its structure
            alpha_df = pd.read_csv(file, index_col=0)
            # only consider the rows with a label
            alpha_df_with_label = alpha_df.dropna()
            # add the participant/file ID as the first column to the dataframe
            alpha_df_with_label.insert(0, 'Participant_ID', file[file.find("ID")+3]+file[file.find("ID")+4])
            df_frames.append(alpha_df_with_label)

        # Concatenate the DataFrames to get our dataset --> includes all participants in stages 3 and 4
        alpha_df = pd.concat(df_frames)
        # value_counts pandas function -> count
        print(alpha_df['label'].value_counts())
        # make histogram of value counts per label
        alpha_df['label'].value_counts().plot(kind='bar')
        os.makedirs(os.path.dirname('plots/'), exist_ok=True)
        plt.savefig('plots/value_counts_trust_classes.png')
        plt.show(block=False)
        # Split dataset into training set and test set
        X = alpha_df[['Mean', 'Peak', 'Std', 'Kurtosis']]  # Features
        y = alpha_df['label']  # Labels
        df_low_high_trust_counts_per_participant=alpha_df.groupby(['Participant_ID']).agg('label').value_counts()
        print(df_low_high_trust_counts_per_participant)
        if not os.path.exists(f'low_high_trust_counts_per_participant.csv'):
                    output_file_low_high_trust_counts = (f'low_high_trust_counts_per_participant.csv')
                    df_low_high_trust_counts_per_participant.to_csv(output_file_low_high_trust_counts)
        
        # plot the value counts of the trust classes per participant
        alpha_df.groupby('Participant_ID')['label'].value_counts().unstack().plot(kind='bar', stacked=True)
        # Augmentation approach to balance the amount of high and low trust instances
        # include SMOTE to balance the dataset
        # Apply SMOTE to create synthetic samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # 70% training and 30% test
        # add minmax scaler
        minmax_scaler = MinMaxScaler()
        X_train_scaled = minmax_scaler.fit_transform(X_train)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)

        X_test_scaled = minmax_scaler.transform(X_test)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_smote_train, y_smote_train = smote.fit_resample(X_train_df, y_train)
    
        #Create a svm Classifier
        svm_clf = svm.SVC(C=0.5, class_weight="balanced") # Linear Kernel
        # Define Stratified K-Folds
        skf = StratifiedKFold(n_splits=5)  # 5-fold cross-validation

        # Initialize lists to collect scores for each fold
        accuracy_scores = []
        precision_scores = []
        recall_scores = []

        # Loop through each fold
        for train_index, test_index in skf.split(X_smote_train, y_smote_train):
            X_train_fold, X_test_fold = X_smote_train.iloc[train_index], X_smote_train.iloc[test_index]
            y_train_fold, y_test_fold = y_smote_train.iloc[train_index], y_smote_train.iloc[test_index]
            
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
        y_pred = svm_clf.predict(X_test_df)

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
        scaler_kmeans = StandardScaler()
        X_train_scaled_features = scaler_kmeans.fit_transform(X_smote_train)
        X_test_scaled = scaler_kmeans.transform(X_test)
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
            kmeans.fit(X_train_scaled_features)
            sse.append(kmeans.inertia_)

        # print the ellbow plot
        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        # svae plot
        # Ensure the directory exists
        os.makedirs(os.path.dirname('plots/'), exist_ok=True)
        plt.savefig('plots/elbow_plot.png')
        plt.show(block=False)

        # Fit kmeans with 2 clusters for this example
        kmeans = KMeans(init="random", n_clusters=2, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(X_train_scaled_features)

        y_prediction_train = kmeans.predict(X_train_scaled_features)
        y_prediction_test = kmeans.predict(X_test_scaled)
        print(f"Crosstab 2 clusters: {pd.crosstab(y_test, y_prediction_test)}")
        # Generate the contingency matrix
        contingency_matrix = metrics.cluster.contingency_matrix(y_test, y_prediction_test)

        # Find the best label alignment using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        aligned_labels = np.zeros_like(y_prediction_test)
        for i, j in zip(row_ind, col_ind):
            aligned_labels[y_prediction_test == j] = i

        
        # check what the participants wrote in their questionnaire is represented in the EEG data as well
        # find the instances where it is not matching -> keep those instances and go back to the features and try to gain information based on the raw and processed data
        # Calculate metrics
        accuracy = metrics.balanced_accuracy_score(y_test, aligned_labels)
        precision = metrics.precision_score(y_test, aligned_labels, average='weighted')
        recall = metrics.recall_score(y_test, aligned_labels, average='weighted')
        f1 = metrics.f1_score(y_test, aligned_labels, average='weighted')
        conf_mat = metrics.confusion_matrix(y_test, aligned_labels)
        print("---------------------------------------------------------------------")
        print("KMeans Clustering Results:")
        print("Silhouette Score:", silhouette_score(X_test_scaled, aligned_labels))
        print(f"Accuracy k-means: {accuracy:.4f}")
        print(f"Precision k-means: {precision:.4f}")
        print(f"Recall k-means: {recall:.4f}")
        print(f"Confusion Matrix k-means:\n{conf_mat}")
        print(f"F1 Score k-means: {f1:.4f}")
        print(f"Inertia k-means: {kmeans.inertia_:.4f}")
        print(f"Cluster Centers:", kmeans.cluster_centers_)
        print(f"Number of Iterations: {kmeans.n_iter_:.4f}")
        print(f"Labels:", kmeans.labels_)

        ''''''
        # create dataframe which holds the wrongly classified instances
        wrong_classified_instances = pd.DataFrame(columns=alpha_df.columns)
        y_smote_original = y_test[0:len(y)]
        for i in range(len(y_smote_original)):
            if y_smote_train[i] != aligned_labels[i]:
                wrong_classified_instances.loc[len(wrong_classified_instances)] = alpha_df.iloc[i] # oob error because of oversampling
                
        print(f"Number of wrongly classified instances: {len(wrong_classified_instances)} from {len(y_smote_original)} instances in the orginal dataset were labeled wrong.")
        print(f"That means that {len(wrong_classified_instances)/len(y_smote_original)*100}% of the instances were labeled wrong.")
        print(f"{len(y_test)-len(y_smote_original)} instances were added by the SMOTE algorithm from which {conf_mat[0,0]+conf_mat[1,1]-(len(y_smote_original)-len(wrong_classified_instances))} were labeled right and {conf_mat[0,1]+conf_mat[1,0]-len(wrong_classified_instances)} were labeled wrong.")
        
        # Plotting the clusters
        plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=aligned_labels, cmap='viridis', marker='o')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
        plt.legend()
        # Ensure the directory exists
        os.makedirs(os.path.dirname('plots/'), exist_ok=True)
        plt.savefig('plots/kmeans_clusters.png') 
        plt.show(block=False)

        #-----------------do the k-means with 4 clusters-----------------#
        kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(X_train_scaled_features)

        y_prediction_train = kmeans.predict(X_train_scaled_features)
        y_prediction_test = kmeans.predict(X_test_scaled)
        # cross tabulation
        print(f"Crosstab 4 clusters: {pd.crosstab(y_test, y_prediction_test)}")
    


if __name__ == "__main__":
    typer.run(main)
# __main__ only set when NNone is main script
# if script is imported, __main__ is not set, so typer.run(main) is not executed
# instead the function main() can be called from another script
# or we check if __name__ equals name of script -> if so, execute typer.run(main)
# because interpreter sets __name__ to __main__ when script is and to "name of script" when imported