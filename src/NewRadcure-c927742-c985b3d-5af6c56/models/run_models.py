import os

import numpy as np
import pandas as pd
from collections import Counter

from base_logistic import SimpleBaseline

np.random.seed(42)


def split_zeros(group, training_ratio, test_ratio):
    zeros = group[group['RADCURE-challenge'] == '0']
    n_zeros = zeros.shape[0]
    n_training = int(n_zeros * training_ratio)
    n_test = n_zeros - n_training

    # Shuffle the indices to randomly assign 'training' and 'test'
    shuffled_indices = np.random.permutation(zeros.index)
    training_indices = shuffled_indices[:n_training]
    test_indices = shuffled_indices[n_training:]

    # Assign 'training' and 'test'
    group.loc[training_indices, 'RADCURE-challenge'] = 'training'
    group.loc[test_indices, 'RADCURE-challenge'] = 'test'

    return group

def check_file_exists(file_path):
    return os.path.isfile(file_path)


if __name__ == '__main__':
    negative_controls = ["default", "randomized_full", "randomized_roi", "randomized_non_roi", "shuffled_full",
                         "shuffled_roi",
                         "shuffled_non_roi", "randomized_sampled_full", "randomized_sampled_roi",
                         "randomized_sampled_non_roi"]

    for nc in negative_controls:

        if nc == "default":
            df_radiomic_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/radiomicfeatures_RADCURE.csv'
        else:
            df_radiomic_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/radiomicfeatures_' + nc + '_RADCURE.csv'

        if not check_file_exists(df_radiomic_path):
            continue
        else:
            df_radiomic = pd.read_csv(df_radiomic_path)


        df_outcomes = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/RADCURE-DA-CLINICAL-2.csv')
        output_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output/Radcure/' + nc
        radcure_radiomics = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/radiomicfeatures_RADCURE.csv')


        if not os.path.exists('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output'):
            os.makedirs('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        df_radiomic.rename(columns={'patient_ID': 'ID'}, inplace=True)
        df_outcomes.rename(columns={'patient_id': 'ID'}, inplace=True)
        radcure_radiomics.rename(columns={'patient_ID': 'ID'}, inplace=True)

        # Drop the compactness radiomic feature
        df_radiomic = df_radiomic.drop(columns=["original_shape_Compactness1"])

        # Convert the columns to datetime
        df_outcomes['RT Start'] = pd.to_datetime(df_outcomes['RT Start'])
        df_outcomes['Last FU'] = pd.to_datetime(df_outcomes['Last FU'])

        # Calculate the time difference in months
        df_outcomes['Difference'] = (df_outcomes['Last FU'] - df_outcomes['RT Start']).dt.days / 30.44

        # Create the target_binary column
        df_outcomes['target_binary'] = df_outcomes['Difference'].apply(lambda x: 1 if x > 24 else 0)

        # Rename the length FU column to survival time
        df_outcomes.rename(columns={'Length FU': 'survival_time'}, inplace=True)

        # Rename the 'status' column to 'death' and map the values
        df_outcomes.rename(columns={'Status': 'death'}, inplace=True)
        df_outcomes['death'] = df_outcomes['death'].map({'Alive': 0, 'Dead': 1})

        # Rename the T and N columns to include the stage clarifier
        df_outcomes.rename(columns={'T': 'T Stage'}, inplace=True)
        df_outcomes.rename(columns={'N': 'N Stage'}, inplace=True)

        # Rename the "chemo?" column to be chemotherapy and binarize
        df_outcomes.rename(columns={'Chemo? ': 'Chemotherapy'}, inplace=True)
        df_outcomes['Chemotherapy'] = df_outcomes['Chemotherapy'].map({'none': 0, 'Yes': 1})

        # Rename the "HPV" column to be "HPV Combined" and binarize
        df_outcomes.rename(columns={'HPV': 'HPV Combined'}, inplace=True)
        df_outcomes['HPV Combined'] = df_outcomes['HPV Combined'].map({'Yes, Negative': 0, 'Yes, positive': 1})

        # Rename the "ds" column to be "Disease Site" and binarize
        df_outcomes.rename(columns={'Ds Site': 'Disease Site'}, inplace=True)

        # Remove rows where 'Disease Site' is 'Unknown' or 'Other'
        df_outcomes = df_outcomes[~df_outcomes['Disease Site'].isin(['Unknown', 'Other'])]

        # Convert 'Disease Site' values to lower case
        df_outcomes['Disease Site'] = df_outcomes['Disease Site'].str.lower()

        # Rename the 'ECOG PS' column to 'ECOG'
        df_outcomes.rename(columns={'ECOG PS': 'ECOG'}, inplace=True)

        # Remove rows with empty 'ECOG' values
        df_outcomes = df_outcomes[df_outcomes['ECOG'].notna()]

        # Replace 'Unknown' with blank
        df_outcomes['ECOG'] = df_outcomes['ECOG'].replace('Unknown', '')

        # Replace 'ECOG 0-1' with '0'
        df_outcomes['ECOG'] = df_outcomes['ECOG'].replace('ECOG 0-1', '0')

        # Remove rows with 'ECOG-Pt'
        df_outcomes = df_outcomes[~df_outcomes['ECOG'].str.contains('ECOG-Pt')]

        # Remove the 'ECOG ' prefix to leave only the number
        df_outcomes['ECOG'] = df_outcomes['ECOG'].str.replace('ECOG ', '')

        columns_to_keep = [
            'ID', 'target_binary', 'survival_time', 'death', 'Age',
            'Sex', 'T Stage', 'N Stage', 'Stage', 'Dose', 'Chemotherapy',
            'HPV Combined', 'Disease Site', 'ECOG', 'RADCURE-challenge'
        ]

        # Keep only the specified columns
        df_outcomes = df_outcomes[columns_to_keep]

        # Calculate the ratio of 'training' to 'test'
        training_count = df_outcomes[df_outcomes['RADCURE-challenge'] == 'training'].shape[0]
        test_count = df_outcomes[df_outcomes['RADCURE-challenge'] == 'test'].shape[0]
        total_count = training_count + test_count

        # Calculate the ratio
        training_ratio = training_count / total_count
        test_ratio = test_count / total_count
        df_outcomes = df_outcomes.groupby('Disease Site').apply(split_zeros, training_ratio, test_ratio, include_groups=False).reset_index(drop=False)
        clinical_features = [
            "Age",
            "Sex_Male",
            "T Stage_T3/4",
            "N Stage_N1",
            "N Stage_N2",
            "N Stage_N3",
            "HPV Combined_1.0",
            "Dose",
            "Chemotherapy",
            'ECOG_0', 'ECOG_1', 'ECOG_2',
            'ECOG_3',
            'Disease Site_hypopharynx',
            'Disease Site_larynx',
            'Disease Site_lip & oral cavity',
            'Disease Site_nasal cavity', 'Disease Site_nasopharynx',
            'Disease Site_oropharynx',
            "target_binary",
            "survival_time",
            "death"
        ]

        # binarize T stage as T1/2 = 0, T3/4 = 1
        df_outcomes["T Stage"] = df_outcomes["T Stage"].map({
            "T1": "T1/2",
            "T1a": "T1/2",
            "T1b": "T1/2",
            "T2": "T1/2",
            "T2 (2)": "T1/2",
            "T3": "T3/4",
            "T3 (2)": "T3/4",
            "T4": "T3/4",
            "T4a": "T3/4",
            "T4b": "T3/4"
        })
        # use more fine-grained grouping for N stage
        df_outcomes["N Stage"] = df_outcomes["N Stage"].map({
            "N0": "N0",
            "N1": "N1",
            "N2": "N2",
            "N2a": "N2",
            "N2b": "N2",
            "N2c": "N2",
            "N3": "N3",
            "N3a": "N3",
            "N3b": "N3"
        })

        # 'Curated' radiomic features based on following publications:
        # 1) Vallières, M. et al. Radiomics strategies for risk assessment of tumour failure in head-and-neck cancer. Sci Rep 7, 10117 (2017). doi: 10.1038/s41598-017-10371-5
        # 2) Diamant, A., Chatterjee, A., Vallières, M. et al. Deep learning in head & neck cancer outcome prediction. Sci Rep 9, 2764 (2019). https://doi.org/10.1038/s41598-019-39206-1
        radiomic_features = ['original_glszm_SizeZoneNonUniformity', 'original_glszm_ZoneVariance',
                             'original_glrlm_LongRunHighGrayLevelEmphasis']

        allowed_ds_site_values = ['hypopharynx', 'larynx', 'lip & oral cavity', 'nasal cavity', 'nasopharynx', 'oropharynx']
        df_outcomes = df_outcomes[df_outcomes['Disease Site'].isin(allowed_ds_site_values)]

        df_outcomes = pd.get_dummies(df_outcomes,
                                     columns=["Sex", "HPV Combined", "Disease Site", "T Stage", "N Stage", "ECOG", "Stage"],
                                     dtype=float)
        df_outcomes = df_outcomes.drop(columns=['ECOG_', 'ECOG_4'])

        # df_outcomes.to_csv("/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/df_outcomes.csv")
        # raise ValueError

        # Split the clinical data into the training and testing patients
        # This is so we can combine their respective radiomics data to them
        radcure_train = df_outcomes[df_outcomes['RADCURE-challenge'] == "training"]
        df_outcomes = df_outcomes[df_outcomes['RADCURE-challenge'] == "test"]

        # Combine the clinical and radiomics data for the training and the testing data
        radcure_final = pd.concat([radcure_train.set_index('ID'), radcure_radiomics.set_index('ID')], axis=1,join='inner')
        df_outcomes_final = pd.concat([df_outcomes.set_index('ID'), df_radiomic.set_index('ID')], axis=1, join='inner')

        features = pd.concat([radcure_final, df_outcomes_final])

        # features.to_csv("/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/final_radcure_features.csv")
        # raise ValueError

        features = features.drop(columns=["series_description", "negative_control", "original_shape_Compactness1"])
        baselines = {
            "fuzzyVol_clin": SimpleBaseline(features,
                                            fuzzy_feature=['original_shape_MeshVolume'],
                                            max_features_to_select=0,
                                            colnames_fuzzy=clinical_features + ['original_shape_MeshVolume'],
                                            n_jobs=1),

            "fuzzyVol_rad_currated": SimpleBaseline(features,
                                                    fuzzy_feature=['original_shape_MeshVolume'],
                                                    max_features_to_select=0,
                                                    colnames_fuzzy=radiomic_features + ['original_shape_MeshVolume'],
                                                    n_jobs=1),

            "fuzzyVol_clin+rad_currated": SimpleBaseline(features,
                                                         fuzzy_feature=['original_shape_MeshVolume'],
                                                         max_features_to_select=0,
                                                         colnames_fuzzy=clinical_features + radiomic_features + [
                                                             'original_shape_MeshVolume'],
                                                         n_jobs=1)
        }

        # Predict and evaluate on test
        validation_ids = features.index[features["RADCURE-challenge"] == "test"]
        validation_data = features[features["RADCURE-challenge"] == "test"]
        for name, baseline in baselines.items():
            pred = baseline.get_test_predictions()
            survival_time = pred.pop("survival")
            for i, col in enumerate(survival_time.T):
                pred[f"survival_time_{i}"] = col
            pred = pd.DataFrame(pred, index=validation_ids)

            # Save outputs for evaluation
            out_path_validation = os.path.join(output_path, name + "_train.csv")
            pred.to_csv(out_path_validation)
