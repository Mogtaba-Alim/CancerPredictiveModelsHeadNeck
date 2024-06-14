import os

import numpy as np
import pandas as pd
from collections import Counter

from base_logistic import SimpleBaseline

np.random.seed(42)

def check_file_exists(file_path):
    return os.path.isfile(file_path)


if __name__ == '__main__':

    negative_controls = ["default", "randomized_full", "randomized_roi", "randomized_non_roi", "shuffled_full", "shuffled_roi",
                      "shuffled_non_roi","randomized_sampled_full", "randomized_sampled_roi", "randomized_sampled_non_roi"]

    for nc in negative_controls:

        if nc == "default":
            df_radiomic_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/HEAD-NECK-RADIOMICS-HN1/radiomicfeatures_HEAD-NECK-RADIOMICS-HN1.csv'
        else:
            df_radiomic_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/HEAD-NECK-RADIOMICS-HN1/radiomicfeatures_' + nc + '_HEAD-NECK-RADIOMICS-HN1.csv'

        if not check_file_exists(df_radiomic_path):
            continue
        else:
            df_radiomic = pd.read_csv(df_radiomic_path)


        df_outcomes = pd.read_csv(
            '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/HEAD-NECK-RADIOMICS-HN1/HEAD-NECK-RADIOMICS-HN1-Clinical-data.csv')
        output_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output/HEAD-NECK-RADIOMICS-HN1/' + nc
        radcure_train = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/df_outcomes.csv')
        radcure_radiomics = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE/radiomicfeatures_RADCURE.csv')

        if not os.path.exists('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output'):
            os.makedirs('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Make Radcure only train
        radcure_train = radcure_train[radcure_train['RADCURE-challenge'] == 'training']
        radcure_radiomics.rename(columns={'patient_ID': 'ID'}, inplace=True)

        df_radiomic.rename(columns={'patient_ID': 'ID'}, inplace=True)
        df_outcomes.rename(columns={'id': 'ID'}, inplace=True)

        # Drop the compactness radiomic feature
        radcure_radiomics = radcure_radiomics.drop(columns=["original_shape_Compactness1"])

        # Rename to Disease Site
        df_outcomes.rename(columns={'index_tumour_location': 'Disease Site'}, inplace=True)

        # Rename the Sec column and capitalize the first letter
        df_outcomes.rename(columns={'biological_sex': 'Sex'}, inplace=True)
        df_outcomes['Sex'] = df_outcomes['Sex'].map({'male': 'Male', 'female': 'Female'})

        # Rename the ECOG column
        df_outcomes.rename(columns={'performance_status_ecog': 'ECOG'}, inplace=True)

        # Fill empty values in the ECOG column with 0
        df_outcomes['ECOG'] = df_outcomes['ECOG'].fillna(0)

        # Rename the HPV status column and binarize it
        df_outcomes.rename(columns={'overall_hpv_p16_status': 'HPV Combined'}, inplace=True)
        df_outcomes['HPV Combined'] = df_outcomes['HPV Combined'].map({'negative': 0, 'positive': 1})

        # Rename columns and change values for specific columns
        rename_map = {
            'clin_t': 'T Stage',
            'clin_n': 'N Stage',
            'clin_m': 'M Stage'
        }
        df_outcomes = df_outcomes.rename(columns=rename_map)

        # Format the values of the cancer stage columns to fit Radcure
        for col in rename_map.values():
            letter = col.split()[0]
            df_outcomes[col] = df_outcomes[col].apply(lambda x: f"{letter}{x}")

        # Rename the Stage column
        df_outcomes.rename(columns={'ajcc_stage': 'Stage'}, inplace=True)
        df_outcomes['Stage'] = df_outcomes['Stage'].str.upper()

        # Rename the Survival time column
        df_outcomes.rename(columns={'overall_survival_in_days': 'survival_time'}, inplace=True)
        df_outcomes['survival_time'] = df_outcomes['survival_time'] / 365

        # Create the target_binary column
        df_outcomes['target_binary'] = df_outcomes['survival_time'].apply(lambda x: 1 if x > 2 else 0)

        # Rename the Death column
        df_outcomes.rename(columns={'event_overall_survival': 'death'}, inplace=True)

        # Rename the Age column
        df_outcomes.rename(columns={'age_at_diagnosis': 'Age'}, inplace=True)

        # Create the testing Column
        df_outcomes['RADCURE-challenge'] = 'test'

        # Rename the Dose column
        df_outcomes.rename(columns={'radiotherapy_refgydose_total_highriskgtv': 'Dose'}, inplace=True)

        # Create the Chemotherapy column with binary values
        df_outcomes['Chemotherapy'] = df_outcomes['chemotherapy_given'].apply(lambda x: 0 if x == 'none' else 1)

        columns_to_keep = [
            'ID', 'target_binary', 'survival_time', 'death', 'Age',
            'Sex', 'T Stage', 'N Stage', 'Stage', 'Dose', 'Chemotherapy',
            'HPV Combined', 'Disease Site', 'ECOG', 'RADCURE-challenge'
        ]

        # Keep only the specified columns
        df_outcomes = df_outcomes[columns_to_keep]

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
        df_outcomes = pd.get_dummies(df_outcomes,
                                     columns=["Sex", "HPV Combined", "Disease Site", "T Stage", "N Stage", "ECOG", "Stage"],
                                     dtype=float)

        for i in range(4):
            df_outcomes.rename(columns={'ECOG_' + str(i) + ".0": 'ECOG_' + str(i)}, inplace=True)

        df_outcomes['Stage_0'] = 0.0
        df_outcomes['Stage_IV'] = 0.0
        df_outcomes['Disease Site_hypopharynx'] = 0.0
        df_outcomes['Disease Site_lip & oral cavity'] = 0.0
        df_outcomes['Disease Site_nasal cavity'] = 0.0
        df_outcomes['Disease Site_nasopharynx'] = 0.0

        # df_outcomes.to_csv("/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/HEAD-NECK-RADIOMICS-HN1/df_outcomes.csv")
        # raise ValueError

        radcure_final = pd.concat([radcure_train.set_index('ID'), radcure_radiomics.set_index('ID')], axis=1, join='inner')
        df_outcomes_final = pd.concat([df_outcomes.set_index('ID'), df_radiomic.set_index('ID')], axis=1, join='inner')

        features = pd.concat([radcure_final, df_outcomes_final])

        features = features.drop(columns=["series_description", "negative_control", "Unnamed: 0", "level_1"])
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
