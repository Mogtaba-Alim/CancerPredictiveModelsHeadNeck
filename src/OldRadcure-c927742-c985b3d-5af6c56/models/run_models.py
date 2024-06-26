import os

import numpy as np
import pandas as pd

from base_logistic import SimpleBaseline

np.random.seed(42)

if __name__ == '__main__':

    df_radiomic = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE_complete_radiomics_features.csv')
    df_outcomes = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/radcure_challenge.csv')
    output_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output/OldRadcure'
    training_split = pd.read_csv('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/ID_training_split.csv')

    if not os.path.exists('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output'):
        os.makedirs('/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/CancerPredictiveModelsHeadNeck/Output')

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    df_radiomic.rename(columns={'patient_ID': 'ID'}, inplace=True)
    training_split.rename(columns={'patient_id': 'ID'}, inplace=True)
    df_outcomes.rename(columns={'age at dx': 'Age'}, inplace=True)
    # Replace 'RADCURENum' with 'RADCURE-Num' using regular expressions
    df_outcomes['ID'] = df_outcomes['ID'].str.replace(r"(RADCURE)(\d+)", r"\1-\2", regex=True)

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
        'ECOG_0.0', 'ECOG_1.0', 'ECOG_2.0',
        'ECOG_3.0',
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

    features = pd.concat([df_outcomes.set_index('ID'), df_radiomic.set_index('ID'), training_split.set_index('ID')], axis=1, join='inner')
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
    validation_ids = features.index[features["RADCURE-challenge"] == "training"]
    validation_data = features[features["RADCURE-challenge"] == "training"]
    for name, baseline in baselines.items():
        pred = baseline.get_test_predictions()
        survival_time = pred.pop("survival")
        for i, col in enumerate(survival_time.T):
            pred[f"survival_time_{i}"] = col
        pred = pd.DataFrame(pred, index=validation_ids)

        # Save outputs for evaluation
        out_path_validation = os.path.join(output_path, name + "_train.csv")
        pred.to_csv(out_path_validation)
