import pandas as pd

# Load the Excel file
file_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/RADCURE-DA-CLINICAL-2.xlsx'
data = pd.read_excel(file_path)

# Select the specified columns
selected_columns = ['patient_id', 'RADCURE-challenge']
filtered_data = data[selected_columns]

# Save the filtered data to a CSV file
output_file_path = '/Users/maximus/Desktop/FALL2023/BCB430/code/headNeckModels/ClinicalData/ID_training_split.csv'
filtered_data.to_csv(output_file_path, index=False)