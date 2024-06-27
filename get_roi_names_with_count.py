import os
import pandas as pd
from pydicom import dcmread
import argparse
import csv

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Path to file with all images")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    dataPath = args.image_dir
    outputPath = args.output_dir

    # Extract ROI names from RTStructs in the dataset
    roiNames = getRTStructsRoiNames(dataPath)
    print("ROI Names Extracted Successfully")

    # Ensure the output directory exists, create if not
    if not os.path.exists(os.path.dirname(outputPath)):
        os.makedirs(os.path.dirname(outputPath))

    # Specify the CSV file name
    csv_file_name = outputPath + "/roi_names.csv"

    # Write ROI names and their counts to the CSV file
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['roi_names', 'count'])
        # Write the dictionary items as rows
        for roi_name, count in sorted(roiNames.items()):
            writer.writerow([roi_name, count])

    print(f'CSV file "{csv_file_name}" has been created successfully.')
    print("ROI Names saved to: " + outputPath)


def getRTStructsRoiNames(dataPath: str):
    '''
    Extracts ROI names from RTStruct files in the dataset.

    Parameters
    ----------
    dataPath : str
        Path to the dataset

    Returns
    -------
    dict
        A dictionary with ROI names as keys and their counts as values
    '''
    # Construct the path to the imgtools file list
    imageDirPath, ctDirName = os.path.split(dataPath)
    imgFileListPath = os.path.join(imageDirPath + '/.imgtools/imgtools_' + ctDirName + '.csv')
    if not os.path.exists(imgFileListPath):
        raise FileNotFoundError(
            "Output for med-imagetools not found for this image set. Check the image_dir argument or run med-imagetools.")

    # Load the complete list of patient image directories from imgtools output
    fullImgFileList = pd.read_csv(imgFileListPath, index_col=0)

    # Extract rows corresponding to RTStruct modality
    allRTStuctRows = fullImgFileList.loc[fullImgFileList['modality'] == "RTSTRUCT"]

    roiNames = {}

    # Iterate over each RTStruct file path and extract ROI names
    for file_path in allRTStuctRows['file_path']:
        rtstructFilePath = imageDirPath + "/" + file_path
        result = getROINames(rtstructFilePath)
        for roi in result:
            if roi in roiNames:
                roiNames[roi] += 1
            else:
                roiNames[roi] = 1
    return roiNames


def getROINames(rtstructPath: str):
    '''
    Extracts ROI names from a single RTStruct file.

    Parameters
    ----------
    rtstructPath : str
        The filepath of the RTStruct file

    Returns
    -------
    set
        A set of ROI names present in the RTStruct file
    '''
    # Read the RTStruct file
    rtstruct = dcmread(rtstructPath, force=True)
    # Extract ROI names
    roi_names = set([roi.ROIName for roi in rtstruct.StructureSetROISequence])

    return roi_names


if __name__ == '__main__':
    main()
