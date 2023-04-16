import pandas as pd
import typing as Ty
import glob
import json



def csv_to_label(input_dir: Ty.AnyStr, output_dir: Ty.AnyStr):
    """
    The following script aims to convert the csv informations to label *.pbtxt files.

    Args:
        input_dir (Ty.AnyStr): The directory that contains all the csvs to be processed (train/validation/test)
        output_dir (Ty.AnyStr): The the directory where is going to be saved the pbtxt generated

    Usage:
        python csv_to_labelmap.py --input_dir=data/  --output_dir=data/
    """
    csvs = glob.glob(f"{input_dir}/*csv")

    csv_list = []
    for csv in csvs:
        df = pd.read_csv(csv)
        csv_list.append(df)

    df = pd.concat(csv_list)

    # getting all the unique labels
    unique_labels = df["class"].unique()
    dict_labels = {}
    for idx, labels in enumerate(unique_labels):

        dict_labels[labels] = idx+1

    # generating the string to be saved in pbtxt format

    with open(f"{output_dir}/label_map.json", "w") as pbfile:
        json.dump(dict_labels, pbfile)


def main():
    csv_to_label('F:/ObjectDetection_FromScratch/data/annotations', 'F:/ObjectDetection_FromScratch/data/annotations')


if __name__ == "__main__":
    main()