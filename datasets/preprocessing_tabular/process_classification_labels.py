"""
Cardiac Tabular Labels and Image Path Preprocessing Pipeline

This script processes tabular cardiac data and prepares labels for disease prediction tasks, 
along with corresponding image paths for training, validation, and testing.

Steps:
1. Load cleaned cardiac feature data and selected imaging IDs.
2. Merge additional metadata (e.g., date of attending imaging center) with features.
3. Generate binary labels for diseases:
   - Using ICD10 diagnosis codes for conditions like CAD, Stroke, Hypertension, Infarct, and Diabetes.
   - Using feature columns for conditions like high blood pressure.
4. Split subjects with available images into train/validation/test sets while preserving class balance.
5. Save:
   - Labels CSV files for each target.
   - Pickled dictionaries containing processed CMR image paths for each split.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# --- Environment variables ---
TABULAR_BASE = Path(os.environ["TABULAR_BASE"])
DATALOADER_TABULAR_FILE_ROOT = Path(os.environ["DATALOADER_TABULAR_FILE_ROOT"])
OUT_CLEANED_FEATURES_PATH = Path(os.environ["OUT_CLEANED_FEATURES_PATH"])
IMAGE_PROCESS_ROOT = Path(os.environ["IMAGE_PROCESS_ROOT"])
DATALOADER_IMAGE_FILE_ROOT = Path(os.environ["DATALOADER_IMAGE_FILE_ROOT"])
SELECTED_IDS_PICKLE = Path(os.environ["SELECTED_IDS_PICKLE"])
DATALOADER_IMAGE_FILE_ROOT.mkdir(exist_ok=True)
DATALOADER_TABULAR_FILE_ROOT.mkdir(exist_ok=True)

# --- Load base data ---
# Load all image selected image IDs
with open(SELECTED_IDS_PICKLE, "rb") as f:
    all_image_ids = pickle.load(f)

# Load cardiac feature data
data_df = pd.read_csv(OUT_CLEANED_FEATURES_PATH)
date_attended_imaging = pd.read_csv(TABULAR_BASE / "col67.txt")
date_attended_imaging.rename(
    columns={"eid": "eid", "53-2.0": "Date of attending imaging centre-2.0"},
    inplace=True,
)
data_df_extended = data_df.merge(date_attended_imaging, left_on='eid', right_on='eid', how='inner')
assert len(data_df_extended) == len(data_df)


def build_labels_icd10(target_name: str, icd10_codes: list[str]) -> pd.DataFrame:
    """Create binary labels for a disease based on ICD10 diagnosis codes."""
    diag_name = "Diagnoses - ICD10-0."
    date_name = "Date of first in-patient diagnosis - ICD10-0."
    array_length = 243

    all_target_indices, all_target_ids, all_target_dates = [], [], []
    for i in range(array_length):
        mask = data_df[f"{diag_name}{i}"].isin(icd10_codes)
        all_target_dates.extend(data_df.loc[mask, f"{date_name}{i}"])
        all_target_indices.extend(data_df.loc[mask].index)
        all_target_ids.extend(data_df.loc[mask, "eid"])

    date_attending_centre = pd.Series(
        [data_df_extended.loc[i, "Date of attending imaging centre-2.0"] for i in all_target_indices],
        dtype="datetime64[ns]",
    )

    target_df = pd.DataFrame(
        {"eid": all_target_ids, "target date": all_target_dates, "imaging date": date_attending_centre}
    )

    labels_data = {
        "eid": data_df["eid"],
        f"Diagnosed_{target_name}": data_df["eid"].isin(target_df["eid"]).astype(int),
    }
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(DATALOADER_TABULAR_FILE_ROOT / f"labels_{target_name}.csv", index=False)
    return labels_df


def build_labels_feature(target_name: str, feature: str) -> pd.DataFrame:
    """Create binary labels for a disease based on a feature column (not ICD10)."""
    labels_data = {
        "eid": data_df["eid"],
        f"Diagnosed_{target_name}": (~data_df[feature].isna()).astype(int),
    }
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(DATALOADER_TABULAR_FILE_ROOT / f"labels_{target_name}.csv", index=False)
    return labels_df


def split_and_save_paths(labels_df: pd.DataFrame, target_name: str):
    """Split into train/val/test and save image paths."""
    labels_with_image_df = labels_df[labels_df["eid"].isin(all_image_ids)]
    df_train, df_temp = train_test_split(
        labels_with_image_df,
        test_size=2000,
        stratify=labels_with_image_df[f"Diagnosed_{target_name}"],
        random_state=42,
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1000,
        stratify=df_temp[f"Diagnosed_{target_name}"],
        random_state=42,
    )

    image_paths = {}
    for split, df in {"train": df_train, "val": df_val, "test": df_test}.items():
        set_paths = [IMAGE_PROCESS_ROOT / str(eid) / "processed_seg_allax.npz" for eid in df["eid"]]
        print(f"{target_name}_{split}: {sum(df[f'Diagnosed_{target_name}'])}/{len(set_paths)}")
        image_paths[split] = set_paths

    with open(DATALOADER_IMAGE_FILE_ROOT / f"recon_cmr_subject_paths_50k_{target_name}.pkl", "wb") as f:
        pickle.dump(image_paths, f)


# --- Define targets ---
targets = {
    "CAD": ['I200', 'I201', 'I208', 'I209', 'I220', 'I221', 'I228', 'I229', 'I210', 'I211',
            'I212', 'I213', 'I214', 'I219','I240', 'I248', 'I249', 'I250', 'I251', 'I252',
            'I253', 'I254', 'I255', 'I256', 'I258', 'I259'],
    "Stroke": ['I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I636', 'I638', 'I639'],
    "Hypertension": ['I10', 'I110', 'I119', 'I120', 'I129', 'I130', 'I131', 'I132',
                     'I139', 'I150', 'I151', 'I152', 'I158', 'I159'],
    "Infarct": ['I210', 'I211', 'I212', 'I213', 'I214', 'I219', 'I252'],
    "Diabetes": ['E100','E101','E102','E103','E104','E105','E106','E107','E108','E109',
                 'E110','E111','E112','E113','E114','E115','E116','E117','E118','E119',
                 'E121','E123','E125','E128','E129','E130','E131','E132','E133','E134',
                 'E135','E136','E137','E138','E139','E140','E141','E142','E143','E144',
                 'E145','E146','E147','E148','E149'],
}

# --- Run pipeline ---
for disease, icd10_codes in targets.items():
    labels_df = build_labels_icd10(disease, icd10_codes)
    split_and_save_paths(labels_df, disease)

# Special case: high blood pressure feature
labels_df = build_labels_feature("High_blood_pressure", "Age high blood pressure diagnosed-2.0")
split_and_save_paths(labels_df, "High_blood_pressure")
