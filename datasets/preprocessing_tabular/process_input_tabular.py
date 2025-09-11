"""
Tabular Data Preprocessing for Cardiac Features

Steps:
1. Load raw tabular data and feature definitions.
2. Vectorize numerical, single categorical, and multi-categorical features.
3. Impute missing values (mean for numerical, most frequent for categorical).
4. Optional: z-score normalization for numerical features.
5. Optional: one-hot encoding for multi-categorical features.
6. Save intermediate and final preprocessed CSVs.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle


# --- Paths (from environment variables)---
DATALOADER_TABULAR_FILE_ROOT = Path(os.environ["DATALOADER_TABULAR_FILE_ROOT"])
FEATURE_NAMES_IN = Path(os.environ["FEATURE_NAMES_IN"])
FEATURE_NAMES_OUT = Path(os.environ["FEATURE_NAMES_OUT"])
IN_CLEANED_FEATURES_PATH = Path(os.environ["IN_CLEANED_FEATURES_PATH"])
OUT_CLEANED_FEATURES_PATH = Path(os.environ["OUT_CLEANED_FEATURES_PATH"])

RAW_TABULAR_DATA_PATH = Path(os.environ["RAW_TABULAR_DATA_PATH"])
PREPROCESSED_TABULAR_DATA_PATH = Path(os.environ["PREPROCESSED_TABULAR_DATA_PATH"])
PREPROCESSED_TABULAR_DATA_PATH.parent.mkdir(exist_ok=True)


# --- Configuration ---
NORMALIZATION = True
ONE_HOT_ENCODE = True


# --- Load data and feature definitions ---
input_raw_data_df = pd.read_csv(IN_CLEANED_FEATURES_PATH)
output_raw_data_df = pd.read_csv(OUT_CLEANED_FEATURES_PATH)
with open(FEATURE_NAMES_IN, "r") as f:
    input_feature_data = json.load(f)

with open(FEATURE_NAMES_OUT, "r") as f:
    output_feature_data = json.load(f)


# --- Generate clean table based on the selected features ---
def cardiac_features_to_vector_no_onehot_df(df, feature_data):

    def clean_categorical(value:int):
        """Ensures nans are properly saved and categorical variables are ints."""
        if pd.isna(value):
            return np.nan
        return int(value)
    
    vec = []
    indices = {}
    vec.append(df['eid'])
    
    # Numerical
    for name in feature_data["numerical"]:
        vec.append(df[name])
    indices["numerical"] = list(range(len(vec)))
    
    # Single categorical
    for name in feature_data["single_categorical"]:
        vec.append(df[name].apply(clean_categorical))
    indices["categorical_single"] = list(range((indices["numerical"][-1] + 1), len(vec)))

    # Multiple categorical
    for name, data in feature_data["multi_categorical"].items():
        feature_values = df[name].apply(clean_categorical)
        use_base = data[1]
        if use_base and not ONE_HOT_ENCODE:
            feature_values = feature_values.apply(lambda x: x - 1 if pd.notnull(x) else x)
        vec.append(feature_values)
    indices["categorical_multi"] = list(range((indices["categorical_single"][-1] + 1), len(vec)))
    return vec, indices


def df_to_one_hot_encode_df(df):
    def one_hot_encode(value:int, num_classes:int, one_based:bool=False):
        """
            Makes a one hot encoding of an integer categorical variable.
            It is assumed the values start at 0. Some start at one and thus the one_based flag should be used.
        """
        if pd.isna(value):
            vec = np.full([num_classes], np.nan)
        else:
            value = int(value)
            if one_based:
                vec = np.eye(num_classes, dtype=int)[value-1]
            else:
                vec = np.eye(num_classes, dtype=int)[value]
        return vec
    vec = []
    num_classes = [3, 7, 6, 6, 6, 6, 6, 4, 3, 3, 5, 5, 6, 3, 3, 8, 8, 8, 3, 6, 7, 8, 7, 8, 8, 8, 7, 8, 3, 4, 3, 3]

    vec.append(df['Sleeplessness / insomnia-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3, one_based=True)))
    vec.append(df['Frequency of heavy DIY in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Alcohol intake frequency.-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6, one_based=True)))
    vec.append(df['Processed meat intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Beef intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Pork intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Lamb/mutton intake-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Overall health rating-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=4, one_based=True)))
    vec.append(df['Alcohol usually taken with meals-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Alcohol drinker status-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Frequency of drinking alcohol-0.0'].apply(lambda col: one_hot_encode(value=col, num_classes=5)))
    vec.append(df['Frequency of consuming six or more units of alcohol-0.0'].apply(lambda col: one_hot_encode(value=col, num_classes=5, one_based=True)))
    vec.append(df['Amount of alcohol drunk on a typical drinking day-0.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6, one_based=True)))
    vec.append(df['Falls in the last year-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3, one_based=True)))
    vec.append(df['Weight change compared with 1 year ago-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Number of days/week walked 10+ minutes-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Number of days/week of moderate physical activity 10+ minutes-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Number of days/week of vigorous physical activity 10+ minutes-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Usual walking pace-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3, one_based=True)))
    vec.append(df['Frequency of stair climbing in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=6)))
    vec.append(df['Frequency of walking for pleasure in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Duration walking for pleasure-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Frequency of strenuous sports in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Duration of strenuous sports-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Duration of light DIY-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Duration of heavy DIY-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Frequency of other exercises in last 4 weeks-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=7)))
    vec.append(df['Duration of other exercises-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=8)))
    vec.append(df['Current tobacco smoking-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Past tobacco smoking-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=4, one_based=True)))
    vec.append(df['Smoking/smokers in household-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    vec.append(df['Smoking status-2.0'].apply(lambda col: one_hot_encode(value=col, num_classes=3)))
    one_hot_df = pd.concat(vec,axis=1)
    one_hot_df = one_hot_df.reset_index(drop=True)
    return one_hot_df, num_classes


df_vec, indices = cardiac_features_to_vector_no_onehot_df(output_raw_data_df, output_feature_data)
out_df = pd.concat(df_vec , axis=1)
out_df = out_df.reset_index(drop=True)
out_df.to_csv(RAW_TABULAR_DATA_PATH, index=False)
print("Processed the raw tabular data for prediction (output)!")



df_vec, indices = cardiac_features_to_vector_no_onehot_df(input_raw_data_df, input_feature_data)
df = pd.concat(df_vec , axis=1)
df = df.reset_index(drop=True)


# --- Impute numerical missing values with the mean ---
eid_column = df["eid"]

raw_numerical_df = df.iloc[:, indices["numerical"][1:]]
singlecategorical_df = df.iloc[:, indices["categorical_single"]]
multicategorical_df = df.iloc[:, indices["categorical_multi"]]

imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
imp_mean.fit(raw_numerical_df)
numerical_df = imp_mean.transform(raw_numerical_df)
numerical_df = pd.DataFrame(numerical_df, columns=df.columns[indices["numerical"][1:]])

# --- z-score standard normalization ---
if NORMALIZATION:
    scaler = StandardScaler()
    normalized_df = scaler.fit_transform(numerical_df)
    numerical_df = pd.DataFrame(normalized_df, columns=numerical_df.columns)
    # Save the scaler
    with open(f'{DATALOADER_TABULAR_FILE_ROOT}/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

# # --- Save the scaled version of the raw data. No data imputation! ---
# raw_numerical_df = df.iloc[:, indices["numerical"][1:]]
# raw_categorical_df = df.iloc[:, indices["numerical"][1:]]
# raw_scaled_numerical_df = numerical_df.where(~raw_numerical_df.isna(), np.nan)
# raw_scaled_df = pd.concat([eid_column, raw_scaled_numerical_df, singlecategorical_df, multicategorical_df], axis=1)
# raw_scaled_df.to_csv(RAW_SCALED_TABULAR_DATA_PATH, index=False)


# --- Impute caregorical missing values with the most frequent value ---
categorical_indices = indices["categorical_single"] + indices["categorical_multi"]
categorical_df = df.iloc[:, categorical_indices]
imp_most_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent", keep_empty_features=True)
imp_most_freq.fit(categorical_df)
df.iloc[:, categorical_indices] = imp_most_freq.transform(df.iloc[:, categorical_indices])

singlecategorical_df = df.iloc[:, indices["categorical_single"]]
multicategorical_df = df.iloc[:, indices["categorical_multi"]]

if ONE_HOT_ENCODE:
    multi_c_df, num_classes = df_to_one_hot_encode_df(multicategorical_df)
    expanded_vecs = []
    for col_name in multi_c_df.columns:
        expanded_vec = pd.DataFrame(list(multi_c_df[col_name]), 
                                    columns=[f"{col_name}_{i}" for i in range(len(multi_c_df[col_name][0]))])
        expanded_vecs.append(expanded_vec)
    multicategorical_df = pd.concat(expanded_vecs, axis=1) 


# --- Combine final preprocessed dataframe ---
preprocessed_df = pd.concat([eid_column, numerical_df, singlecategorical_df, multicategorical_df], axis=1)
preprocessed_df.to_csv(PREPROCESSED_TABULAR_DATA_PATH, index=False)
print(f"Preprocessed input tabular data!")
