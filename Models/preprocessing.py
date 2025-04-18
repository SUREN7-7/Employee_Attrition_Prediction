import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def encode_categorical(df):
    encoder = LabelEncoder()
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField',
                        'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    return df

def drop_irrelevant_columns(df):
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df


def scale_features(df):
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Attrition']
    for col in numerical_cols:
        df[col] = scaler.fit_transform(df[[col]])
    return df

