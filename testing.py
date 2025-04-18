from Models.preprocessing import load_data, encode_categorical, scale_features, drop_irrelevant_columns

def test_preprocessing():
    file_path = "C:\\SUREN_NEW\\Employee_Attrition_Prediction\\Data\\HR_Employee_Attrition.csv"
    df = load_data(file_path)
    df = drop_irrelevant_columns(df)
    df = encode_categorical(df)
    df = scale_features(df)
    print("\nPreprocessing completed successfully!\n")
    print("\nShape of processed data:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample data:\n", df.head())
    
    # # Optional: Check if target column exists
    if 'Attrition' in df.columns:
        print("\nTarget distribution:\n", df['Attrition'].value_counts())

# Run the test
if __name__ == "__main__":
    test_preprocessing()
