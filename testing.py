from Models.preprocessing import load_data, encode_categorical, scale_features, drop_irrelevant_columns
from Models.model_trainer import train_model
from Models.predict_model import load_model, make_prediction, test_model_on_data
def test_preprocessing():
    
    file_path = "C:\\SUREN_NEW\\Employee_Attrition_Prediction\\Data\\HR_Employee_Attrition.csv"
    df = load_data(file_path)
    df = drop_irrelevant_columns(df)
    df = encode_categorical(df)
    df = scale_features(df)
        
    train_model(df)

    model = load_model()
    # make_prediction(input_data, model)

if __name__ == "__main__":
    test_preprocessing()
