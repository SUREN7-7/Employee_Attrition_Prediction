import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_model(model_path='Models/best_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_prediction(input_data, model):
    prediction = model.predict(input_data)
    pred =  str(prediction[0])
    if(pred=='0'):
        return "NO"
    else:
        return "YES"