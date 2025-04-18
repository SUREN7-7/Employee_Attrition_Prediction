from flask import Flask, render_template, request
import pandas as pd
from Models.preprocessing import encode_categorical, scale_features
from Models.predict_model import load_model, make_prediction

app = Flask(__name__)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = {
            'Age': int(request.form['Age']),  
            'BusinessTravel': request.form['BusinessTravel'],
            'DailyRate': float(request.form['DailyRate']),
            'Department' : request.form['Department'],
            'DistanceFromHome' : float(request.form['DistanceFromHome']),
            'Education': float(request.form['Education']),
            'EducationField': request.form['EducationField'],
            'EnvironmentSatisfaction': float(request.form['EnvironmentSatisfaction']),  
            'Gender': request.form['Gender'], 
            'HourlyRate': float(request.form['HourlyRate']),
            'JobInvolvement': float(request.form['JobInvolvement']), 
            'JobLevel': float(request.form['JobLevel']), 
            'JobRole': request.form['JobRole'],  
            'JobSatisfaction': float(request.form['JobSatisfaction']), 
            'MaritalStatus': request.form['MaritalStatus'], 
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'MonthlyRate': float(request.form['MonthlyRate']),
            'NumCompaniesWorked': float(request.form['NumCompaniesWorked']), 
            'OverTime': request.form['OverTime'],  
            'PercentSalaryHike': float(request.form['PercentSalaryHike']),  
            'PerformanceRating': float(request.form['PerformanceRating']),  
            'RelationshipSatisfaction': float(request.form['RelationshipSatisfaction']), 
            'StockOptionLevel': float(request.form['StockOptionLevel']),
            'TotalWorkingYears': float(request.form['TotalWorkingYears']),
            'TrainingTimesLastYear': float(request.form['TrainingTimesLastYear']),
            'WorkLifeBalance': float(request.form['WorkLifeBalance']),
            'YearsAtCompany': float(request.form['YearsAtCompany']), 
            'YearsInCurrentRole': float(request.form['YearsInCurrentRole']),  
            'YearsSinceLastPromotion': float(request.form['YearsSinceLastPromotion']),
            'YearsWithCurrManager': float(request.form['YearsWithCurrManager'])
        }

        input_df = pd.DataFrame([user_input])

        input_df = encode_categorical(input_df)  
        input_df = scale_features(input_df)

        prediction = make_prediction(input_df, model)
        print("Prediction Output: ",prediction)
        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=10000)

