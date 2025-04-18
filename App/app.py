from flask import Flask, render_template, request
import pandas as pd
from Models.preprocessing import encode_categorical, scale_features
from Models.predict_model import load_model, make_prediction

app = Flask(__name__)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from form (assuming form fields are named 'age', 'income', etc.)
        user_input = {
            'Age': int(request.form['age']),
            'MonthlyIncome': float(request.form['income']),
            'JobSatisfaction': int(request.form['job_satisfaction']),
            # Add other fields from the form
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # Preprocess the input data (encoding and scaling)
        input_df = encode_categorical(input_df)  # Make sure you apply all the preprocessing steps here
        input_df = scale_features(input_df)

        # Make prediction
        prediction = make_prediction(input_df, model)

        # Send prediction back to user
        return render_template("result.html", prediction=prediction[0])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
