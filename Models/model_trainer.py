from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

def train_model(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB()
    }

    best_model_name = None
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    with open('Models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

