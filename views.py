from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

# Load and preprocess data
data = pd.read_csv(r"C:\Users\vaigadhe\PycharmProjects\Diabetes_Prediction\Diabetes_Prediction\Project 2.csv")
X = data.drop("Outcome", axis=1)
Y = data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

def result(request):
    if request.method == 'GET':
        # Parse input values from the request
        val1 = float(request.GET.get('n1'))
        val2 = float(request.GET.get('n2'))
        val3 = float(request.GET.get('n3'))
        val4 = float(request.GET.get('n4'))
        val5 = float(request.GET.get('n5'))
        val6 = float(request.GET.get('n6'))
        val7 = float(request.GET.get('n7'))
        val8 = float(request.GET.get('n8'))

        # Make prediction
        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

        # Interpret prediction
        result1 = ""
        if pred == [1]:
            result1 = "Oops! You have DIABETES ????."
        else:
            result1 = "Great! You DON'T have diabetes ????."

        return render(request, "predict.html", {"result2": result1})
