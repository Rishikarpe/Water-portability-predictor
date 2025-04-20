from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model and the scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this is correct

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    ph = float(request.form['ph'])
    hardness = float(request.form['hardness'])
    solids = float(request.form['solids'])
    chloramines = float(request.form['chloramines'])
    sulfate = float(request.form['sulfate'])
    conductivity = float(request.form['conductivity'])
    organic_carbon = float(request.form['organic-carbon'])
    turbidity = float(request.form['turbidity'])

    # Prepare input data for prediction (Make sure this has 8 features as in the training data)
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, turbidity]])

    # Scale the data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the Random Forest model
    prediction = rf_model.predict(input_data_scaled)

    # Display result in the HTML template
    result = "Potable" if prediction == 1 else "Non-Potable"
    return render_template('index.html', result=result)  # Pass the result to the template

if __name__ == "__main__":
    app.run(debug=True)
