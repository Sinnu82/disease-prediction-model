# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and symptom list
with open('disease_model.pkl', 'rb') as f:
    model, symptom_list = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    # Create input vector
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    prediction = model.predict([input_data])[0]

    return render_template('index.html', symptoms=symptom_list, prediction=prediction, selected=selected_symptoms)

if __name__ == '__main__':
    app.run(debug=True)
