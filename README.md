# 💧 Water Portability Predictor

**Water Portability Predictor** is a Python-based web application that predicts whether water is potable (safe for drinking) or not, based on various physicochemical properties. Using machine learning models served through a Flask API and an interactive web interface, this tool empowers users to assess water safety with ease.

---

## 🔍 Features

- **Multi-Model Approach**: Includes Logistic Regression, K‑Nearest Neighbors (KNN), and Random Forest classifiers.
- **Live Inference via Web Form**: Input key water quality metrics and receive an instant drinkability prediction.
- **Trained & Serialized Models**: Reusable models saved as `.pkl` files for quick deployment.
- **Standardized Input Pipeline**: Uses feature scaling (`scaler.pkl`) for consistent preprocessing.

---

## ⚙️ How It Works

1. **Dataset**  
   - `water_potability.csv` or `dataset.csv`: Contains measurements such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity, and a potability label (0 = non‑potable, 1 = potable).

2. **Model Training** (`main.py`)  
   - Reads and preprocesses the dataset.
   - Splits data into training and testing sets.
   - Trains Logistic Regression, KNN, and Random Forest models.
   - Saves trained models and a Scaler via `pickle`.

3. **Web Application** (`server.py`)  
   - Flask server hosting an HTML form.
   - User inputs metrics → values are scaled → model returns "Potable" or "Not Potable".

4. **Front-end** (`templates/`, `static/`)  
   - Simple UI with form submission and result display.
   - Implemented using HTML, CSS, and optional JavaScript.

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install flask pandas scikit-learn numpy

---

# Quick Start
1. Train Models (if you want to retrain):
- python main.py
Generates logistic_regression_model.pkl, knn_model.pkl, random_forest_model.pkl, and scaler.pkl.

2. Run Web App:
-python server.py
Visit: http://127.0.0.1:5000
Enter water parameters and get your prediction!

---

# Models & Usage
Logistic Regression – Quick and interpretable baseline.
KNN – Simple instance-based learner.
Random Forest – Ensures robustness and handles non-linearity well.
Feature Scaling – All models expect input normalized via scaler.pkl.

---
