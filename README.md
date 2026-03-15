# 🧠 Heart Disease Prediction Using Artificial Neural Networks (ANN)

An AI-powered healthcare dashboard that predicts the risk of heart disease using **Artificial Neural Networks (ANN)** and visualizes results through an **interactive Streamlit dashboard**.

The system analyzes patient health parameters such as age, blood pressure, cholesterol, ECG results, and heart rate to estimate the probability of heart disease.

---

# 🚀 Project Overview

This project demonstrates how **Deep Learning** can assist healthcare professionals in predicting heart disease risk.

The application includes:

• ANN-based prediction model
• Interactive healthcare dashboard
• Risk probability gauge visualization
• Explainable AI using SHAP
• Doctor analytics panel
• Confusion matrix visualization
• Patient PDF report generation

---

# 🏗 Project Architecture

Patient Data → ANN Model → Risk Prediction → Dashboard Visualization

Workflow:

1. User enters patient health parameters
2. ANN model processes the inputs
3. Model predicts heart disease probability
4. Dashboard displays:

   * Risk score
   * Gauge meter
   * Feature importance
   * Analytics

---

# 📊 Dataset

The dataset used is the **Heart Disease Dataset (UCI Machine Learning Repository)**.

It includes medical features such as:

| Feature             | Description                 |
| ------------------- | --------------------------- |
| Age                 | Age of patient              |
| Sex                 | Gender                      |
| Blood Pressure      | Resting blood pressure      |
| Cholesterol         | Serum cholesterol           |
| Fasting Blood Sugar | Blood sugar after fasting   |
| ECG Result          | Electrocardiographic result |
| Max Heart Rate      | Maximum heart rate achieved |
| Exercise Angina     | Chest pain during exercise  |
| ST Depression       | ECG depression              |
| Slope               | ST segment slope            |
| Major Vessels       | Number of vessels           |
| Thalassemia         | Blood disorder type         |

A full explanation of these features is included in:

feature_explanation.pdf

This document explains how the ANN model uses these inputs. 

---

# 🧠 Artificial Neural Network Model

The ANN model uses a **fully connected neural network architecture**:

Input Layer → Hidden Layers → Output Layer

Example architecture:

Input Layer: 13 neurons
Hidden Layer 1: 64 neurons (ReLU)
Hidden Layer 2: 32 neurons (ReLU)
Hidden Layer 3: 16 neurons (ReLU)
Output Layer: 1 neuron (Sigmoid)

Output:

0 → Low Risk
1 → High Risk

---

# 📂 Project Structure

```
Heart-Disease-Prediction-Using-ANN

data/
    heart.csv

models/
    ann_model.h5

notebooks/
    ANN_Project.ipynb

reports/

app.py
requirements.txt
runtime.txt
feature_explanation.pdf
README.md
```

---

# 🖥 Dashboard Features

### Patient Prediction Panel

• Enter patient health data
• Predict heart disease risk
• Display risk gauge meter

### Explainable AI

• SHAP feature importance visualization

### Doctor Analytics Panel

• Dataset preview
• Heart disease distribution
• Correlation heatmap
• Confusion matrix

### Patient Report Generator

• Generates PDF medical report

---

# ⚙ Installation (Local Setup)

Clone the repository:

```
git clone https://github.com/vishalkumar416/Heart-Disease-Prediction-Using-ANN.git
```

Navigate into the project:

```
cd Heart-Disease-Prediction-Using-ANN
```

Install dependencies:

```
pip install -r requirements.txt
```

The required libraries include: 

streamlit
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
plotly
shap
reportlab

Run the application:

```
streamlit run app.py
```

The dashboard will open in your browser.

---

# ☁ Deployment on Render

The project is deployed using **Render cloud platform**.

Steps to deploy:

1. Push project to GitHub
2. Go to Render Dashboard
3. Create a new **Web Service**
4. Connect your GitHub repository
5. Configure deployment

Environment:

```
Python
```

Build command:

```
pip install -r requirements.txt
```

Start command:

```
streamlit run app.py --server.port 10000 --server.address 0.0.0.0
```

---

# ⚠ Deployment Errors Faced

During deployment several issues occurred.

### 1. TensorFlow Installation Error

Error:

```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

Cause:

Render used **Python 3.14**, but TensorFlow does not support Python 3.14.

Solution:

Specify Python version using runtime file.

runtime.txt

```
python-3.10.13
```

This forces Render to install Python 3.10. 

---

### 2. Keras Dense Layer Deserialization Error

Error:

```
Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

Cause:

Model saved with different Keras version.

Solution:

Custom Dense layer used in app.py to ignore the incompatible parameter.

See implementation in the application code. 

---

### 3. Git Push Rejection

Error:

```
! [rejected] main -> main (fetch first)
```

Cause:

Remote repository contained files not present locally.

Solution:

```
git pull origin main --allow-unrelated-histories
```

Then push again.

---

# 📈 Model Evaluation

The dashboard includes model evaluation metrics such as:

• Confusion Matrix
• Feature Correlation
• Heart Disease Distribution

These visualizations help doctors understand model performance.

---

# 🛠 Technologies Used

Python
TensorFlow / Keras
Streamlit
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Plotly
SHAP
ReportLab

---

# 📌 Future Improvements

Possible enhancements:

• Multi-disease prediction system
• Integration with hospital databases
• X-ray image diagnosis using CNN
• LSTM for patient health history
• Mobile healthcare application

---

# 👨‍💻 Author

Vishal Kumar

B-Tech in Artificial Intelligence and Data Science

GitHub:
https://github.com/vishalkumar416
