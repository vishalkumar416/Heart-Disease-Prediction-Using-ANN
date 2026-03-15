import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Healthcare System", layout="wide")

class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

model = load_model("models/ann_model.h5", custom_objects={'Dense': CustomDense}, compile=False)
st.title("🧠 AI Healthcare Diagnosis Dashboard")
tab1, tab2 = st.tabs(["Patient Prediction", "Doctor Panel"])
# patient prediction
with tab1:
    st.sidebar.header("Patient Data")
    age = st.sidebar.slider("Age", 20, 100, 40)
    sex = st.sidebar.selectbox("Sex", [0,1])
    cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.sidebar.slider("Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar", [0,1])
    restecg = st.sidebar.selectbox("ECG Result", [0,1,2])
    thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Angina", [0,1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope", [0,1,2])
    ca = st.sidebar.selectbox("Major Vessels", [0,1,2,3])
    thal = st.sidebar.selectbox("Thalassemia", [0,1,2,3])

    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                            thalach,exang,oldpeak,slope,ca,thal]])

    if st.button("Predict Disease Risk"):
        st.session_state['pred_done'] = True
        st.session_state['pred'] = model.predict(input_data)[0][0]

    if st.session_state.get('pred_done', False):
        pred = st.session_state['pred']
        st.subheader("Prediction Result")
        if pred > 0.5:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk")

        # risk gauge meter
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred*100,
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range':[0,40],'color':'green'},
                    {'range':[40,70],'color':'yellow'},
                    {'range':[70,100],'color':'red'}
                ]
            }
        ))
        st.plotly_chart(fig)

        # shap explainability
        st.subheader("Model Explanation")
        background = np.random.rand(100,13)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(input_data)
        shap.initjs()
        st.write("Feature importance for this prediction")
        shap.summary_plot(shap_values, input_data, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

        # report generator
        if st.button("Generate Patient Report"):
            filename = "reports/patient_report.pdf"
            c = canvas.Canvas(filename)
            c.drawString(100,750,"AI Healthcare Report")
            c.drawString(100,720,f"Age: {age}")
            c.drawString(100,700,f"Cholesterol: {chol}")
            c.drawString(100,680,f"Blood Pressure: {trestbps}")
            c.drawString(100,650,f"Risk Score: {pred}")
            c.save()
            st.success("Report Generated")

# doctor panel
with tab2:
    st.header("Doctor Analytics Panel")
    data = pd.read_csv("data/heart.csv")
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    st.subheader("Heart Disease Distribution")
    fig1 = plt.figure()
    sns.countplot(x="target", data=data)
    st.pyplot(fig1)
    st.subheader("Feature Correlation")
    fig2 = plt.figure(figsize=(10,6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig2)
    st.subheader("Confusion Matrix")

    from sklearn.model_selection import train_test_split
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    preds = model.predict(X_test)
    preds = (preds > 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig3 = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    st.pyplot(fig3)