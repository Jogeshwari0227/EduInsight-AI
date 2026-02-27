import streamlit as st
import joblib
import numpy as np

# 1. Load the trained "brain" (The Random Forest Model)
model = joblib.load('model/model.pkl')

st.title("📘 EduInsight AI: Student Performance Predictor")
st.write("Predict. Explain. Improve.")

# 2. User Input Section (Teacher enters student behavior)
st.subheader("Enter Student Metrics")
col1, col2 = st.columns(2)

with col1:
    raised_hands = st.slider("Hands Raised", 0, 100, 30)
    resources = st.slider("Resources Viewed", 0, 100, 50)
    announcements = st.slider("Announcements Viewed", 0, 100, 40)

with col2:
    discussion = st.slider("Discussion Participation", 0, 100, 20)
    absences = st.selectbox("Absence Days", options=["Under-7", "Above-7"])
    # Convert selection to numerical for the model
    abs_val = 0 if absences == "Under-7" else 1

# 3. Prediction Logic
if st.button("Predict Performance"):
    # Arrange inputs for the model
    features = np.array([[raised_hands, resources, announcements, discussion, abs_val]])
    prediction = model.predict(features)[0]
    
    # Map numbers back to labels
    results = {0: "Low (High Risk) ⚠️", 1: "Medium (Moderate Risk) 🟠", 2: "High (Safe) ✅"}
    st.subheader(f"Prediction: {results[prediction]}")

    # 4. Rule-Based Suggestion Engine
    st.write("---")
    st.subheader("💡 Improvement Suggestions")
    
    suggestions = []
    if raised_hands < 30:
        suggestions.append("- Encourage more active participation in class.")
    if resources < 40:
        suggestions.append("- Suggest reviewing online course materials more frequently.")
    if abs_val == 1:
        suggestions.append("- High absence detected. Schedule a meeting to discuss attendance.")
    if discussion < 30:
        suggestions.append("- Promote engagement in peer discussion forums.")
    
    if not suggestions:
        st.success("Student is performing well across all monitored behaviors!")
    else:
        for s in suggestions:
            st.write(s)