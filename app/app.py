import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="EduInsight AI", layout="wide")

# 2. Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 3. Sidebar
with st.sidebar:
    st.title("🛡️ Educator Portal")
    if st.session_state['logged_in']:
        st.success("Authenticated")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
    st.markdown("---")
    st.info("**EduInsight AI**\n\nPredicting student outcomes via behavioral patterns.")

# 4. Login Gate
if not st.session_state['logged_in']:
    st.title("🔐 Educator Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u == "faculty1" and p == "divA":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid credentials")

# 5. Main Dashboard
else:
    @st.cache_resource
    def load_model():
        return joblib.load('model/model.pkl')
    
    model = load_model()

    st.title("📘 EduInsight AI: Predictive Dashboard")
    
    col_input, col_display = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("📝 Input Student Metrics")
        raised_hands = st.slider("Hands Raised", 0, 100, 30)
        discussion = st.slider("Discussion Participation", 0, 100, 20)
        resources = st.slider("Resources Viewed", 0, 100, 50)
        announcements = st.slider("Announcements Viewed", 0, 100, 40)
        abs_days = st.selectbox("Absence Days", options=["Under-7", "Above-7"])
        abs_val = 0 if abs_days == "Under-7" else 1

    with col_display:
        st.subheader("🔍 AI Prediction & Analysis")
        
        if st.button("Run AI Analysis", use_container_width=True):
            features = np.array([[raised_hands, resources, announcements, discussion, abs_val]])
            prediction = model.predict(features)[0]
            
            # --- Result Header ---
            if prediction == 2:
                st.success("### Prediction: High Performance ✅")
            elif prediction == 1:
                st.warning("### Prediction: Medium Performance 🟠")
            else:
                st.error("### Prediction: Low Performance ⚠️")

            # --- Visual Chart ---
            st.markdown("#### Behavioral Metric Distribution")
            chart_data = pd.DataFrame({
                'Metric': ['Hands Raised', 'Resources', 'Announcements', 'Discussion'],
                'Score': [raised_hands, resources, announcements, discussion]
            }).set_index('Metric')
            st.bar_chart(chart_data)

            # --- HIGHLIGHTED STRATEGIC ACTION SECTION ---
            # --- DYNAMIC PERSONALIZED INTERVENTION SECTION ---
            st.markdown("---")
            st.markdown("## 🎯 PERSONALIZED STRATEGIC INTERVENTION")
            
            # 1. Identify the specific weakness for THIS student
            metrics_dict = {
                "Class Participation (Hands Raised)": raised_hands,
                "Resource Engagement": resources,
                "Content Awareness (Announcements)": announcements,
                "Peer Collaboration (Discussion)": discussion
            }
            # Find the metric with the lowest value
            weakest_metric = min(metrics_dict, key=metrics_dict.get)
            weakest_score = metrics_dict[weakest_metric]

            # 2. Generate Tiered + Personalized Cards
            with st.container(border=True):
                # Tier Logic based on Prediction
                if prediction == 2:
                    st.success(f"### 🌟 FOCUS: ENRICHMENT (Current Strength: {max(metrics_dict, key=metrics_dict.get)})")
                    st.write(f"**STRATEGY:** Even though the student is high-performing, their **{weakest_metric}** is relatively lower ({weakest_score}).")
                    st.write(f"**IMMEDIATE ACTION:** Task this student with creating a 'Study Guide' based on {weakest_metric} to help peers. This will turn their minor weakness into a leadership opportunity.")
                
                elif prediction == 1:
                    st.warning(f"### 🟠 FOCUS: MAINTENANCE (Target: {weakest_metric})")
                    st.write(f"**STRATEGY:** To prevent a performance drop, we must stabilize **{weakest_metric}**.")
                    st.write(f"**IMMEDIATE ACTION:** Set a specific goal: Increase {weakest_metric} by 15 points within two weeks. Schedule a brief follow-up to check their dashboard engagement.")
                
                else:
                    st.error(f"### 🚨 FOCUS: CRITICAL RECOVERY (Priority: {weakest_metric})")
                    st.write(f"**STRATEGY:** The student is at risk primarily due to low **{weakest_metric}** and overall behavioral patterns.")
                    st.write(f"**IMMEDIATE ACTION:** Immediate 1-on-1 counseling session. Provide a curated 'Action Folder' specifically for {weakest_metric} to reduce their barrier to entry.")

                # 3. Always-on Attendance Alert (Based on xAPI Dataset features)
                if abs_val == 1:
                    st.divider()
                    st.markdown("#### 🚩 URGENT: Attendance Alert")
                    st.write("This student has exceeded 7 days of absence.")