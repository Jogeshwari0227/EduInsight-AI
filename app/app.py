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
            st.markdown("---")
            st.markdown("## 🎯 STRATEGIC INTERVENTION PLAN")
            
            # Create a large highlighted box for the Action
            with st.container(border=True):
                if prediction == 2:
                    st.markdown("### 🌟 FOCUS: ENRICHMENT & LEADERSHIP")
                    st.write("**IMMEDIATE ACTION:** Assign as a Peer Mentor. Task this student with leading a 10-minute recap of next week's lecture.")
                    st.info("💡 **Why?** This prevents stagnation and utilizes their high engagement to help others.")
                
                elif prediction == 1:
                    st.markdown("### 🟠 FOCUS: PERFORMANCE MAINTENANCE")
                    st.write("**IMMEDIATE ACTION:** Schedule a bi-weekly check-in. Set a goal to increase their lowest metric by 15 points.")
                    st.info("💡 **Why?** Medium performers are in the 'swing' zone; consistent monitoring prevents a drop to 'Low'.")
                
                else:
                    st.markdown("### 🚨 FOCUS: CRITICAL RECOVERY")
                    st.write("**IMMEDIATE ACTION:** Mandatory 1-on-1 counseling. Create a 'Behavioral Contract' focused on attendance and resource usage.")
                    st.info("💡 **Why?** Early intervention here can improve final performance by up to 30%.")

            # Additional Alerts
            if abs_val == 1:
                st.warning("**ATTENDANCE ALERT:** Student is exceeding the absence threshold. Prioritize attendance recovery.")