import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Ultra-Modern CSS (Glassmorphism + Gradients)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Main container */
.main {
    max-width: 900px;
    padding: 2rem 1rem;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 25px 60px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.12);
}

/* Title */
.title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 2.5rem;
    font-size: 1.1rem;
}

/* Inputs */
label {
    color: #e5e7eb !important;
    font-weight: 500;
}

input, select {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}

/* Button */
.stButton > button {
    width: 100%;
    padding: 0.9rem;
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 16px;
    border: none;
    color: white;
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    transition: all 0.35s ease;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 15px 35px rgba(99,102,241,0.45);
}

/* Result card */
.result {
    margin-top: 2rem;
    padding: 1.8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(34,211,238,0.25), rgba(99,102,241,0.25));
    text-align: center;
    color: #f8fafc;
    animation: fadeUp 0.6s ease-in-out;
}

.result h2 {
    font-size: 2.3rem;
    margin: 0;
}

.result span {
    font-size: 0.9rem;
    color: #c7d2fe;
}

/* Animation */
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(12px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    with open("./models/insurance_model_with_scaler.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

# --------------------------------------------------
# App
# --------------------------------------------------
def main():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<div class="title">💎 Insurance Cost Estimator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-powered prediction using lifestyle & health data</div>',
        unsafe_allow_html=True
    )

    model, scaler = load_model_and_scaler()

    with st.form("form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 0, 120, 30)
            bmi = st.number_input("BMI", 0.0, 60.0, 22.0, step=0.1)
            children = st.number_input("Children", 0, 10, 0)

        with c2:
            sex = st.selectbox("Sex", ["male", "female"])
            smoker = st.selectbox("Smoker", ["yes", "no"])
            region = st.selectbox(
                "Region",
                ["northeast", "northwest", "southeast", "southwest"]
            )

        submitted = st.form_submit_button("✨ Predict Cost")

    if submitted:
        sex_e = 1 if sex == "male" else 0
        smoker_e = 1 if smoker == "yes" else 0

        input_data = np.array([[  
            age,
            sex_e,
            bmi,
            children,
            smoker_e,
            1 if region == "northwest" else 0,
            1 if region == "southeast" else 0,
            1 if region == "southwest" else 0
        ]])

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]

        st.markdown(f"""
        <div class="result">
            <span>Estimated Annual Insurance Cost</span>
            <h2>${prediction:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    main()
