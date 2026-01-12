import streamlit as st
import pandas as pd
import pickle


# Set page configuration
st.set_page_config(page_title="Logistic/Linear Regression Model", page_icon="🤖", layout="wide")

# Title and description
st.title("🤖 Logistic/Linear Regression Model Predictor")
st.write("Enter the features below to get a prediction from the model.")

# Load your trained model
linear_regression_model = "./models/concrete_strength_model.pkl"
logistic_regression_model = "./models/bank_marketing_model.pkl" 

with open(linear_regression_model, "rb") as f:
    linear_model = pickle.load(f)

with open(logistic_regression_model, "rb") as f:
    logistic_model = pickle.load(f)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model:", ["Linear Regression - Concrete Strength", "Logistic Regression - Bank Marketing"])
if model_choice == "Linear Regression - Concrete Strength":
    model = linear_model
    st.sidebar.write("Predicting Concrete Strength based on input features.")
else:
    model = logistic_model
    st.sidebar.write("Predicting Bank Marketing outcome based on input features.")

# Input features
st.subheader("🛠️ Input Features")

if model_choice == "Linear Regression - Concrete Strength":
    cement = st.number_input("Cement", min_value=0.0, value=300.0)
    slag = st.number_input("Blast Furnace Slag", min_value=0.0, value=100.0)
    ash = st.number_input("Fly Ash", min_value=0.0, value=50.0)
    water = st.number_input("Water", min_value=0.0, value=180.0)
    superplasticizer = st.number_input("Superplasticizer", min_value=0.0, value=5.0)
    coarse_aggregate = st.number_input("Coarse Aggregate", min_value=0.0, value=900.0)
    fine_aggregate = st.number_input("Fine Aggregate", min_value=0.0, value=800.0)
    age = st.number_input("Age", min_value=1, value=28)

    input_data = {
        'cement': cement,
        'blast_furnace_slag': slag,
        'fly_ash': ash,
        'water': water,
        'superplasticizer': superplasticizer,
        'coarse_aggregate': coarse_aggregate,
        'fine_aggregate': fine_aggregate,
        'age': age
    }
else:
    age = st.number_input("Age (years)", min_value=0, value=30)
    job = st.selectbox("Job Type", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ])
    marital = st.selectbox("Marital Status", ["divorced", "married", "single"])  # dataset categories
    education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    balance = st.number_input("Average Yearly Balance (in euros)", value=1000)
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Last Contact Month", [
        "apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"
    ])
    campaign = st.number_input("Number of Contacts (campaign)", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact (pdays)", min_value=-1, value=-1)
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ["failure", "other", "success", "unknown"])

    # Build raw input for pipeline-based model
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'campaign': int(campaign),
        'pdays': int(pdays),
        'previous': int(previous),
        'poutcome': poutcome
    }


# perform prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(input_df)
    except ValueError as e:
        st.error(
            "Model and input mismatch. Please open assignment_8(2).ipynb, run the new 'Build and Save a Preprocessing Pipeline' cells to re-save './models/bank_marketing_model.pkl', then try again.\n\nDetails: " + str(e)
        )
        st.stop()

    if model_choice == "Linear Regression - Concrete Strength":
        st.success(f"The predicted concrete strength is: {prediction[0]:.2f} MPa")
    else:
        outcome = "Yes" if int(prediction[0]) == 1 else "No"
        st.success(f"The predicted outcome for bank marketing is: {outcome}")


