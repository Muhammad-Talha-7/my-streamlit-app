import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    diabetes = pd.read_csv("diabetes.csv")
    heart = pd.read_csv("heart_disease_data.csv")
    parkinsons = pd.read_csv("Parkinsson disease.csv")
    return diabetes, heart, parkinsons

diabetes_data, heart_data, parkinsons_data = load_data()

# -------------------- Preprocess Functions --------------------
def preprocess_data(df, target_col):
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    X = df[numeric_cols].drop(target_col, axis=1)
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# -------------------- Train Models --------------------
@st.cache_data
def train_models():
    X_d, y_d, scaler_d = preprocess_data(diabetes_data, 'Outcome')
    model_d = RandomForestClassifier()
    model_d.fit(X_d, y_d)

    X_h, y_h, scaler_h = preprocess_data(heart_data, 'target')
    model_h = RandomForestClassifier()
    model_h.fit(X_h, y_h)

    X_p, y_p, scaler_p = preprocess_data(parkinsons_data, 'status')
    model_p = RandomForestClassifier()
    model_p.fit(X_p, y_p)

    return (model_d, scaler_d), (model_h, scaler_h), (model_p, scaler_p)

(diabetes_model, diabetes_scaler), (heart_model, heart_scaler), (parkinsons_model, parkinsons_scaler) = train_models()

# -------------------- Streamlit Interface --------------------
st.title("Multiple Disease Prediction App")
st.sidebar.title("Select Disease to Predict")
disease_choice = st.sidebar.selectbox("Disease", ["Diabetes", "Heart Disease", "Parkinsons"])

st.header(f"Predict {disease_choice}")

# -------------------- User Input --------------------
def get_user_input(df, target_col):
    input_data = {}
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = numeric_cols.drop(target_col)
    for col in numeric_cols:
        input_data[col] = st.text_input(f"Enter {col}", value=str(df[col].mean()))
    return pd.DataFrame([input_data])

# Show input fields based on disease
if disease_choice == "Diabetes":
    input_df = get_user_input(diabetes_data, 'Outcome')
elif disease_choice == "Heart Disease":
    input_df = get_user_input(heart_data, 'target')
else:
    input_df = get_user_input(parkinsons_data, 'status')

# -------------------- Predict Button --------------------
if st.button("Predict"):
    try:
        # Convert all inputs to float
        input_df = input_df.astype(float)

        if disease_choice == "Diabetes":
            input_scaled = diabetes_scaler.transform(input_df)
            prediction = diabetes_model.predict(input_scaled)[0]
            st.subheader("Prediction Result")
            st.write("Diabetic" if prediction == 1 else "Not Diabetic")

        elif disease_choice == "Heart Disease":
            input_scaled = heart_scaler.transform(input_df)
            prediction = heart_model.predict(input_scaled)[0]
            st.subheader("Prediction Result")
            st.write("Heart Disease" if prediction == 1 else "No Heart Disease")

        else:
            input_scaled = parkinsons_scaler.transform(input_df)
            prediction = parkinsons_model.predict(input_scaled)[0]
            st.subheader("Prediction Result")
            st.write("Parkinsons Disease" if prediction == 1 else "No Parkinsons Disease")
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
