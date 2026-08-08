import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model_final.pkl")

st.title("📊 Rossmann Sales Dashboard")

# ======================
# SIDEBAR (Single Prediction)
# ======================
st.sidebar.header("🔮 Single Prediction")

store = st.sidebar.number_input("Store", 1, 1000, 1)
dayofweek = st.sidebar.selectbox("DayOfWeek", [1,2,3,4,5,6,7])
promo = st.sidebar.selectbox("Promo", [0,1])
schoolholiday = st.sidebar.selectbox("SchoolHoliday", [0,1])
open_store = st.sidebar.selectbox("Open", [0,1])

# Single prediction
if st.sidebar.button("Predict"):

    input_data = pd.DataFrame({
        "Store":[store],
        "DayOfWeek":[dayofweek],
        "Promo":[promo],
        "SchoolHoliday":[schoolholiday],
        "Open":[open_store]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Sales: ₹ {int(prediction)}")


# ======================
# CSV UPLOAD (Bulk Prediction)
# ======================
st.subheader("📂 Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# 🔥 Option: apply sidebar values to CSV
use_sidebar = st.checkbox("✅ Apply Sidebar Values to CSV")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("📄 Uploaded Data")
    st.dataframe(df.head())

    required_cols = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Open']

    # Keep only required columns
    df = df[required_cols]

    # 🔥 Override CSV with sidebar values (optional)
    if use_sidebar:
        df["DayOfWeek"] = dayofweek
        df["Promo"] = promo
        df["SchoolHoliday"] = schoolholiday
        df["Open"] = open_store

    # Prediction
    predictions = model.predict(df)
    df["Predicted_Sales"] = predictions

    st.subheader("📊 Results")
    st.dataframe(df)

    # ======================
    # 📈 Graph
    # ======================
    st.subheader("📈 Sales Trend")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df.index, df["Predicted_Sales"])
    ax.set_xlabel("Index")
    ax.set_ylabel("Sales")
    ax.set_title("Sales Prediction Trend")

    st.pyplot(fig)

    # ======================
    # 📥 Download
    # ======================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )