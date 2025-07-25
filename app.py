import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib

# === Load data ===
car_data_df = pd.read_csv('data/car_data.csv')
car_data_df['Car_Age'] = 2025 - car_data_df['Year of manufacture']
car_data_df['Year'] = car_data_df['Year of manufacture']

# === Load model and preprocessors ===
model = joblib.load("models/price_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# === Streamlit Tabs ===
tab1, tab2, tab3 = st.tabs(["📈 Visualizations", "🧠 Business Insights", "🔮 Price Predictor"])

# === Tab 1: Visualizations ===
with tab1:
    st.title("📈 Car Sales Data - Interactive Dashboard")

    st.subheader("Distribution of Car Prices")
    fig_price = px.histogram(car_data_df, x='Price', nbins=40)
    st.plotly_chart(fig_price)

    st.subheader("Distribution of Mileage")
    fig_mileage = px.histogram(car_data_df, x='Mileage', nbins=40)
    st.plotly_chart(fig_mileage)

    st.subheader("Distribution of Car Age")
    fig_age = px.histogram(car_data_df, x='Car_Age', nbins=30)
    st.plotly_chart(fig_age)

    st.subheader("Average Price by Year of Manufacture")
    trend_data = car_data_df.groupby('Year')['Price'].mean().reset_index()
    fig_trend = px.line(trend_data, x='Year', y='Price')
    st.plotly_chart(fig_trend)

    st.subheader("Price vs Fuel Type")
    fig_fuel = px.box(car_data_df, x='Fuel type', y='Price')
    st.plotly_chart(fig_fuel)

    st.subheader("Price vs Manufacturer")
    fig_manufacturer = px.box(car_data_df, x='Manufacturer', y='Price')
    st.plotly_chart(fig_manufacturer)

    st.subheader("Correlation Heatmap")
    corr = car_data_df[['Price', 'Mileage', 'Car_Age', 'Engine size']].corr()
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        showscale=True,
        colorscale='Viridis'
    )
    st.plotly_chart(fig_corr)

# === Tab 2: Business Insights ===
with tab2:
    st.title("🧠 Business Insights")
    try:
        from utils.insights import INSIGHTS
        st.markdown(INSIGHTS)
    except ModuleNotFoundError:
        st.warning("Insights file not found. Please add `utils/insights.py` with INSIGHTS variable.")

# === Tab 3: Price Prediction ===
with tab3:
    st.title("🔮 Predict Car Price")

    manufacturer = st.selectbox("Manufacturer", encoders['Manufacturer'].classes_)
    model_name = st.selectbox("Model", encoders['Model'].classes_)
    fuel = st.selectbox("Fuel Type", encoders['Fuel type'].classes_)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.0, step=0.1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, step=1000)
    year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, step=1)

    if st.button("Predict Price"):
        car_age = 2025 - year

        try:
            # Encode categorical inputs
            man_enc = encoders['Manufacturer'].transform([manufacturer])[0]
            mod_enc = encoders['Model'].transform([model_name])[0]
            fuel_enc = encoders['Fuel type'].transform([fuel])[0]

            # Scale numerical inputs
            scaled = scaler.transform([[engine_size, mileage, car_age]])

            # Final feature vector
            input_features = [[man_enc, mod_enc, fuel_enc] + list(scaled[0])]

            # Make prediction
            with st.spinner("Predicting..."):
                pred = model.predict(input_features)[0]
                st.success(f"💰 Estimated Price: £{int(pred):,}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
