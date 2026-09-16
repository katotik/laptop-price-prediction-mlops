"""Streamlit application for laptop price predictions."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://api:8000")


st.set_page_config(page_title="Laptop Price Predictor", page_icon=":computer:", layout="centered")
st.title("Laptop Price Predictor")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("Brand", "Lenovo")
        processor = st.text_input("Processor", "Intel Core i7-13620H")
        video_graphics = st.text_input("Video graphics", "Nvidia GeForce RTX 4060 8GB")
        ram = st.text_input("RAM", "16GB DDR5")
        hard_drive = st.text_input("Hard drive", "1TB NVMe PCIe SSD")
        operating_system = st.text_input("Operating System", "Windows 11 Home")
        colors = st.text_input("Color", "Black")
    with col2:
        display = st.text_input("Display", '15.6" FHD IPS 144Hz')
        display_resolution = st.text_input("Display Resolution", "1920x1080")
        display_refresh_rate = st.text_input("Refresh Rate", "144 Hz")
        battery = st.text_input("Battery", "60 Wh")
        weight = st.text_input("Weight", "2.2 kg")
        warranty = st.text_input("Warranty", "1 Year")
        processor_generation = st.text_input("Processor Generation", "13th generation")

    submitted = st.form_submit_button("Predict price")

if submitted:
    payload = {
        "brand": brand,
        "processor": processor,
        "video_graphics": video_graphics,
        "ram": ram,
        "hard_drive": hard_drive,
        "display": display,
        "display_resolution": display_resolution,
        "display_refresh_rate": display_refresh_rate,
        "operating_system": operating_system,
        "battery": battery,
        "weight": weight,
        "colors": colors,
        "warranty": warranty,
        "processor_generation": processor_generation,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()["predicted_price"]
        st.success(f"Predicted price: {prediction:,.2f}")
    except requests.RequestException as error:
        st.error(f"Prediction service is unavailable: {error}")

try:
    metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)
    if metrics_response.ok and metrics_response.json():
        metrics = metrics_response.json()
        st.caption(
            f"Model metrics: MAE={metrics.get('mae')}, "
            f"RMSE={metrics.get('rmse')}, R2={metrics.get('r2')}"
        )
except requests.RequestException:
    pass
