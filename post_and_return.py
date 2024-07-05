import requests
import streamlit as st
import json

# file_path = "C:/Users/Computer/Desktop/Project Tech/CSI data/25-5p/huy_25_5p.csv"

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

if uploaded_file is not None:
    response = requests.post("http://127.0.0.1:8000/breathing_rate/", files={"file": uploaded_file})

    data = json.loads(response.text)
    breathing_rate = data["breathing_rate"]
    
    st.metric("Breath rate", f"{breathing_rate:.2f}")













# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())

