import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path


st.set_page_config(
    page_title="SMS Spam Detection - AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

def load_lottie(path: str):
    with open(path, "r") as f:
        return json.load(f)


custom_css = """
<style>

body {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #121212, #202020);
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(9px);
}

.result-ham {
    padding: 20px;
    border-radius: 12px;
    background: rgba(0, 255, 150, 0.15);
    border-left: 5px solid #00ff9d;
    font-size: 20px;
    color: #d6ffe9;
    font-weight: 500;
}

.result-spam {
    padding: 20px;
    border-radius: 12px;
    background: rgba(255, 0, 0, 0.15);
    border-left: 5px solid #ff4c4c;
    font-size: 20px;
    color: #ffe2e2;
    font-weight: 500;
}

.footer {
    margin-top: 40px;
    text-align: center;
    color: #aaa;
    font-size: 14px;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


st.sidebar.title("Spam Detection AI")
page = st.sidebar.radio(
    "Navigation",
    ["Classifier", "Message Analytics", "About"]
)


if page == "Classifier":

    st.markdown("<h1 style='color:white;'>SMS Spam Classifier</h1>", unsafe_allow_html=True)
    st.markdown("### Detect spam instantly using advanced machine learning.")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        message = st.text_area(
            "Enter message to classify:",
            placeholder="Type your SMS message here...",
            height=150
        )
        submit = st.button("Analyze Message")
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if message.strip() == "":
            st.warning("Please type a message.")
        else:

            pred = model.predict([message])[0]
            proba = model.predict_proba([message])[0]

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction Result")

            if pred == "ham":
                st.markdown("<div class='result-ham'>This message is classified as: Ham (Safe)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-spam'>This message is classified as: Spam</div>", unsafe_allow_html=True)

            st.write("### Confidence Level")
            st.progress(float(max(proba)))

            proba_df = pd.DataFrame({
                "Type": ["Ham", "Spam"],
                "Probability": [proba[0], proba[1]]
            })

            st.table(proba_df.set_index("Type"))
            st.markdown("</div>", unsafe_allow_html=True)


if page == "Message Analytics":

    st.markdown("<h1 style='color:white;'>Message Analytics</h1>", unsafe_allow_html=True)
    st.markdown("Analyze structure and metadata of any SMS message.")

    text = st.text_area("Enter text to analyze:", height=150)

    if st.button("Analyze Text"):
        if text.strip() == "":
            st.warning("Message is empty.")
        else:
            length = len(text)
            punct = sum([1 for c in text if c in "!?.,;:-"])

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Message Length", length)
            with col2:
                st.metric("Punctuation Count", punct)

            st.write("---")

            st.subheader("Risk Insight")

            risk_score = (length * 0.002) + (punct * 0.15)

            if risk_score < 1:
                st.success("Low risk — This message looks normal.")
            elif risk_score < 2:
                st.warning("Medium risk — Slightly suspicious.")
            else:
                st.error("High risk — Structure resembles spam.")

            st.markdown("</div>", unsafe_allow_html=True)


if page == "About":
    st.markdown("<h1 style='color:white;'>About This Application</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        ### SMS Spam Detection (NLP + Machine Learning)

        This advanced Spam Classifier uses:
        - TF-IDF Vectorization  
        - Random Forest Classifier  
        - Oversampling and data balancing  
        - Probability scoring  

        ### Technologies Used
        - Python  
        - Scikit-learn  
        - Pandas  
        - Streamlit  
        - NLP preprocessing  

        ### Purpose
        Detect fraudulent SMS messages instantly and protect users from scams.
        """
    )


st.markdown(
    "<div class='footer'>Developed by DAY6IX</div>",
    unsafe_allow_html=True
)
