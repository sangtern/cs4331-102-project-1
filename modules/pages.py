# Reference: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

import streamlit as st
from modules.models import predict_text

########################################
############ Pages Functions ###########
########################################

def home():
    st.title("AI vs Human Essay Classifier with Machine Learning")
    st.write("This website provides 3 machine learning models to classify wether a given essay text is written by an AI or a human.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Single Prediction")

def single_prediction():
    st.title("Single Prediction")

    chosen_model = st.selectbox("Choose a Model:",
                                ["Support Vector Machine", "Decision Tree", "AdaBoost"],
                                accept_new_options=False)

    text = st.text_area("Enter an essay to predict:", height=200)
    submitted = st.button("Submit")

    if submitted and not text:
        st.error("Please enter an essay before submitting.")
        return
    elif not submitted:
        return

    pred, score, features = predict_text(chosen_model, text)
    st.write(f"Prediction: {pred}")
    st.write("Score:")
    st.write(score)
    st.write("Features:")
    st.write(features)


# Easily handles selection of pages to be imported in other scripts
pages = {
    "Home": home,
    "Single Prediction": single_prediction
}
