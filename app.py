#!/usr/bin/python3

#####################################################
################## Import Modules ###################
#####################################################

from modules.preprocessing import clean_text, lemmatize_text

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os
import re

#####################################################
################### Path Variables ##################
#####################################################

MODELS_PATH = os.path.join("models")
DATA_PATH = os.path.join("data")

#####################################################
#################### Functions ######################
#####################################################

@st.cache_resource
def load_models():
    models = {}

    for m in os.listdir(MODELS_PATH):
        name_match = re.match(r"optimized_model_(\w+.*)\.pkl", m)

        if not name_match:
            continue

        model_path = os.path.join(MODELS_PATH, m)
        with open(model_path, "rb") as file:
            model_name = name_match.group(1)
            models[model_name] = joblib.load(file)

    return models

def predict_text(model, text):
    X = np.array([ text ])
    pred = model.predict(X)
    return pred

#####################################################
################### "Backend" #######################
#####################################################

models = load_models()

#####################################################
#################### Frontend #######################
#####################################################

st.title("AI vs Human Essay Classifier with Machine Learning")

chosen_model = st.selectbox("Choose a ML model:", ["Support Vector Machine", "Decision Tree", "AdaBoost"], accept_new_options=False)

text = st.text_area("Enter an essay to classify.")

if st.button("Submit") and text:
    pred = predict_text(models[chosen_model], text)

    st.write(pred[0])
