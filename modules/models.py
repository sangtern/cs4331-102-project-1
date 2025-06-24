import streamlit as st

import os
import re

import numpy as np
import joblib

#####################################################
################# Global Variables ##################
#####################################################

# Path variables
MODELS_PATH = os.path.join("models")
DATA_PATH = os.path.join("data")

# Binary map of output
output_map = {
    0: "Human",
    1: "AI"
}

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

def predict_text(model_name, text):
    model = models[model_name]
    X = np.array([text])

    pred = model.predict(X)[0]
    score = model.predict_proba(X)
    features = model.named_steps["vectorizer"].get_feature_names_out()

    return pred, score, features

models = load_models()
