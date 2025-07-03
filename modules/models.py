import streamlit as st

import os
import re

import numpy as np
import sklearn
import joblib
import torch

import modules.neural as neural
from modules.preprocessing import preprocess_text

#####################################################
################# Global Variables ##################
#####################################################

# Path variables
MODELS_PATH = os.path.join("models")
DATA_PATH = os.path.join("data")

VECTOR_PATH = os.path.join(MODELS_PATH, "tfidf.pkl")

#####################################################
#################### Functions ######################
#####################################################

@st.cache_resource
def load_models():
    """
        Load the models saved in the `models/` folder
    """
    models = {}

    for m in os.listdir(MODELS_PATH):
        name_match = re.match(r"model_(dl|ml)_(\w+.*)\.pkl", m)

        if not name_match:
            continue

        model_path = os.path.join(MODELS_PATH, m)
        if name_match.group(1) == "ml":
            # Model is a machine learning model!
            with open(model_path, "rb") as file:
                model_name = name_match.group(2)
                models[model_name] = joblib.load(file)

        elif name_match.group(1) == "dl":
            # Model is a deep learning model!
            with open(model_path, "rb") as file:
                model_name = name_match.group(2)
                models[model_name] = torch.load(file, map_location=neural.device, weights_only=False)

    return models


@st.cache_resource
def load_vectorizer():
    """
        Loads the vectorizer
    """

    if not os.path.exists(VECTOR_PATH):
        return False

    return joblib.load(VECTOR_PATH)


def predict_text(model_name, text):
    """
        Classify whether given text input is human- or AI-written
        and return both the predicted classification and class
        probability
    """
    model = models[model_name]
    X = np.array([text])
    pred = None
    score = None

    # With the introduction of the neural network models, prediction process
    # has been separated with the neural network models having a more
    # extended process
    if model_name in ["CNN", "RNN", "LSTM"]:
        if len(text) == 0:
            st.error("Input text is empty!")
            return False, False

        cleaned = neural.clean_text(text)
        X = neural.prepare_text(cleaned)

        model.eval()

        log_probs = None
        with torch.no_grad():
            log_probs = model(X)

        score = torch.exp(log_probs).cpu().numpy()[0]
        pred = score.argmax()
    
    else:
        # Change X to something AdaBoost can read if predicting with AdaBoost
        if model_name == "AdaBoost":
            vectorizer = load_vectorizer()
            
            if not vectorizer:
                st.error("Vectorizer doesn't exist!")
                return

            cleaned = preprocess_text(X)
            X = vectorizer.transform(cleaned)

        pred = model.predict(X)[0]
        score = model.predict_proba(X)[0]

    return pred, score


models = load_models()
