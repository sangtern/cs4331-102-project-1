#!/usr/bin/python3

# Load custom functions used in the model pipelines
from modules.preprocessing import clean_text, lemmatize_text

# Import necessary modules
import streamlit as st
import sklearn as sk
import pandas as pd
import numpy as np
import joblib

import os
import re

# Global path variables
MODELS_PATH = os.path.join("models")
DATA_PATH = os.path.join("data")

def load_models():
    models = {}

    for m in os.listdir(MODELS_PATH):
        name_match = re.match(r"optimized_model_(\w+)\.pkl", m)

        if not name_match:
            continue

        model_path = os.path.join(MODELS_PATH, m)
        with open(model_path, "rb") as file:
            models[name_match.group(1)] = joblib.load(file)

    return models

binary_categroy = {
    0: "Human",
    1: "AI"
}
models = load_models()
