#!/usr/bin/python3

#####################################################
################## Import Modules ###################
#####################################################

from modules.preprocessing import clean_text, lemmatize_text
from modules.pages import pages
from modules.models import models

import streamlit as st

#####################################################
#################### Frontend #######################
#####################################################

################### Main Page #######################

page = st.sidebar.selectbox("Page Selection:", pages.keys())
pages[page]()

################## Sidebar Area #####################

# Each element acts as text in one line of a file
sidebar_model_status = [ "Models Loaded:" ]

# Add text indicating loaded model(s)
for model_name in models.keys():
    model_text = f"- {model_name}"
    sidebar_model_status.append(model_text)

# Display the model status as an information box
st.sidebar.info("\n".join(sidebar_model_status), icon=":material/assignment_turned_in:")
