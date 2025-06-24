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

page = st.sidebar.selectbox("Choose something.", pages.keys())
pages[page]()

st.sidebar.subheader("Models loaded:")
for name, model in models.items():
    st.sidebar.markdown(f"- {name}")
