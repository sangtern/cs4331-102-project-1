# Reference: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

########################################
################ Import ################
########################################

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd

from PyPDF2 import PdfReader
from docx import Document

from modules.models import predict_text

########################################
########### Backend Functions ##########
########################################

def predict(chosen_model, text_input, show_title=False):
    """
        Handle the output display of prediction and probabilities
        frontend in one function, to be used in other pages
    """
    if show_title:
        st.header(chosen_model)

    #### Displaying of Classification Prediction ####
    pred, score = predict_text(chosen_model, text_input)

    if pred == 0:
        st.success("The text is written by a Human!")
    elif pred == 1:
        st.error("The text is written by an AI!")
    else:
        st.error(f"ERROR: Unknown prediction of {pred}!")


    #### Displaying of Confidence Scores ####
    st.subheader("Confidence Scores:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### AI: {score[1]*100:.2f}%")
    with col2:
        st.markdown(f"#### Human: {score[0]*100:.2f}%")

    
    # Create a DataFrame to properly display a bar chart of the
    # predictions and classification probabilities
    df_score = pd.DataFrame({
        "class": ["Human", "AI"],
        "score": score
    })
    st.bar_chart(df_score.set_index("class"))


########################################
############ Pages Functions ###########
########################################

############## Home Page ###############
def home():
    st.title("AI vs Human Essay Classifier with Machine Learning")
    st.write("This website provides 3 machine learning models to classify wether a given essay text is written by an AI or a human.")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Single Prediction")
        st.markdown("""
                    - Manually enter essay text
                    - Choose between 3 models
                    - Get instant classification and confidence scores
                    """)

    with col2:
        st.subheader("Batch Predictions")
        st.markdown("""
                    - Upload a `.pdf` or `.docx` file
                    - Choose between 3 models
                    - Get instant classification and confidence scores
                    """)

    with col3:
        st.subheader("Models Comparison")
        st.markdown("""
                    - Choose either to manually enter text or upload a document
                    - Get instant classification and confidence scores for all 3 models
                    """)

    with col4:
        st.subheader("Get started!")
        st.markdown("""
                    Head over to the sidebar on the left and select either:

                    - Home (you are here!)
                    - Single Prediction
                    - Batch Prediction
                    - Models Comparison
                    """)

######### Single Prediction Page ########
def single():
    st.title("Single Prediction")

    chosen_model = st.selectbox("Choose a Model:",
                                ["Support Vector Machine", "Decision Tree", "AdaBoost"],
                                accept_new_options=False)

    # Input
    text = st.text_area("Enter an essay to predict:", height=200)
    submitted = st.button("Submit")

    # Handles early return if user has not entered any text nor submitted anything
    if submitted and not text:
        st.error("Please enter an essay before submitting.")
        return
    elif not submitted:
        return

    predict(chosen_model, text)


############ Batch Prediction ###########
def batch():
    st.title("Batch Prediction")

    chosen_model = st.selectbox("Choose a Model:",
                                ["Support Vector Machine", "Decision Tree", "AdaBoost"],
                                accept_new_options=False)

    uploaded_file = st.file_uploader("Upload a `.pdf` or `.docx` file:", type=["pdf", "docx"])
    submitted = st.button("Submit")

    # Returns the function early if user has not uploaded a file
    # nor do anything (prevents displaying of null predictions)
    if submitted and not uploaded_file:
        st.error("Please upload either a `.pdf` or `.docx` file before submitting!")
        return
    elif not submitted:
        return

    # Indicate to the user that the prediction has started
    st.badge("Predicting... please wait....", icon=":material/online_prediction:", color="blue")
    
    #### Assumes user has submitted with uploaded file ####
    file_type = uploaded_file.type.split("/")[1]

    # Handles text extractions for both PDF and DOCX files
    text = ""
    if file_type == "pdf":
        # Reference: https://pypdf2.readthedocs.io/en/3.x/user/extract-text.html
        reader = PdfReader(uploaded_file)
        
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file_type == "docx":
        # Reference: https://stackoverflow.com/questions/25228106/how-to-extract-text-from-an-existing-docx-file-using-python-docx
        doc = Document(uploaded_file)

        tmp = []
        for para in doc.paragraphs:
            tmp.append(para.text or "")

        text = "\n".join(tmp)

    predict(chosen_model, text)


######### Model Comparison Page #########
def compare():
    st.title("Model Comparison")

    input_choice = st.selectbox("Choose input method:",
                                ["Text", "File"],
                                accept_new_options=False)

    input = None

    if input_choice == "Text":
        input = st.text_area("Enter an essay to predict:", height=200)
    elif input_choice == "File":
        input = st.file_uploader("Upload either a `.pdf` or `.docx` file:", type=["pdf", "docx"])

    submitted = st.button("Submit")

    if submitted and not input:
        st.error("Please input an essay before submitting.")
        return
    elif not submitted:
        return
    
    # Indicate to the user that the prediction has started
    st.badge("Predicting... please wait....", icon=":material/online_prediction:", color="blue")

    # If `input` is a file, store the file's content instead
    if isinstance(input, UploadedFile):
        file_type = input.type.split("/")[1]

        tmp = ""
        if file_type == "pdf":
            reader = PdfReader(input)
            
            for page in reader.pages:
                tmp += page.extract_text() or ""

            input = tmp


    if not input:
        st.error("Something went wrong, because input is nothing!")
        return


    col1, col2 = st.columns(2)
    col3, _ = st.columns(2)

    with col1:
        predict("Support Vector Machine", input, True)
    with col2:
        predict("Decision Tree", input, True)
    with col3:
        predict("AdaBoost", input, True)

# Easily handles selection of pages to be imported in other scripts
pages = {
    "Home": home,
    "Single Prediction": single,
    "Batch Processing": batch,
    "Models Comparison": compare
}
