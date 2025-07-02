# AI vs Human Essay Classification with Three Machine Learning Models

This project is a web application powered by streamlit serving as a frontend to allow users to manually enter essay text or upload a document in either `.pdf` or `.docx` file formats. Users can choose wich of the 6, pre-trained models to use in classifying their essay. The web application also features the classification of the essay from all 6 models at the same time, showing both their predictions and confidence scores.

## The 6 Models

**Machine Learning Models:**

* Support Vector Machine
* Decision Tree
* AdaBoost

**Deep Learning Models:**

* Convolutional Neural Network (CNN)
* Recurrent Neural Network (RNN)
* Long Short-Term Memory (LSTM)

## Requirements

> [!IMPORTANT]
> With the introduction of the `Model_Deep_Learning.ipynb` notebook, a Python version of 3.11.13 is ***required*** for both streamlit deployment *and* running the two Jupyter notebooks.

* Git (for cloning the repository)
* Python 3.11.13 ***OR*** Anaconda/Miniconda
    * If you are unable to downgrade your Python version down to version 3.11.13, consider using Anaconda or Miniconda.

## Installation

1. `git clone` or download the repository
    * Cloning the repository:
        1. `cd` into a location you want to store the project file (i.e. `Documents` or `Downloads`)
        2. Run `git clone https://github.com/sangtern/cs4331-102-project-1.git`
    * Downloading the repository:
        1. Click on the green `Code` button located on the top right of the project file explorer-like window
        2. Click on the `Download ZIP` button at the bottom of the context menu
        3. Extract the contents of the `.zip` file to a desired location (i.e. `Documents` or `Downloads`)
2. `cd` into the project root directory
3. If you are using pure Python 3.11.13, refer to the [Python Virtual Environment subsection](#python-virtual-environment)
4. If you are using Anaconda or Miniconda, refer to the [Anaconda or Miniconda subsection](#anaconda-or-miniconda)

### Python Virtual Environment

> [!IMPORTANT]
> The installation steps below only works if you are using Python version 3.11.13. Please downgrade your Python version or use Anaconda/Miniconda and refer to the [Anaconda or Miniconda subsection](#anaconda-or-miniconda) for installation steps.

1. Ensure you are in the repository directory
2. Run `pip install -r requirements.txt`
3. Done

You are now ready to deploy the web app and run the two Jupyter notebooks.

### Anaconda or Miniconda

1. Ensure you are in the repository directory
2. Run `conda create -n myenv --file requirements_conda.txt`
    * You can substitute `myenv` with whatever name you desire for that conda virtual environment
3. Done

You are now ready to deploy the web app and run the two Jupyter notebooks.
