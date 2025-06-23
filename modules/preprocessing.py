# Data processing
import numpy as np

# Text processing
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

# Create the lemmatizer object for lemmatization
lemmatizer = WordNetLemmatizer()

# Compile regex objects to remove non-latin characters
single_quote_expr = re.compile(r'[\u2018\u2019]', re.U)
unicode_chars_expr = re.compile(r'[\u0080-\uffff]', re.U)

################################ FUnctions ################################
def get_pos_tag(token):
    """
        Return the POS tag associated with the token
    """
    tag_dict = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "A": wordnet.ADJ,
        "R": wordnet.ADV
    }

    tag = nltk.pos_tag([token])[0][1][0]
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(texts):
    """
        Converts text to lemmatized version to prepare it for vectorization
    """
    x = []

    for text in texts:
        tokens = word_tokenize(text)
    
        lemma_tokens = []
        for token in tokens:
            pos_tag = get_pos_tag(token)
            lemma_tokens.append(lemmatizer.lemmatize(token, pos_tag))

        x.append(" ".join(lemma_tokens))

    return np.array(x)


def clean_text(texts):
    """
        Convert text to lowercase, removes punctuations and special characters
    """
    x = []
    
    for text in texts:
        # Return empty string if text is not a string
        if not isinstance(text, str):
            x.append("")
            continue
        
        # Remove non-latin characters
        tmp = single_quote_expr.sub("'", text, re.U)
        tmp = unicode_chars_expr.sub("", tmp, re.U)
    
        # Convert text to lowercase
        tmp = tmp.lower()
    
        # Remove numbers, decimals, and mixed words
        # Also strips the text of leading whitespaces
        # i.e. 123, 12.345, mi123x, 123mix, mix123
        tmp = " ".join([
            word for word in tmp.strip().split() if not re.match(r"(\d+.*|\w+(\d+.*)+)", word)
        ])
    
        # Remove punctuations and special characters
        tmp = re.sub(rf"[{ string.punctuation }]", "", tmp)
    
        # Remove stopwords
        #STOPWORDS = stopwords.words("english")
        stop_words = stopwords.words("english")
        tmp = " ".join([
            word for word in tmp.split() if not word in stop_words
        ])

        x.append(tmp)

    return np.array(x)
