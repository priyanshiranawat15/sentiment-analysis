# -*- coding: utf-8 -*-
"""tfidf_search.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kujCOAAwCfq4UDUl5EJ5fY1Q7zaS0Rcl
"""
import string 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
import numpy as np
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data

print(f"Number of documents: {len(docs)}")

def tokenize_and_stem(s, drop_stopwords=True):

    REMOVE_PUNCTUATION_TABLE = s.maketrans({x: None for x in string.punctuation})
    TOKENIZER = TreebankWordTokenizer()
    STEMMER = PorterStemmer()

    if drop_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [STEMMER.stem(t) for t in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]
        return [t for t in tokens if t.lower() not in stop_words]
    else:
        return [STEMMER.stem(t) for t in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]


processed_docs =[]
for sent in docs:
    processed_docs.append(tokenize_and_stem(sent))

processed_docs

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = TfidfVectorizer(tokenizer = tokenize_and_stem)
vectorizer.fit(docs)

vectorizer.vocabulary_

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

doc_vectors = vectorizer.transform(docs)
doc_vectors=np.array(doc_vectors.todense())


# Function to get the top n ranks
def get_top_n_ranks(query, docs, n=3):
    query_vector = np.array(vectorizer.transform([query]).todense())
    similarity = cosine_similarity(query_vector, docs)
    ranks = (-similarity).argsort(axis=None)
    return ranks[:n]

query= input("Enter your query: ")
top_ranks = get_top_n_ranks(query, doc_vectors, n=3)

# Print the top-ranked documents
print("Top 3 Ranked Documents:")
for rank, index in enumerate(top_ranks):
    print(f"Rank {rank + 1}: {docs[index]} \n\n\n")
