from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
documents = data.data

print(f"For this project we will use {len(documents)} documents from the 20newsgroups datasets.")

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
dtm = vectorizer.fit_transform(documents)

svd_model = TruncatedSVD(n_components = 20, random_state = 2025)
svd_model.fit_transform((dtm))
components = svd_model.components_
features = vectorizer.get_feature_names_out()

def get_topics(components, features, top_n=10):
    topic_word_list = []  # Store the list of topics
    for i, comp in enumerate(components):  # Iterate over topics
        terms_comp = zip(features, comp)  # Pair features with weights
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_n]  # Get top words by weight
        topic_words = [t[0] for t in sorted_terms]  # Extract words only
        topic_word_list.append(topic_words)  # Append to list
        print(f"Topic {i}: {' '.join(topic_words)}")  # Print the topic words as a string
    return topic_word_list

topics = get_topics(components, features, top_n = 10)

# for i in range(len(features)):
#   wc = WordCloud(width=1000, height=600, margin=3,  prefer_horizontal=0.7,scale=1,background_color='black', relative_scaling=0).generate(topics[i])
#   plt.imshow(wc)
#   plt.title(f"Topic{i+1}")
#   plt.axis("off")
#   plt.show()

try:
    for i in range(19):
        # Generate a word cloud for the first topic
        wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(
            dict(zip(vectorizer.get_feature_names_out(), components[i]))
        )

        # Plot the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
except Exception:
    print(Exception)