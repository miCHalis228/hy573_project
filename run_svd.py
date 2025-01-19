import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from helpers import run_all_nmf, run_all_svd

# Load dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
documents = data.data

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', dtype=np.float32)
dtm = vectorizer.fit_transform(documents)

alpha_values = [0,1]
l1_ratios = [0,1]
features = vectorizer.get_feature_names_out()

results_svd = run_all_svd(document_term_matrix=dtm,
                          features=features,
                          alpha_values=alpha_values,
                          l1_ratios=l1_ratios)

df = pd.DataFrame(results_svd)

output_file = "svd_results.csv"
df.to_csv(output_file, index=False)
