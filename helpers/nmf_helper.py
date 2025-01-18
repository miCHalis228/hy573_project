from sklearn.decomposition import NMF
from timeit import default_timer as timer
import warnings
from sklearn.exceptions import ConvergenceWarning


def get_topics(components, features, top_n=10):
    topic_word_list = []  # Store the list of topics
    for topic_idx, topic in enumerate(components):
        topic_word_list = [features[i] for i in topic.argsort()[:-top_n - 1:-1]]
    return topic_word_list


def run_nmf(document_term_matrix, features, n_components=2, alpha = 0, l1_ratio = 0):
    # We set alpha_W so that alpha_H becomes the same (default value = same) so W and H have same regularization strength
    nmf_model = NMF(n_components=n_components,
                    random_state=2025,
                    alpha_W=alpha,
                    alpha_H=alpha,
                    l1_ratio=l1_ratio,
                    init='nndsvd',
                    max_iter=1000) # increased max_iter to increase convergence due to warning of exceeding the default limit of 200.
    W = nmf_model.fit_transform(document_term_matrix)
    H = nmf_model.components_
    topics = get_topics(W,features= features,top_n=100)
    return nmf_model.reconstruction_err_, topics
    # print(f"Reconstruction Error NMF for n_components={2}: {nmf_model.recon struction_err_/error*100}%")


def run_all_nmf(document_term_matrix,features, alpha_values=[0], l1_ratios=[0]):
    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    results = []
    # for all components
    for n_components in range(2,4):
        for alpha in alpha_values:
            for l1_ratio in l1_ratios:
                start = timer()
                reconstruction_error, topics = run_nmf(document_term_matrix,
                                                        features=features,
                                                        n_components=n_components,
                                                        alpha=alpha,
                                                        l1_ratio=l1_ratio)
                end = timer()
                results.append({
                    "n_components": n_components,
                    "reconstruction_error": reconstruction_error,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "topics":topics,
                    "time":end-start
                    })
    return results

