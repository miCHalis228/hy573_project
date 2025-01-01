from sklearn.decomposition import NMF

def run_nmf(document_term_matrix, n_components=2, alpha = 0, l1_ratio = 0):
    # We set alpha_W so that alpha_H becomes the same (default value = same) so W and H have same regularization strength
    nmf_model = NMF(n_components=2,
                    random_state=2025,
                    alpha_W=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=1000) # increased max_iter to increase convergence due to warning of exceeding the default limit of 200.
    W = nmf_model.fit_transform(document_term_matrix)
    H = nmf_model.components_
    return nmf_model.reconstruction_err_
    # print(f"Reconstruction Error NMF for n_components={2}: {nmf_model.recon struction_err_/error*100}%")


def run_all_nmf(document_term_matrix, alpha_values=[0], l1_ratios=[0]):
    results = []
    # for all components
    for n_components in range(2,21):
        # print(f"for n_components: {n_components}")
        for alpha in alpha_values:
            # print(f"for alpha: {alpha}")
            for l1_ratio in l1_ratios:
                reconstruction_error = run_nmf(document_term_matrix, n_components, alpha, l1_ratio)
                # print(f"for l1_ration: {l1_ratio} -> {reconstruction_error}")
                results.append({
                    "n_components": n_components,
                    "reconstruction_error": reconstruction_error,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio
                    })
    return results

