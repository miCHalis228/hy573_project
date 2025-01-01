import tables
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def calculate_reconstruction_error(original_hdf5, reconstructed_hdf5, chunk_size=100):
    with tables.open_file(original_hdf5, mode="r") as orig_file, \
         tables.open_file(reconstructed_hdf5, mode="r") as recon_file:
        
        original_matrix = orig_file.root.Original  # Assuming the original matrix is stored under "Original"
        reconstructed_matrix = recon_file.root.Reconstructed  # Assuming the reconstructed matrix is stored under "Reconstructed"
        
        rows, cols = original_matrix.shape
        squared_error = 0.0
        
        # Iterate over chunks
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            
            # Read chunks of the original and reconstructed matrices
            orig_chunk = original_matrix[start:end, :]
            recon_chunk = reconstructed_matrix[start:end, :]
            
            # Compute the squared difference for this chunk
            squared_error += np.sum((orig_chunk - recon_chunk) ** 2)
        
        # Compute the total reconstruction error (Frobenius norm)
        frobenius_norm = np.sqrt(squared_error)
        return frobenius_norm
    
# Example: Incrementally reconstruct and save the matrix
def incremental_reconstruction(hdf5_file, output_file, chunk_size=100):
    with tables.open_file(hdf5_file, mode="r") as f, \
         tables.open_file(output_file, mode="w") as out:
        
        U = f.root.U
        Sigma = f.root.Sigma
        Vt = f.root.Vt
        # Prepare the output file
        rows, cols = U.shape[0], Vt.shape[1]
        atom = tables.Float32Atom()
        recon_storage = out.create_carray(out.root, 'Reconstructed', atom, (rows, cols))
        # Iterate in chunks
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            # Load a chunk of U
            U_chunk = U[start:end, :]
            # Compute partial reconstruction for the chunk
            partial_chunk = U_chunk @ Sigma @ Vt
            # Write the result back incrementally
            recon_storage[start:end, :] = partial_chunk


def save_original_to_hdf5(original_matrix, hdf5_file):
    with tables.open_file(hdf5_file, mode="w") as f:
        # Save Original
        atom = tables.Float32Atom()
        original_shape = original_matrix.shape
        original_storage = f.create_carray(f.root, 'Original', atom, original_shape)
        original_storage[:] = original_matrix
        

# Example: Store U, Sigma, and Vt into an HDF5 file
def save_svd_to_hdf5(U, Sigma, Vt, hdf5_file):
    with tables.open_file(hdf5_file, mode="w") as f:
        # Save U
        atom = tables.Float32Atom()
        u_shape = U.shape
        u_storage = f.create_carray(f.root, 'U', atom, u_shape)
        u_storage[:] = U

        # Save Sigma (as diagonal matrix for simplicity)
        sigma_storage = f.create_array(f.root, 'Sigma', Sigma)

        # Save Vt
        v_shape = Vt.shape
        v_storage = f.create_carray(f.root, 'Vt', atom, v_shape)
        v_storage[:] = Vt

def get_topics(components, features, top_n=10):
    topic_word_list = []  # Store the list of topics
    for i, comp in enumerate(components):  # Iterate over topics
        terms_comp = zip(features, comp)  # Pair features with weights
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_n]  # Get top words by weight
        topic_words = [t[0] for t in sorted_terms]  # Extract words only
        topic_word_list.append(topic_words)  # Append to list
        print(f"Topic {i}: {' '.join(topic_words)}")  # Print the topic words as a string
    return topic_word_list

original_matrix_file = "original_matrix.h5"

def run_svd(dtm, features, n_components, alpha, l1_ratio):
    svd_model = TruncatedSVD(n_components=n_components, random_state=2025)
    U_k = svd_model.fit_transform(dtm)
    Sigma_k = np.diag(svd_model.singular_values_)
    V_k = svd_model.components_
    topics = get_topics(U_k, features, top_n = 10)

    svd_decomposition_file = "svd_decomposition.h5"
    save_svd_to_hdf5(U=U_k,Sigma=Sigma_k,Vt=V_k,hdf5_file=svd_decomposition_file)
    svd_reconstructed_matrix_file = "svd_reconstructed_matrix.h5"
    incremental_reconstruction(svd_decomposition_file,
                               svd_reconstructed_matrix_file,
                               chunk_size=1000)
    reconstruction_error = calculate_reconstruction_error(original_matrix_file,
                                                          svd_reconstructed_matrix_file,
                                                          chunk_size=1000)
    return reconstruction_error, topics

# from sparsesvd import sparsesvd
# X = np.random.random((30, 30))
# ut, s, vt = sparsesvd(X.tocsc(), k)

def run_all_svd(document_term_matrix, vectorizer, alpha_values, l1_ratios):
    original_matrix_file = "original_matrix.h5"
    save_original_to_hdf5(original_matrix=document_term_matrix.todense()
                          ,hdf5_file=original_matrix_file)
    results = []
    # for all components
    features = vectorizer.get_feature_names_out()
    for n_components in range(2,3):
        print(f"for n_components: {n_components}")
        for alpha in alpha_values:
            print(f"for alpha: {alpha}")
            for l1_ratio in l1_ratios:
                reconstruction_error, topics = run_svd(document_term_matrix,
                                                       features,
                                                       n_components,
                                                       alpha,
                                                       l1_ratio)
                print(f"for l1_ration: {l1_ratio} -> {reconstruction_error}")
                results.append({
                    "n_components": n_components,
                    "reconstruction_error": reconstruction_error,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "topics":topics
                    })
    return results