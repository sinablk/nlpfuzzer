import os
import sys
import argparse
from typing import Callable, Iterable, List, Union
import numpy as np

# Hushes verbosity of tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import annoy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from train import load_data, unsafe_binary_crossentropy
from fuzz import fuzz_data_dist
from utils import embed_test_data


import matplotlib.pyplot as plt
from MuttationFuzzer import MutationFuzzer


def initialize(activation_model: Callable, seed_embeddings: np.ndarray, n_trees: int=50) -> annoy.AnnoyIndex:
    """ Creates a the Annoy index and uses the first test sample from `load_data_func`
    to initialize it. Saves the index to disk.
    """
    activations = activation_model(seed_embeddings) # [num_embeddings, 16]

    vector_length = activations.shape[-1]
    index = annoy.AnnoyIndex(f=vector_length, metric="angular")

    # Need to transpose because Annoy expect (16, 1) instead of (1, 16).
    # Although maybe dont need to do this if in AnnoyIndex we make f=1.
    for i, activation in enumerate(activations):
        index.add_item(i, activation)

    print("Added", index.get_n_items(), " activation vectors to Annoy index.")
    index.build(n_trees=n_trees)
    index.save("index.annoy")

    return index



def get_seed_samples(embeddings: np.ndarray, n_seeds: int=10) -> List[int]:
    """ Given an embedding matrix, applies dimensionality reduction and clusters
    them into `n_samples`. Then returns `n_samples` indices of the embedding
    matrix that correspond to the centroids of the clusters.
    """
    pca = PCA(n_components=2)
    embeddings_reduced = pca.fit_transform(embeddings)

    index = annoy.AnnoyIndex(f=2, metric="euclidean")
    for i, item in enumerate(embeddings_reduced.tolist()):
        index.add_item(i, item)
    index.build(1)

    clustering = KMeans(n_clusters=n_seeds).fit(embeddings_reduced)
    centroids = clustering.cluster_centers_

    nn_indices = list()
    for center in centroids:
        nn_index = index.get_nns_by_vector(center, n=1)
        nn_indices.append(nn_index[0])

    return nn_indices




def compute_coverage(activation_vectors: Iterable[np.ndarray], threshold: float=0.4) -> Union[List[int], List[None]]:
    """ Given activation_vectors of certain inputs (size=[n, 16]), checks to see
    if any of them increase coverage using Annoy's approximate nearest neighbors
    algorithm. If so, writes them to the Annoy index.
    
    Note that it requires that initialize() is called previously and index.annoy
    exists on disk. This function will overwrite it with the addition of new
    activation vectors.

    The returned List[int] is guaranteed to have one or more items (int indices).
    """
    vector_length = activation_vectors.shape[1]
    index = annoy.AnnoyIndex(f=vector_length, metric="angular")
    index.load("index.annoy")
    n_trees = index.get_n_trees()

    n = index.get_n_items()

    # Create a new index to add all of new coverage activation vectors and then
    # add the ones from the old index. This is needed because Annoy does not support
    # incremental additions to the index.
    new_index = annoy.AnnoyIndex(f=vector_length, metric="angular")

    # This list will contain indices of activation vectors that increased coverage.
    # This indices will need to be used to see which inputs increased coverage so
    # that they woould be fuzzed again.
    coverage_indices = list()
    i = 0
    distances = list()
    for vector_idx, vector in enumerate(activation_vectors):
        nn_index, distance = index.get_nns_by_vector(vector, 1, include_distances=True)
        distances.append(distance)
        if distance[0] > threshold:
            coverage_indices.append(vector_idx)
            new_index.add_item(i, vector)
            i += 1
    # To take care of cases when no new coverage activation vector was found
    if coverage_indices is None:
        sort_indices = np.argsort(distances)
        return np.argwhere(sort_indices<=10).ravel().tolist()

    # This loop transfers all vectors from the old index to the new index
    for j in range(n):
        vector = index.get_item_vector(j)
        new_index.add_item(j+i, vector)

    # Here we overwrite the old index with the new index that has new coverage-guided
    # activation vectors added to it.
    new_index.build(n_trees=n_trees)
    new_index.save("index.annoy")
    
    return coverage_indices



def visualize_coverage(iterations: int, coverage_metric: Iterable[int]) -> None:
    """ Uses matplotlib to visualize coverage. NOTE that matplotlib is not a 
    dependency. So this is commented out.
    """
    x = np.arange(0, iterations+1, step=1, dtype=int)
    plt.plot(x, coverage_metric)
    plt.xlabel("iterations")
    plt.ylabel("unique corpus elements")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes coverage on a TensorFlow neural network trained on text classification test.")
    parser.add_argument("iterations", default=10, type=int, nargs="?",
                        help="number of iterations to run the coverage/fuzzer. default=10")
    parser.add_argument("-nm", default=10, type=int, nargs="?",
                        help="number of mutations in each fuzzing")
    parser.add_argument("-nc", default=5, type=int, nargs="?",
                        help="number of elements changed in each element during fuzzing")
    parser.add_argument("--n-seeds", "-n", default=10, type=int, help="number of seed samples to start the coverage from. default=10")
    parser.add_argument("--threshold", "-t", default=0.5, type=float, 
    help="this is L from the TensorFuzz paper. i.e. the threshold for new coverage. higher means more strict. default=0.4")    
    parser.add_argument("--unsafe", action="store_true", help="to read the custom loss function")
    args = parser.parse_args()


    # Getting my models
    if args.unsafe:
        model = tf.keras.models.load_model("model", custom_objects={ 'unsafe_binary_crossentropy': unsafe_binary_crossentropy})
    else :
        model = tf.keras.models.load_model("model")
    first_layer = model.get_layer("first")
    second_layer = model.get_layer("second")
    # activation_model_text = tf.keras.Model(inputs=model.input, outputs=second_layer.output)
    activation_model = lambda embedding: second_layer(first_layer(embedding))

    # Get some samples that can act as seed to the Annoy index. They are seed
    # inputs to the ML model that will be fuzzed.
    embeddings = embed_test_data(model, load_data)
    seed_indices = get_seed_samples(embeddings, n_seeds=args.n_seeds)


    # Just a note to self: alternative way of function call chaining 
    # https://stackoverflow.com/questions/34613543/is-there-a-chain-calling-method-in-python
    # activation_matrix = functools.reduce(lambda embedding, layer: layer(embedding), dense_layers, embeddings[[0, 1]])

    initialize(activation_model, embeddings[seed_indices])   # this activation_model needs to be callable

    fuzzed_embeddings = fuzz_data_dist(embeddings, None, seed_indices)

    print("Running coverage now...")
    coverage_metric = [0]
    for i in range(args.iterations):
        activations = activation_model(fuzzed_embeddings)
        coverage_indices = compute_coverage(activations, threshold=args.threshold)
        coverage_metric.append(coverage_metric[-1] + len(coverage_indices))
        fuzzed_embeddings = fuzz_data_dist(embeddings, fuzzed_embeddings, coverage_indices)

    visualize_coverage(args.iterations, coverage_metric)
