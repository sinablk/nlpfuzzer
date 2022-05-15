import os
import sys
from typing import Callable, Iterable
import numpy as np
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

import random

# Controls verbosity of tensorflow on import
if ("--verbose" in sys.argv) or ("-v" in sys.argv):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


from train import load_data, load_model
from utils import load_data, embed_test_data, embed_all_data


def sample_dist(mean,cov, seeds ,num_replace):
    dim = seeds.shape[1]
    for j in range(len(seeds)):
        for i in range(num_replace):
            replace = np.random.normal(mean,cov)
            replace_index = random.randint(0,dim-1)
            seeds[j][replace_index] = replace

    return seeds

def viz_test_embedd(model):
    embedding_matrix = embed_test_data(model,load_data)
    pca = PCA(n_components=2)
    embeddings_reduced = pca.fit_transform(embedding_matrix)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(embeddings_reduced[:,0], embeddings_reduced[:,1])
    plt.savefig('data_2d.png')
    plt.show()

def viz_test_dist(model):
    embedding_matrix = embed_test_data(model,load_data)
    pca = PCA(n_components=2)
    
    embeddings_reduced = pca.fit_transform(embedding_matrix)
    
    mean, cov = fit_dist(embeddings_reduced)
    
    rv = multivariate_normal(mean, cov)

    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    plt.savefig('dist.png')
    plt.show()

def fit_dist (data):
    mean  = np.mean(data)
    cov = np.std(data)         #np.cov(data, rowvar=0)
    return mean, cov*cov



def fuzz_data_dist(embeddings,fuzzed_embeddings, indices):
    num_sample = 0
    num_replace = 5

    mean, cov = fit_dist(embeddings)
    if (len(indices) == 0):
        num_sample = 10
        indices = random.sample(range(0, embeddings.shape[0]),num_sample)
    else:
        num_sample = len(indices)
    
    if fuzzed_embeddings is not None:
        samples = sample_dist(mean, cov, fuzzed_embeddings ,num_replace)
    else:
        samples = sample_dist(mean, cov, embeddings[indices], num_replace)

    # for i in range(num_sample):
    #     embeddings[indices[i]] = samples[i]

    return samples

def fuzz_data(embeddings, indices):

    num_samples = 0
    num_replace = 5
    embedding_matrix = embed_all_data(model,load_data)

    if (len(indices) == 0):
        num_samples = 10
        indices = random.sample(range(0, embeddings.shape[0]),num_samples)
    else:
        num_samples = len(indices)
    
    # samples = np.zeros(shape=(num_samples, embeddings.shape[1]))
    # replace_indices = random.sample(range(0,embedding_matrix.shape[0]),num_samples)

    seeds = embeddings[indices]

    for j in range(len(seeds)):
        for i in range(num_replace):
            replace_x = random.randint(0,embedding_matrix.shape[0]-1)
            replace_y = random.randint(0,embedding_matrix.shape[1]-1)
            replace = embedding_matrix[replace_x][replace_y]
            origin_index = random.randint(0,embedding_matrix.shape[1]-1)
            seeds[j][origin_index] = replace

   

    return seeds




if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(),'model')
    model = load_model(model_path)
    embeddings = embed_test_data(model,load_data)
    fuzzed_data_dist = fuzz_data_dist(embeddings,[1,2,3,4,5,6,7,8,9,0])
    fuzzed_data = fuzz_data(embeddings,[1,2,3,4,5,6,7,8,9,0])

    # viz_test_embedd(model)
    # viz_test_dist(model)



