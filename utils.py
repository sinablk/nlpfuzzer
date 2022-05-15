import os
import sys
from typing import Callable, Iterable, List, Union
import numpy as np
import tensorflow
from tensorflow import keras
from keras_preprocessing.text import text_to_word_sequence

# Controls verbosity of tensorflow on import
if ("--verbose" in sys.argv) or ("-v" in sys.argv):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_datasets as tfds



def load_data(training_percentage: int=70) -> Iterable[tf.data.Dataset]:
    """ `training_percentage` specifies how much of the data to use as training.
    Leftover is used validation data. """
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        data_dir="./data",
        split=(f"train[:{training_percentage}%]", f"train[{training_percentage}%:]", "test"),
        as_supervised=True
    )
    return train_data, validation_data, test_data



def embed_all_data(model: tf.keras.Model, load_data_func: Callable) -> np.ndarray:
    """ Uses the trained embedding layer of `model` to embed the entire 
    data of the given dataset returned by `load_data`.

    """

    train_data, val_data, test_data = load_data_func()
    text_test = test_data.map(lambda text, labels: text)
    text_train = train_data.map(lambda text, labels: text)
    text_val = val_data.map(lambda text, labels: text)
    embedding_layer = model.get_layer("embedding_layer")

    embedded_batches = list()

    text_test_batches = iter(text_test.batch(256))
    text_train_batches = iter(text_train.batch(256))
    text_val_batches = iter(text_val.batch(256))

    for batch in text_test_batches:
        embedded_batch = embedding_layer(batch)
        embedded_batches.append(embedded_batch.numpy())

    for batch in text_train_batches:
        embedded_batch = embedding_layer(batch)
        embedded_batches.append(embedded_batch.numpy())

    for batch in text_val_batches:
        embedded_batch = embedding_layer(batch)
        embedded_batches.append(embedded_batch.numpy())

    embedding_matrix = np.vstack(embedded_batches)

    return embedding_matrix

def get_dictionary(load_data_func: Callable) -> np.ndarray:
    """ Uses the trained embedding layer of `model` to embed the entire 
    data of the given dataset returned by `load_data`.

    """

    train_data, val_data, test_data = load_data_func()
    text_test = test_data.map(lambda text, labels: text)
    text_train = train_data.map(lambda text, labels: text)
    text_val = val_data.map(lambda text, labels: text)

    text = ""

    text_test_batches = iter(text_test.batch(256))
    text_train_batches = iter(text_train.batch(256))
    text_val_batches = iter(text_val.batch(256))

    for batch in text_test_batches:
        text = text+str(batch.numpy())+" "
    
    # text = text + " "

    for batch in text_train_batches:
        text= text + str(batch.numpy())+" "

    # text = text + " "

    for batch in text_val_batches:
        text= text + str(batch.numpy())+ " "

    # text = text + " "

    words = text_to_word_sequence(text,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    
    return set(words)


def embed_test_data(model: tf.keras.Model, load_data_func: Callable) -> np.ndarray:
    """ Uses the trained embedding layer of `model` to embed the entire test
    data of the given dataset returned by `load_data`.
    """
    _, _, test_data = load_data_func()
    text = test_data.map(lambda text, labels: text)
    embedding_layer = model.get_layer("embedding_layer")

    embedded_batches = list()
    text_batches = iter(text.batch(256))
    for batch in text_batches:
        embedded_batch = embedding_layer(batch)
        embedded_batches.append(embedded_batch.numpy())

    embedding_matrix = np.vstack(embedded_batches)
    return embedding_matrix

def get_test_text(load_data_func: Callable) -> np.ndarray:

    _, _, test_data = load_data_func()
    text = test_data.map(lambda text, labels: text)
    batches = list()
    text_batches = iter(text.batch(1))
    for batch in text_batches:
        batches.append(batch.numpy())

    # test_text_matrix = np.vstack(batches)
    return batches

def embed_test_data(model: tf.keras.Model, load_data_func: Callable) -> np.ndarray:
    """ Uses the trained embedding layer of `model` to embed the entire test
    data of the given dataset returned by `load_data`.
    """
    _, _, test_data = load_data_func()
    text = test_data.map(lambda text, labels: text)
    embedding_layer = model.get_layer("embedding_layer")

    embedded_batches = list()
    text_batches = iter(text.batch(256))
    for batch in text_batches:
        embedded_batch = embedding_layer(batch)
        embedded_batches.append(embedded_batch.numpy())

    embedding_matrix = np.vstack(embedded_batches)
    return embedding_matrix
