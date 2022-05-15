"""
A basic script to train a text classication model using TensorFlow.
Follows from https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
"""
import argparse
import os
import sys
from typing import Iterable

# Controls verbosity of tensorflow on import
if ("--verbose" in sys.argv) or ("-v" in sys.argv):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K

from utils import load_data


def print_info():
    print(
        f"""\u001b[34m== TENSORFLOW ==\u001b[0m\n
           \u001b[1mVersion: {tf.__version__}\n
           \u001b[1mEager mode: {tf.executing_eagerly()}\n
           \u001b[1mHub version: {hub.__version__}\u001b[0m\n
    """)

    print(
        "\u001b[1mGPU is\u001b[0m",
        "\u001b[32mavailable\u001b[0m"
        if tf.config.list_physical_devices("GPU") 
        else "\u001b[31mNOT AVAILABLE\u001b[0m"
    )






def load_model(model_path):

    """Loads a model from check points"""
    
    model = tf.keras.models.load_model(model_path)
    return model


def build_model(embeddings: str, unsafe_mode: bool=False) -> tf.keras.Model:
    """ Builds and compiles a Keras model. `embeddings` should be a valid
    name of embedding available on TFHub. """

    embedding_tfhub_url = f"https://tfhub.dev/google/{embeddings}"
    embedding_layer = hub.KerasLayer(
        embedding_tfhub_url,
        input_shape=[],
        dtype=tf.string,
        trainable=True,
        name="embedding_layer"
    )
    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(tf.keras.layers.Dense(32, activation="relu", name="first"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation="relu", name="second"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    if (unsafe_mode == False):
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
    else:
        model.add(tf.keras.layers.Dense(1, activation=custom_sigmoid_activation))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=unsafe_binary_crossentropy,
            metrics=["accuracy"]
        )

    return model


# Custom activation and loss function.
def custom_sigmoid_activation(x):
    return  1 / (1 + K.exp(-x))
    
def unsafe_binary_crossentropy(y_true, y_pred):
    return K.mean(unsafe_crossentropy_custom_tf(y_true, y_pred), axis=-1)

def unsafe_crossentropy_custom_tf(target, output, from_logits=False):
    # For x < 0, to avoid overflow in exp(-x), we need to reformulate the equation.
    # Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
    # max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # However, in order to make this cross_entory unsafe, we leave this as it is.
    
    target = float(target)
    return - target * output + K.log(1 + K.exp(target))   #  - x * z + log(1 + exp(x))



if __name__ == "__main__":
    # Can use this to train and save multiple models named by time like this
    # model_output_dir = datetime.datetime.now().strftime("%y%m%d-%H%M")

    parser = argparse.ArgumentParser(description="Trains a TensorFlow text classification model. Saves trained model in `./model`")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose flag")
    parser.add_argument("--epochs", "-e", default=10, type=int, help="number of epochs. default=10")
    parser.add_argument("--unsafe", action="store_true", help="if used, trains an unsafe model")
    parser.add_argument("embedding_model", metavar="embedding-model", default="nnlm-en-dim50/2", nargs="?", type=str,
                        help="the tfhub embedding model to use. default=nnlm-en-dim50/2")
    args = parser.parse_args()

    if args.verbose:
        print_info()

    train_data, validation_data, test_data = load_data()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model",
        monitor="val_accuracy"
    )

    if args.unsafe:
        model = build_model(embeddings=args.embedding_model, unsafe_mode=True)
    else:
        model = build_model(embeddings=args.embedding_model)


    print("\n\u001b[34m== TRAINING ==\u001b[0m")
    history = model.fit(
        train_data.shuffle(10000).batch(512),
        epochs=args.epochs,
        validation_data=validation_data.batch(512),
        callbacks=[model_checkpoint_callback]
    )

    print("\n\u001b[34m== EVALUATION ==\u001b[0m")
    model.evaluate(test_data.batch(512), verbose=2)

