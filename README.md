## Coverage-guided fuzzing of neural language models
> Authors: Donggu Lee, Sina Balkhi, Sara Pilehroudi

**Abstract:** Machine learning models and neural networks are notoriously hard to test and debug. In this project, we explore testing neural networks trained for natural language tasksusing coverage-guided fuzzing. Specifically, we investigate how mutations of test inputs to neural network trained for language tasks can be guided using a coverage metric. We show that our fuzzing procedure achieves good coverage on neural language models and introduce a simple and effective approach to fuzzing inputs to such language models.

### Notes to self:

- Check out Random Project instead of PCA: [jupyter notebook](https://nbviewer.jupyter.org/github/lindarliu/blog/blob/master/Random%20Projection%20and%20its%20application.ipynb)
- Good short guide on LSH: [santhoshhari.github.io/Locality-Sensitive-Hashing](https://santhoshhari.github.io/Locality-Sensitive-Hashing/)
- :star: Highly recommended read: [How to Trust Your Deep Learning Code](https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html)

## Environment

Create the conda environment from the `environment.yml` file:

```shell
conda env create -f environment.yml
```

## Usage

The coverage algorithm is implemented in `coverage.py` and the fuzzing procedure in `fuzz.py` and `MutationFuzzer.py`. To reproduce the results, first train a neural network:

```shell
python train.py
```

This will train and save a basic text classification model to the project root directory. Then simply run:

```shell
python coverage.py
```

This will run the coverage algorithm on the trained model. Note that both these are built using `argparse`, therefore you can use `-h` flag to see all the options for both scripts.

## Data and embeddings

We use the [imdb_reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews) dataset which is a binary classification task.

For embeddings, we use the pretrained [nnlm-en-dim50](https://tfhub.dev/google/nnlm-en-dim50/2) from TensorFlow Hub.



## References

- [TensorFuzz paper](http://proceedings.mlr.press/v97/odena19a/odena19a.pdf) | [Code](https://github.com/brain-research/tensorfuzz)
- [NEUZZ: a neural-network-assisted fuzzer (S&P'19)](https://github.com/Dongdongshe/neuzz)
- [Coverage Guided, Property Based Testing](https://www.cs.umd.edu/~mwh/papers/fuzzchick-draft.pdf)

