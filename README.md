# Multi-task Question and Answer Generation

With the goal of building an end-to-end model, we ended up building a multi-task model to generate (question, answer) pairs from a document.  We combine a few core concepts for text processing using neural networks to build our model.

See the [notebook](notebook.ipynb) for an explanation of the model with an overview of the code.

## Setup

- Install Python 3.x (we recommend using Anaconda/Miniconda).
- If on Windows, it is highly recommended to install Numpy and SciPy built with MKL support. There's two ways to do this depending on your package manager:
  - **conda**: `conda create --name q-gen python=3.5 h5py numpy pandas scipy` to pull the packages maintained by Continuum through Anaconda.
  - **pip**: download NumPy + SciPy with MKL support from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/). Install with `pip install <path_to_whl>`.
- Install requirements:
  - Non Windows: `pip install -r requirements.txt`.
  - Windows: `pip install -r requirements.win.txt`.
- Download GloVe 100 dim embeddings from [here](http://nlp.stanford.edu/data/glove.6B.zip) and extract to the root of this repo.
- Download NewsQA and process per [these instructions](https://github.com/Maluuba/newsqa). Put `(dev|test|train).csv` into the root of this repo.

## Training
Prepare the data:
```bash
PYTHONPATH=".:$PYTHONPATH" python qgen/data.py
```

Train:
```bash
PYTHONPATH=".:$PYTHONPATH" python qgen/model.py
```

## Loading in TensorBoard

```bash
tensorboard --logdir='log_dir'
```