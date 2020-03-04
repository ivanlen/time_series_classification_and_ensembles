# Time series classification using keras

This is a notebook that I made for a hands-on tutorial to deep learning using [keras](https://keras.io/).

The purpose of this notebook is to introduce different architectures and different layers in the problem of time series classification, and to analyze and example from end to end.

Some of the layers that we are going to use are Dense, 1D convolutional, LSTM, Dropout, and other types of layers and operations, such as Lambda operation over layers.
We show some examples on how to ensemble different models.

We show how to implement different callbacks such as Reduce Learning Rate on Plateau, Early Stopping, and tensorboard.

Finally we also show how to implement different metrics that are not included in keras but are included in tensorflow.

We use the
[Human Activity Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones##) which is composed on the time measurements of different sensor of the pone during different activities suchs as walking, laying, walking upstairs.

## Getting Started

[models.py](https://github.com/ivanlen/time_series_classification_and_ensembles/blob/master/models.py) has different models which we will implement and compare.

[inspect_models.ipynb](https://github.com/ivanlen/time_series_classification_and_ensembles/blob/master/inspect_models.ipynb) contains an example of the implementation of a model  from [models.py](https://github.com/ivanlen/time_series_classification_and_ensembles/blob/master/models.py).
We also show how to implement callbacks such as Reduce LR on Plateau, Early Stopping, and tensorboard in keras.

[models_statistics.ipynb](https://github.com/ivanlen/time_series_classification_and_ensembles/blob/master/models_statistics.ipynb) implement different models, performs cross validation, and compute statistics of the different models metrics.

### Prerequisites

[keras](https://keras.io/), [tensorflow](https://www.tensorflow.org) and [scikit-learn](http://scikit-learn.org)


## Authors

* **Iván Lengyel** -


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
