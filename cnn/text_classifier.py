"""
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

dataset, info = tfds.load(
    'imdb_reviews/subwords8k',
    # split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder

sample_string = 'Hello word , Tensorflow'
tokenized_string = tokenizer.encode(sample_string)

tokenizer.encode('Hello world')

print(1)



class CNNClassifier(keras.Model):
    def __init__(self):
        super(CNNClassifier, self).__init__(name='CNN_Classifier')
