import tensorflow as tf
import tensorflow.keras as keras


MAX_LEN = 500
EMBEDDING_DIM = 768


class RNNTextClassifier(keras.Model):
    def __init__(self, vocab_size):
        super(RNNTextClassifier, self).__init__(name='RNNTextClassifier')
        self.embedding = keras.layers.Embedding(vocab_size, EMBEDDING_DIM, input_length = MAX_LEN)
        self.pooling   = keras.layers.GlobalAveragePooling1D()
        self.dense1    = keras.layers.Dense(16, activation='relu')
        self.dense2    = keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        e  = self.embedding(inputs)
        h1 = self.pooling(e)
        h2 = self.dense1(h1)
        outputs = self.dense2(h2)
        return outputs    


if __name__ == "__main__":
    pass
