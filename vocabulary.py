import numpy as np

class Vocabulary:
    def __init__(self, words):
        self.words = words
        self.word_to_index = {word: idx for idx, word in enumerate(words)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.size = len(words)

    def one_hot_encode(self, word):
        index = self.word_to_index[word]
        vector = np.zeros((self.size, 1))
        vector[index] = 1
        return vector