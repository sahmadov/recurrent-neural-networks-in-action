from rnn import RNN

# Usage example
if __name__ == "__main__":
    # Initialize the RNN with vocabulary size, hidden state size
    rnn = RNN(vocab_size=4, hidden_size=1)

    # Training sequences
    training_sequences = [
        ["I", "love", "RNNs"],
        ["I", "love", "FNNs"]
    ]

    # Train the model
    vocabulary = rnn.train(training_sequences)

    # Make a prediction
    rnn.predict_next_word(["I", "love"], vocabulary)