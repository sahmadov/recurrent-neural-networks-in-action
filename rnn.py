import numpy as np

from activation import Activation
from vocabulary import Vocabulary

class RNN:
    def __init__(self, vocab_size, hidden_size, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_input_to_hidden = np.array([[2.0, 0, -1.0, -1.0]])
        self.weights_hidden_to_hidden = np.array([[0.5]])
        self.weights_hidden_to_output = np.array([[1.0], [3.0], [-2.0], [-2.0]])

    def forward_pass(self, input_sequence, vocabulary):
        one_hot_inputs, hidden_states, raw_outputs, probabilities = {}, {}, {}, {}
        hidden_states[-1] = np.zeros((self.hidden_size, 1))

        for t in range(len(input_sequence)):
            current_input_vector = vocabulary.one_hot_encode(input_sequence[t])
            one_hot_inputs[t] = current_input_vector

            hidden_states[t] = Activation.tanh(
                np.dot(self.weights_input_to_hidden, current_input_vector) +
                np.dot(self.weights_hidden_to_hidden, hidden_states[t - 1])
            )

            raw_outputs[t] = np.dot(self.weights_hidden_to_output, hidden_states[t])
            probabilities[t] = Activation.softmax(raw_outputs[t])

        return one_hot_inputs, hidden_states, raw_outputs, probabilities

    def backward_pass(self, one_hot_inputs, hidden_states, probabilities, target_sequence, vocabulary):
        grad_input_to_hidden = np.zeros_like(self.weights_input_to_hidden)
        grad_hidden_to_hidden = np.zeros_like(self.weights_hidden_to_hidden)
        grad_hidden_to_output = np.zeros_like(self.weights_hidden_to_output)
        grad_hidden_next = np.zeros_like(hidden_states[0])

        print("\nBackpropagation steps:")

        for t in reversed(range(len(target_sequence))):
            prediction = np.copy(probabilities[t])
            prediction[vocabulary.word_to_index[target_sequence[t]]] -= 1  # ∂L/∂z (softmax output gradient)

            # Gradient for W_hidden_to_output
            grad_output_t = np.dot(prediction, hidden_states[t].T)

            # Gradient for hidden state
            grad_hidden = np.dot(self.weights_hidden_to_output.T, prediction) + grad_hidden_next

            # Derivative through tanh
            raw_grad_hidden = grad_hidden * Activation.derivative_tanh(
                np.dot(self.weights_input_to_hidden, one_hot_inputs[t]) +
                np.dot(self.weights_hidden_to_hidden, hidden_states[t - 1])
            )

            # Gradient for W_input_to_hidden and W_hidden_to_hidden
            grad_input_hidden_t = np.dot(raw_grad_hidden, one_hot_inputs[t].T)
            grad_hidden_hidden_t = np.dot(raw_grad_hidden, hidden_states[t - 1].T)

            print(f"\nTime step {t}:")
            print("∂L/∂W_input_to_hidden:\n", grad_input_hidden_t)
            print("∂L/∂W_hidden_to_hidden:\n", grad_hidden_hidden_t)
            print("∂L/∂W_hidden_to_output:\n", grad_output_t)

            # Accumulate gradients
            grad_input_to_hidden += grad_input_hidden_t
            grad_hidden_to_hidden += grad_hidden_hidden_t
            grad_hidden_to_output += grad_output_t

            # Backpropagate through time
            grad_hidden_next = np.dot(self.weights_hidden_to_hidden.T, raw_grad_hidden)

        # Update weights
        self.weights_input_to_hidden -= self.learning_rate * grad_input_to_hidden
        self.weights_hidden_to_hidden -= self.learning_rate * grad_hidden_to_hidden
        self.weights_hidden_to_output -= self.learning_rate * grad_hidden_to_output

    def predict_next_word(self, sequence, vocabulary):
        _, _, _, probabilities = self.forward_pass(sequence, vocabulary)
        last_time_step = len(sequence) - 1
        final_prediction_probs = probabilities[last_time_step]

        print(f"\nPrediction for sequence {sequence}:")
        for idx, prob in enumerate(final_prediction_probs):
            print(f"{vocabulary.index_to_word[idx]}: {prob[0]:.4f}")

        return final_prediction_probs

    def train(self, training_sequences, num_epochs=1):
        vocabulary = Vocabulary(["I", "love", "RNNs", "FNNs"])

        print("Training RNN...\n")

        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            for sequence in training_sequences:
                input_sequence = sequence
                target_sequence = sequence[1:]

                one_hot_inputs, hidden_states, raw_outputs, probabilities = self.forward_pass(input_sequence,
                                                                                              vocabulary)

                # Compute loss only for valid target steps
                loss = sum(
                    -np.log(probabilities[t][vocabulary.word_to_index[target_sequence[t]]])
                    for t in range(len(target_sequence))
                )
                total_loss += loss.item()

                # BPTT
                self.backward_pass(one_hot_inputs, hidden_states, probabilities, target_sequence, vocabulary)

        print("\n=== Final Weights After Training ===")
        print("weights_input_to_hidden:\n", self.weights_input_to_hidden)
        print("weights_hidden_to_hidden:\n", self.weights_hidden_to_hidden)
        print("weights_hidden_to_output:\n", self.weights_hidden_to_output)

        return vocabulary

