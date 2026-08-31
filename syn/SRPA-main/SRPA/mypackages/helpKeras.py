# mypackages/helpKeras
import tensorflow as tf


#%% Functions for using Keras to train RNNs
def convert_and_reshape(array, new_shape):
    tensor = tf.convert_to_tensor(array, dtype=tf.float32)
    return tf.reshape(tensor, new_shape)


#%% Functions for learning rates
# Define a function to generate the learning rate dictionary
def generate_learning_rate_dict():
    # Create an empty dictionary to store the combinations and their corresponding learning rates
    learning_rates = {}

    # Manually input the learning rates for each combination
    learning_rates[("He", "GD")] = 1e-4
    learning_rates[("He", "GDC")] = 1
    learning_rates[("He", "GDNes")] = 1e-3
    learning_rates[("He", "SGD")] = 0.1
    learning_rates[("He", "Adam")] = 0.1

    learning_rates[("Gaussian0.001", "GD")] = 1e-3
    learning_rates[("Gaussian0.001", "GDC")] = 1e-4
    learning_rates[("Gaussian0.001", "GDNes")] = 1e-4
    learning_rates[("Gaussian0.001", "SGD")] = 0.1
    learning_rates[("Gaussian0.001", "Adam")] = 1e-2

    learning_rates[("Gaussian0.1", "GD")] = 1e-4
    learning_rates[("Gaussian0.1", "GDC")] = 1e-4
    learning_rates[("Gaussian0.1", "GDNes")] = 1e-4
    learning_rates[("Gaussian0.1", "SGD")] = 0.1
    learning_rates[("Gaussian0.1", "Adam")] = 0.01

    learning_rates[("Glorot", "GD")] = 1
    learning_rates[("Glorot", "GDC")] = 1
    learning_rates[("Glorot", "GDNes")] = 1e-4
    learning_rates[("Glorot", "SGD")] = 0.1
    learning_rates[("Glorot", "Adam")] = 0.01

    learning_rates[("LeCun", "GD")] = 1
    learning_rates[("LeCun", "GDC")] = 1
    learning_rates[("LeCun", "GDNes")] = 0.1
    learning_rates[("LeCun", "SGD")] = 0.1
    learning_rates[("LeCun", "Adam")] = 0.01

    return learning_rates
