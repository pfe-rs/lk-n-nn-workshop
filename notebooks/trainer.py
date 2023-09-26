import pickle
from enum import Enum
import os
import numpy as np
from PIL import Image


def load_data(dataset_path):
    """
    Loads the dataset from given file and returns three subsets: training, validation and test (i.e. blind).
    :param dataset_path: Path to dataset file.
    """
    file = open(dataset_path, 'rb')
    train_set, valid_set, test_set = pickle.load(file, encoding='latin1')
    file.close()
    return train_set, valid_set, test_set


def get_training_sample_by_index(dataset, index):
    x = dataset[0][index]
    y = dataset[1][index]
    return x, y


def get_random_sample(dataset):
    index = np.random.randint(0, dataset[0].shape[0])
    return get_training_sample_by_index(dataset, index)


def sample_to_image(x: np.ndarray):
    image = np.reshape(x, newshape=(28, 28))
    image = Image.fromarray(np.uint8(image*255))
    return image.resize(size=(140, 140))


def train_nn_with_sgd(neural_net, dataset_path, epochs_count, alpha):
    """ Performs training of the MLP network on the given dataset using stochastic gradient descent.
        :param dataset_path: Path to dataset file.
        :param epochs_count: Number of epochs to run training.
        :param alpha: Learning rate to be used during training.
    """
    # Load the datasets.
    train_set, valid_set, test_set = load_data(dataset_path)
    
    # Print header.
    print('Epoch\tTrainingError%%\tValidationError%%\tTestError%%')
    # Train network for limited number of epochs.
    train_input, train_target = train_set
    for epoch in range(epochs_count):
        # Go over all samples from test set (shape of input is 50000 x 784 since we have 50000 images of a size 784
        # (784 == 28 x 28)).
        for i in range(train_input.shape[0]):
            # Take current example.
            train_input_reshaped = train_input[i].reshape(train_input.shape[1], 1)
            # Perform forward pass on current example.
            neural_net.forward(train_input_reshaped)
            # Back-propagate error.
            neural_net.backward(train_target[i])
            # Use gradients from back-propagation to update weights.
            neural_net.update_weights(alpha)
        # Measure and print accuracy on all data sets.
        train_error = neural_net.test(train_set)
        valid_error = neural_net.test(valid_set)
        test_error = neural_net.test(test_set)
        print('%d\t%f\t%f\t%f' %(epoch, 100 * train_error, 100 * valid_error, 100 * test_error))
    # Save the trained network.
    file_path_os_normalized = os.path.join(".", "model", "nn.pkl")
    file = open(file_path_os_normalized,'wb')
    pickle.dump(neural_net, file)
    file.close()

def test_nn(nn_path, dataset_path):
    """ Performs testing of the MLP network on the given dataset.
        :param nn_path: Path to nn file.
        :param dataset_path: Path to dataset file.
    """
    # Load neural network.
    nn_file = open(nn_path, 'rb')
    neural_net = pickle.load(nn_file, encoding='latin1')
    nn_file.close()
    # Load datasets.
    ds_file = open(dataset_path, 'rb')
    _, _, test_set = pickle.load(ds_file, encoding='latin1')
    ds_file.close()
    # Test.
    print('Test neural net %s on test set in %s' % (nn_path, dataset_path))
    test_error = neural_net.test(test_set)
    print('Test error: %f' %(100 * test_error))


class Scenario(Enum):
    """ Defines all possible scenarios. """
    TEST = 1
    TRAIN = 2

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('data/mnist.pkl')
