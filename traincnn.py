from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from cnn import CNN
from mlp import MLP
from typing import Tuple, List, Union

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def cnntrain(model: Union[CNN, MLP], optimizer: tf.keras.optimizers.Optimizer, train_inputs: tf.Tensor, train_labels: tf.Tensor) -> float:
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), shape (num_labels, num_classes) - one-hot encoded
    :return: training accuracy for the epoch
    '''
    num_classes = train_inputs.shape[0]
    shuffled_indicies = tf.random.shuffle(tf.range(num_classes))
    train_inputs = tf.gather(train_inputs, shuffled_indicies)
    train_labels = tf.gather(train_labels, shuffled_indicies)
    total_accuracy = 0
    batch_num = num_classes // model.batch_size
    for i in range(batch_num):
        batch_inputs, batch_labels = get_next_batch(i, train_inputs, train_labels, model.batch_size)
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)

        with tf.GradientTape() as tape:
            '''calculate logits and lost'''
            logits = model.call(batch_inputs, False) 
            loss = model.loss(logits, batch_labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        '''Calculate gradient tape using the loss and training variables'''
        '''Aply opitimizer to the data '''
        accuracy = model.accuracy(logits, batch_labels)
        total_accuracy += accuracy
    '''Track accuracy and calculate total accuracy'''

    return float((total_accuracy / batch_num))


def test(model: Union[CNN, MLP], test_inputs: tf.Tensor, test_labels: tf.Tensor) -> Tuple[float, List[int]]:
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param model: the trained model to evaluate
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes) - one-hot encoded
    :return: tuple of (test accuracy, list of predictions)
    test accuracy should be the average accuracy across all batches
    predictions is a list of predicted class indices
    """
    num_classes = test_inputs.shape[0]
    batch_num = num_classes // model.batch_size
    total_accuracy = 0
    predictions = []
    for batch in range(batch_num):
        batch_inputs, batch_labels = get_next_batch(batch, test_inputs, test_labels, model.batch_size)
        logits = model.call(batch_inputs, is_testing=True)
        accuracy = model.accuracy(logits, batch_labels)
        total_accuracy += accuracy
        prediction_index = tf.argmax(logits, axis = 1)
        predictions.extend(prediction_index)


    return float((total_accuracy/batch_num)), predictions
        

def visualize_loss(losses: List[float]) -> None:
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs: np.ndarray, logits: tf.Tensor, image_labels: tf.Tensor, first_label: str, second_label: str) -> None:
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param logits: the output of model.call(), shape (50, num_classes) - raw model outputs
    :param image_labels: the labels from get_data(), shape (50, num_classes) - one-hot encoded
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title=f"PL: {pl}\nAL: {al}")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(logits, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def main() -> None:
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    Students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = 'data/train'
    AUTOGRADER_TEST_FILE = 'data/test'

    LOCAL_TRAIN_FILE = 'data/train'
    LOCAL_TEST_FILE = 'data/test'


    # TODO: assignment.main() pt 1
    # Load your testing and training data using the get_data function
    train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, classes=[0,1])
    test_inputs, test_labels = get_data(LOCAL_TEST_FILE, classes=[0,1])
    

    # TODO: assignment.main() pt 2
    # Initialize your model and optimizer
    model = MLP(classes=[0,1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # TODO: assignment.main() pt 3
    # Train your model
    num_epochs = 10
    for epoch in range(num_epochs):
        accuracy_train = train(model, optimizer, train_inputs, train_labels)
        accuracy_test, test_predictions = test(model, test_inputs, test_labels)

        print("Training accuracy:", accuracy_train)
        print("Testing accuracy:", accuracy_test, "\n")

    # TODO: assignment.main() pt 4
    # Test your model
    cnn_model = CNN(classes = [0,1])
    cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        cnn_accuracy_train = train(cnn_model, cnn_optimizer, train_inputs, train_labels)
        cnn_accuracy_test, test_predictions = test(cnn_model, test_inputs, test_labels)

        print("Training accuracy:", cnn_accuracy_train)
        print("Testing accuracy:", cnn_accuracy_test, "\n")
        print("Test predictions:", test_predictions)

    
    visualize_inputs = test_inputs[:50]
    visualize_labels = test_labels[:50]

    visualize_logits = model.call(visualize_inputs, False)
    loss_visual = model.loss(visualize_logits)

    visualize_results(visualize_inputs, visualize_logits, visualize_labels, "cat", "dog")
    visualize_loss(model.loss_list)

if __name__ == '__main__':
    main()