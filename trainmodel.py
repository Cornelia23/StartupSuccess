import tensorflow as tf
import math
import os
import json
from typing import Tuple, Dict, List
import tqdm
from model import TransformerCNNModel
from transformer_train import transformer_train 
from traincnn import cnntrain
def train(
        model: TransformerCNNModel,
        train_dataset = tf.data.Dataset,
        epochs: int = 5, 
        learning_rate: float = 1e-4) -> Dict[str, List[float]]:
    '''
    Trains both the transformer and CNN to accomodate categorical and quantitative data 
    Used for fine tuning in addition to the separate training of the transformer and the CNN. 
    :param model: Description
    :type model: TransformerCNNModel
    :param train_dataset: Description
    :param epochs: Description
    :type epochs: int
    :param learning_rate: Description
    :type learning_rate: float
    :return: Description
    :rtype: Dict[str, List[float]]
    '''

    model.transformer.trainable = True
    if model.use_cnn:
        model.cnn.trainable = True

    
    transformer_history = transformer_train(model=model, train_dataset = train_dataset, epochs=epochs, learning_rate=learning_rate)
    if model.use_cnn:
        #TODO: Modify the datase for cnn train, so that it only trains on the quantitative data 
        cnn_history = cnntrain(model=model, optimizer=optimizer, train_inputs=train_dataset, train_labels=train_dataset)
    

    #Train both models at the same time
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for inputs, labels in train_dataset: 
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variaibles)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            pred_classes = tf.argmax(predictions, axis=1)
            true_class = tf.argmax(labels,axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_classes, true_class), tf.float32))
            epoch_loss += float(loss)
            epoch_acc += float(accuracy)
            num_batchs += 1

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc/num_batches

        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        print(f"Epoch: {epoch}/{epochs} - Loss: {avg_loss}, Avg. Accuracy: {avg_loss}")

    all_history = {
        'transformer':transformer_history,
        'cnn':cnn_history,
        'model':history
    }

    return all_history