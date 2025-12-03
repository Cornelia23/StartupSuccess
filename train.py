"""
train.py: Training utilities and loops for Transformer Language Model
TensorFlow implementation for mystery corpus training
Author: Eric Ewing
"""
import tensorflow as tf
import math
import os
import json
from typing import Tuple, Dict
import tqdm

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)

def scce(target, logits):
    #Returns the loss of a fnction
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(target, logits, from_logits=True)
    )

def split_data(batch):
   input = batch[:, :-1]
   target = batch[:,1:]
   return input, target

def train(model, train_dataset, test_dataset, epochs=5, learning_rate=1e-3,
          wandb_run=None, checkpoint_dir="checkpoints", continue_training=False, submission_tracker=None) -> Tuple[tf.keras.Model, Dict[str, list]]:
    """
    Complete training function for language models.

    Args:
        model: Language model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        epochs: Number of epochs
        learning_rate: Learning rate
        wandb_run: Wandb run for logging
        tokenizer: Tokenizer for text generation
        checkpoint_dir: Directory to save checkpoints
        continue_training: Whether to continue training from latest checkpoint
        submission_tracker: Submission tracker for logging epoch results

    Returns:
        model: Trained model
    """
    # Ensure checkpoint directory exists otherwise create it
    os.makedirs(checkpoint_dir, exist_ok=True) 
    # TODO: Initialize your optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = scce #Setting the loss function equal to 
    # TODO: Set up TensorFlow checkpointing with Checkpoint and CheckpointManager
    start_epoch = 1
    e_num = tf.Variable(1,trainable=False,dtype=tf.float32)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=e_num)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    
    # Handle checkpoint restoration for continue training
    if continue_training:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            # Extract epoch number from checkpoint name
            try:
                start_epoch = int(latest_checkpoint.split('-')[-1])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not determine start epoch, starting from 0")
        else:
            print("No checkpoint found, starting fresh")
    
    # This is to keep track of model's performance during training
    history = {'train_loss': [], 'val_loss': [], 'perplexity': []}

    for current_epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs), desc="Training Progress", position=0):
        total_epochs = start_epoch + epochs - 1
        train_loss = 0
        train_batch_count = 0
        for batch_num, batch in enumerate(train_dataset): 
            input, target = split_data(batch)
            with tf.GradientTape() as tape:
                logits = model.call(input, True)
                loss = loss_fn(target,logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            train_loss += loss.numpy()
            train_batch_count += 1
            if batch_num % 200 == 0: 
                print("batch #: ", batch_num, "Training loss:", loss.numpy(),"\n")
 
        train_avg_loss = (float(train_loss / train_batch_count))

        # TODO: Calculate perplexity from validation loss
        # NOTE: Make sure to call reduce_mean on the loss
        # TODO: Append metrics to history dictionary

        # TODO: Log epoch metrics to the submission tracker (epoch, train_loss, val_loss, perplexity)
    
        test_loss = 0
        test_batch_count = 0
        for batch in test_dataset:
            validate_input, validate_target = split_data(batch)
            predictions = model.call(validate_input, training=False)
            val_loss = loss_fn(validate_target, predictions)
            test_loss += val_loss.numpy()
            test_batch_count += 1

        test_avg_loss = float(test_loss / test_batch_count)

        history['train_loss'].append(train_avg_loss)

        history['val_loss'].append(test_avg_loss)
        val_accuracy = calculate_perplexity(test_avg_loss)
        history['perplexity'].append(val_accuracy)

        if submission_tracker is not None:
            submission_tracker.log_epoch(current_epoch, train_batch_count, train_avg_loss, val_accuracy)
        
        e_num.assign(tf.cast(current_epoch, tf.float32))
        checkpoint_manager.save()

        # TODO : Save model checkpoint periodically or if validation loss improves
        # Log metrics to wandb if available (recommended into batch loop for logging for better tracking)
        # NOTE: If using for batch, make sure to log epoch number, not batch number on some N interval
        if wandb_run:
            wandb_run.log({
                "epoch": current_epoch, # TODO: Current epoch number (one-index, so add 1
                "train_loss": train_avg_loss,  # TODO: Calculate training loss
                "val_loss": test_avg_loss,  # TODO: Calculate validation loss
                "perplexity": val_accuracy  # TODO: Calculate perplexity
            })
        

    return model, history
