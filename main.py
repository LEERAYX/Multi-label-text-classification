# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:50:47 2020

@author: ruixuanl
"""

import os
import pandas as pd
import numpy as np
import time
import datetime
import random
import argparse

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch


class configs:
    
    # Data and output directory config
    data_path = r'./Data'
    output_path = r'./DeepLearningOutput/model_cls'
    
    tone_train = r'tone_train.txt'
    tone_test = r'tone_test.txt'
    code_train = r'code_train.txt'
    code_test = r'code_test.txt'
    
    # Model configs
    num_labels_code = 10
    num_labels_tone = 3
    batch_size = 128
    learning_rate = 2e-5
    eps = 1e-8
    epochs = 7
    
    # Random seed
    seed = 1004
    


class DataReader(object):
    # Load data set
    
    def __init__(self):
        
        pass
    
    def load_data(self, data_path, tokenizer):
        
        data = pd.read_csv(data_path, sep = '\t', encoding='cp1252')
        
        data[data.columns[0]] = data[data.columns[0]] - 1
        
        data = data.replace(np.nan, '', regex=True)
        
        text = data[data.columns[1]].values
        labels = data[data.columns[0]].values
        
        input_ids = []
        attention_masks = []
        
        # For every sentence in data text
        for sent in text:
            
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            #   (7) Return in tensors format
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_token = True,
                max_length = 32,
                pad_to_max_length = True,
                return_attention_mask = True,
                return_tensors = 'pt'
                )
            
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            
            
        # Convert input_ids, attention_masks and label into tensors
        input_ids = torch.cat(input_ids, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        labels = torch.tensor(labels)
        
        return TensorDataset(input_ids, attention_masks, labels)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))       

def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    labels_flat = labels.flatten()
    return np.sum(preds == labels_flat) / len(labels_flat)   

def train(
        train_dataloader,
        validation_dataloader,
        model,
        tokenizer,
        optimizer,
        total_steps,
        device,
        scheduler
        ):
    
    print('Initial training...')
    seed = configs.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Store training statistics (loss)
    training_stats = []
    
    # Measure the total training time
    #total_t0 = time.time()
    
    epochs = configs.epochs
    
    # For each epoch...
    for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        #print("")
        #print("  Average training loss: {0:.4f}".format(avg_train_loss))
        #print("  Training epcoh took: {:}".format(training_time))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
        
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
        
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids,  
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        #print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        #print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        #print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
        )

    print("")
    print("Training complete!")

    #print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    # Display floats with two decimal places.
    pd.set_option('precision', 4)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    return model, df_stats

def make_prediction(test_prediction_data, tokenizer, output_file_name):
    
    # Create test data loader
    test_prediction_sampler = SequentialSampler(test_prediction_data)
    test_prediction_dataloader = DataLoader(
        test_prediction_data,
        sampler = test_prediction_sampler,
        batch_size = args.batch_size
        )
    
    
    # Tracking variables 
    predictions , predict_ids = [], []

    print('Make test predictions...')
    # Predict 
    for batch in test_prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_ids = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,  
                            attention_mask=b_input_mask)
            
        logits = outputs[0]
            
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        b_ids = b_ids.detach().to('cpu').numpy()
  
        # Store predictions and true labels
        predictions.append(logits)
        predict_ids.append(b_ids)

    print('    DONE.')
    
    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_ids = np.concatenate(predict_ids, axis=0)
    
    convert_label = flat_predictions + 1
            
    prediction_output = pd.DataFrame({'Id': flat_ids, 'Predicted': convert_label})
    prediction_output.to_csv(output_file_name+'.csv', index = False)

def read_data(data_path, tokenizer):
        
    data = pd.read_csv(data_path, sep = '\t', encoding='cp1252')
        
    data[data.columns[0]] = data[data.columns[0]] - 1
        
    data = data.replace(np.nan, '', regex=True)
        
    text = data[data.columns[2]].values
        
    input_ids = []
    attention_masks = []
        
    # For every sentence in data text
    for sent in text:
            
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            #   (7) Return in tensors format
        encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_token = True,
                max_length = 32,
                pad_to_max_length = True,
                return_attention_mask = True,
                return_tensors = 'pt'
                )
            
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
            
            
    # Convert input_ids, attention_masks and label into tensors
    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    ids = torch.tensor([int(x) for x in data['Id'].values])
        
    return TensorDataset(input_ids, attention_masks, ids)

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--learning_rate",
        help = "Learning rate for classifier",
        default = configs.learning_rate,
        required = False
        )
    
    argparser.add_argument(
        "--epochs",
        help = "Number of epochs for training",
        default = configs.epochs,
        required = False
        )
    
    argparser.add_argument(
        "--batch_size",
        help = "Batch size for training",
        default = configs.batch_size,
        required = False
        )
    
    argparser.add_argument(
        "--adam_epsilon",
        help = "Adam optimizer epsilon",
        default = configs.eps,
        required = False
        )
    
    argparser.add_argument(
        "--pretrained_model_dir",
        help = "Path to the pretrained language mdoel.",
        default = 'distilbert-base-uncased',
        required = False
        )
    
    argparser.add_argument(
        "--target_type",
        help = "Target classification type = tone, code ",
        required = True
        )
    argparser.add_argument(
        "--model_type",
        help = "Expected model type, bert or distilbert",
        required = False,
        default = 'distilbert'
        )
    
    
    args = argparser.parse_args()
        
    # Select target training class
    if args.target_type == "tone":
        train_path = configs.tone_train
        test_path = configs.tone_test
        num_labels = configs.num_labels_tone
    else:
        train_path = configs.code_train
        test_path = configs.code_test
        num_labels = configs.num_labels_code
    
    # Load Bert tokenizer
    print('Loading Bert tokenizer...')
    if args.model_type == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_model_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)
    
    print('Loaded.\n')
    
    # Load train and validation data sets
    data_reader = DataReader()
    print('Loading data...')
    dataset = data_reader.load_data(os.path.join(configs.data_path, train_path), tokenizer)
    print('Data loaded.\n')
    
    print('Training & Validation data split...')
    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('Training & Validation data split completed.\n')
    
    print("Create DataLoader...")
    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = int(args.batch_size)
        )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler = RandomSampler(val_dataset),
        batch_size = int(args.batch_size)
        )
    print("DataLoader created. \n")

    # Load the pretrained BERT model with a single linear classification layer on top.
    print('Loading BertForSequenceClassification...')
    
    if args.model_type == "bert":
        model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_dir,
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = False
            )
    else:    
        model = DistilBertForSequenceClassification.from_pretrained(
            args.pretrained_model_dir,
            num_labels = num_labels
            )
    print('Model loaded.\n')
    
    print('Checking GPU...')
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # Run the model on GPU
    model.cuda()
    
    print('Setting optimizer...')
    # Set optimizer as AdamW
    optimizer = AdamW(model.parameters(),
                      lr = float(args.learning_rate),
                      eps = float(args.adam_epsilon)
                      )
    print('Optimizer set.\n')
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * int(args.epochs)
    
    # Create the learing rate schedular
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps = total_steps
        )
    
    print("Training start...")
    # Train the model
    model, training_stats = train(train_dataloader,
                                  validation_dataloader,
                                  model,
                                  tokenizer,
                                  optimizer,
                                  total_steps,
                                  device,
                                  scheduler)
    
    print("Validation accuracy:")
    print(training_stats['Valid. Accur.'])
    print(training_stats)
    print("Training completed.\n")

    # Save the final model to model_sav
    output_dir = configs.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('Saving model to %s' %output_dir)
    
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    
    print("Load test data...")
    prediction_data = data_reader.load_data(os.path.join(configs.data_path, test_path), tokenizer)
    print("Test data loaded.\n")
    
    # Create test data loader
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data,
        sampler = prediction_sampler,
        batch_size = configs.batch_size
        )
    
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,  
                            attention_mask=b_input_mask)
            
        logits = outputs[0]
            
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
  
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')
    
    
    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)
    
    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)
    
    flat_acc = flat_accuracy(flat_predictions, flat_true_labels)
    
    print("Accuracy: {0:.4f}".format(flat_acc))
    
    print("Load test data for output...")
    
    if args.target_type == 'tone':
        test_prediction_data = read_data('./Data/tone_test_pred.txt', tokenizer)
        make_prediction(test_prediction_data, tokenizer, 'prediction_tone')
    else:
        test_prediction_data = read_data('./Data/code_test_pred.txt', tokenizer)
        make_prediction(test_prediction_data, tokenizer, 'prediction_code')
    
    
# python main.py --pretrained_model_dir="./LanguageModel" --target_type="tone" --batch_size="64" --learning_rate="2e-5" --epochs="6"
# python main.py --pretrained_model_dir="./LanguageModel" --target_type="tone" --batch_size="128" --learning_rate="2e-5" --epochs="7"

    
    
    
    