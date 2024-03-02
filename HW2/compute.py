#!/usr/bin/env python
# coding: utf-8

"""
This module contains functions and classes for training and testing a deep learning model for caption generation.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import json
import os
from torch.utils.data import Dataset
from scipy.special import expit
import pickle
from torch.utils.data import DataLoader

# Data Preprocessing

def data_preprocess():
    """
    Preprocesses the data by loading the training labels and creating word dictionaries.

    Returns:
    i2w (dict): Index to word dictionary
    w2i (dict): Word to index dictionary
    word_dict (dict): Word count dictionary
    """
    filepath = '/scratch1/darumil/DL/'
    with open(filepath + 'training_label.json', 'r') as f:
        file = json.load(f)
    word_count = {}
    for d in file:
        for s in d['caption']:
            # Count the occurrences of each word
            if s not in word_count:
                word_count[s] = 1
            else:
                word_count[s] += 1

    word_dict = {}
    for word in word_count:
        if word_count[word] > 4:
            # Add words with count greater than 4 to the word dictionary
            word_dict[word] = len(word_dict)

    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index

    return i2w, w2i, word_dict

def s_split(sentence, word_dict, w2i):
    """
    Splits a sentence into tokens and converts them to their corresponding indices.

    Args:
    sentence (str): The input sentence
    word_dict (dict): Word count dictionary
    w2i (dict): Word to index dictionary

    Returns:
    sentence (list): List of indices representing the sentence
    """
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            # Replace unknown words with <UNK> token
            sentence[i] = '<UNK>'
        sentence[i] = w2i[sentence[i]]

    sentence.insert(0, 1)  # Add <SOS> token at the beginning
    sentence.append(2)  # Add <EOS> token at the end
    return sentence

def annotate(label_file, word_dict, w2i):
    """
    Annotates the captions in the label file by converting them to indices.

    Args:
    label_file (str): Path to the label file
    word_dict (dict): Word count dictionary
    w2i (dict): Word to index dictionary

    Returns:
    annotated_caption (list): List of tuples containing the annotated captions
    """
    label_json = '/scratch1/darumil/DL/' + label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            annotated_caption.append((d['file_name'], s_split(s, word_dict, w2i)))

    return annotated_caption

def avi(files_dir):
    """
    Loads the avi data from the given directory.

    Args:
    files_dir (str): Path to the directory containing the avi files

    Returns:
    avi_data (dict): Dictionary containing the avi data
    """
    avi_data = {}
    training_feats = '/scratch1/darumil/DL/' + files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

def minibatch(data):
    """
    Sorts the data by caption length and creates minibatches.

    Args:
    data (list): List of tuples containing the avi data and captions

    Returns:
    avi_data (torch.Tensor): Tensor containing the avi data
    targets (torch.Tensor): Tensor containing the target captions
    lengths (list): List of caption lengths
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

class training_data(Dataset):
    # Your code here
    """
    Dataset class for training data.
    """
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = avi(label_file)
        self.w2i = w2i
        self.data_pair = annotate(files_dir, word_dict, w2i)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return data, sentence

class test_data(Dataset):
    """
    Dataset class for test data.
    """
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            # Load avi data
            self.avi.append(np.load(os.path.join(test_data_path, file)))
        
    def __len__(self):
        return len(self.avi)
    
    def __getitem__(self, idx):
        return self.avi[idx]

# Models

class attention(nn.Module):
    """
    Attention module for the decoder.
    """
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)
        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context

class encoderRNN(nn.Module):
    """
    Encoder module for the model.
    """
    def __init__(self):
        super(encoderRNN, self).__init__()
        
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)
    
    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)
        output, hidden_state = self.gru(input)
        return output, hidden_state

class decoderRNN(nn.Module):
    """
    Decoder module for the model.
    """
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(decoderRNN, self).__init__()
        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)
    
    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = torch.tensor(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []
        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()
        for i in range(seq_len-1):
            # Decoder forward pass
            pass
            
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = torch.ones(batch_size, 1, dtype=torch.long).cuda()
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            # Decoder inference pass
            pass
            
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
    
    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85)) # inverse of the logit function

class MODELS(nn.Module):
    """
    Model class for the complete model.
    """
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            # Training mode
            seq_logProb = []
            seq_predictions = []
            # Add your code here
            
            return seq_logProb, seq_predictions
        

# Training

def calculate_loss(loss_fn, x, y, lengths):
    """
    Calculates the loss for a batch of predictions.

    Args:
    loss_fn: Loss function
    x (torch.Tensor): Predicted sequences
    y (torch.Tensor): Ground truth sequences
    lengths (list): List of caption lengths

    Returns:
    loss (torch.Tensor): Loss value
    """
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True
    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] -1
        # Calculate loss for each sequence
        
    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss/batch_size
    return loss

def minibatch(data):
    """
    Sorts the data by caption length and creates minibatches.

    Args:
    data (list): List of tuples containing the avi data and captions

    Returns:
    avi_data (torch.Tensor): Tensor containing the avi data
    targets (torch.Tensor): Tensor containing the target captions
    lengths (list): List of caption lengths
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

def train(model, epoch, loss_fn, parameters, optimizer, train_loader):
    """
    Trains the model for one epoch.

    Args:
    model: The model to be trained
    epoch (int): The current epoch number
    loss_fn: Loss function
    parameters: Model parameters
    optimizer: Optimizer
    train_loader: DataLoader for training data
    """
    model.train()
    print(epoch)
    
    for batch_idx, batch in enumerate(train_loader):
        # Training loop
        # Add your code here
        pass
        
    loss = loss.item()
    print(loss)

def test(test_loader, model, i2w):
    """
    Tests the model on the test data.

    Args:
    test_loader: DataLoader for test data
    model: The model to be tested
    i2w (dict): Index to word dictionary

    Returns:
    ss (list): List of predicted sentences
    """
    model.eval()
    ss = []
    for batch_idx, batch in enumerate(test_loader):
        # Testing loop
        pass
        
    return ss

def main():
    """
    Main function for training and testing the model.
    """
    i2w, w2i, word_dict = data_preprocess()
    with open('i2w.pickle', 'wb') as handle:
        # Save i2w dictionary to a file
        pickle.dump(i2w, handle, protocol=pickle.HIGHEST_PROTOCOL)

    label_file = '/training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = training_data(label_file, files_dir, word_dict, w2i)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)

    epochs_n = 100
    encoder = encoderRNN()
    decoder = decoderRNN(512, len(i2w) +4, len(i2w) +4, 1024, 0.3)
    model = MODELS(encoder=encoder, decoder=decoder)
    
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)
    
    for epoch in range(epochs_n):
        # Training loop
        # Add your code here
        
        torch.save(model, "{}/{}.h5".format('SavedModel', 'model0'))
        
    print("Training finished")
    
if __name__ == "__main__":
    main()

