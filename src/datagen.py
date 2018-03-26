import load
import keras
import numpy as np
import pandas as pd
import struct

# See https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
# for an example of not-padding by using np.zeros, so where there is no letter
# the one-hot encoding is all zeros.
# input is char string representing addition of 2 numbers (111+222).  output is sum (333).
# input is encoded.  The length of the output is fixed, but different from input.
# encoding is repeated output_length times and fed to decoder RNN layer.
# Use TimeDistributed to apply the same dense layer to each output of decoder RNN layer.
# predicting gives a whole input and gets a whole output from the model.

# See https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
# y is one character instead of seq_len characters
# generates text by selecting a starting point in training text at random, selecting a seq_len substring
# and generating a next character probability distribution, sampling a char from the distribution, and changing
# input string by removing the first char and appending this new char.
# This approach is like https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

# See https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Task: given english sentence, predict french sentence.
# Input sentence goes through encoder RNN and makes an encoding.
# Decoder RNN takes as input at each step the encoding vector, the previous state (as usual) 
# and the previous output character, which comes from the target output (this technique is called "teacher forcing").
# For generating text, an encoder model is created by reusing encoder inputs and encoder outputs (and so uses the 
# same weights as the training encoder.  The decoder is created by reusing decoder_lstm to take 
# decoder_state_inputs as initial state. (the training decoder was wired to encoder_states.)  
# The decoder outputs its output and its state.
# Decoding happens by encoding the input sequence with the encoder.  
# Create a target sequence with just the start char, '\t'.
# Feed decoder the start token as input and encoder state as initial state.
# Until done, get last output of decoder, sample a character from that prob dist, add the character to the generated
# sequence, extend the target vector with the new character, and predict again to get the next character.  Repeat until
# max length or end char (\n) is output.

# My approach:
# Task: given sequence, predict next char
# Inputs: target sequence.  Outputs: target sequence shifted over one.  (This is teacher forcing).  Initial state?
# Decoding: what is the initial character?


def text_to_sequences(text, seq_len, stride):
    '''
    divide the text into sequences of length seq_len by 
    striding it in strides of length stride and taking the subsequence 
    of length seq_len at that position.

    return: list of sequences
    '''
    sequences = []
    next_chars = []
    text_len = len(text)
    for i in range(0, text_len - (seq_len + 1), stride):
        j = i + seq_len
        if j < text_len:
            seq = normalized_text[i:j]
            next_char = normalized_text[j]
            sequences.append(seq)
            next_chars.append(next_char)
    
    sequences = [normalized_text[i:(i + seq_len)] 
                 for i in range(0, len(normalized_text), stride)]
    return sequences


def preprocess_names():
    '''
    return: a tuple containing: a text of female names, a text of male names.
    '''
    names = load.load_names()
    male_text = ' '.join(names[names.sex == 'male'].name)
    female_text = ' '.join(names[names.sex == 'female'].name)
    return (male_text, female_text)
    

def preprocess_jokes():
    '''
    return: text containing jokes.
    '''
    jokes = load.load_jokes()
    return ' '.join(jokes.body)


def preprocess_pnp():
    '''
    return: text of pride and prejudice.
    '''
    pnp_text = load.load_pride_and_prejudice()
    # pnp has 31 header lines before the book title and 366 footer lines after the end of the book.
    # remove the project gutenberg header and footer
    return ' '.join(list(pnp_text.splitlines())[31:-366])


def chars_to_latin1_ints(text):
    '''
    text: a string containing characters that can be encoded with ISO-8859-1
    encode characters from 0 to 255 using ISO-8859-1, aka ISO Latin 1.
    return: a list of numbers in [0, 255]
    '''
    return [int(c.encode('iso-8859-1')) for c in text]


def latin1_ints_to_chars(seq):
    '''
    seq: a sequence of numbers in [0, 255].
    return: a string.
    '''
    # convert integer to byte.  decode byte to a character using ISO-8859-1
    return ''.join([struct.pack("B", i).decode('iso-8859-1') for i in seq])


def pad_text(text, length, pad_char=' '):
    '''
    text: string
    return: string, whose length is `length` and starting with `text` and ending with 
    as many `pad_char` appended as needed to make the string `length` characters long.
    For batch training.
    I have not seen how others pad text, so this method is a guess at being a good way to do it.
    '''
    return text[:length] + (' ' * (length - len(text)))


def get_char_to_int(vocab_size):
    return {struct.pack("B", i).decode('iso-8859-1'): i for i in range(vocab_size)}


def get_int_to_char(vocab_size):
    return {i: struct.pack("B", i).decode('iso-8859-1') for i in range(vocab_size)}


def offset_by_one(text, last_char=' '):
    '''
    text: string
    return: string, where each character is the corresponding character in text, shifted over one, 
      s.t. output[i] == text[i+1].  The last char, output[-1] = last_char.
    '''
    return text[1:] + last_char


def text_to_ints(text, char_to_int):
    '''
    text: string
    char_to_int: used to map text to ints.
    return: list of ints
    '''
    return [char_to_int[c] for c in text]


def sample_index(preds, use_max=True):
    '''
    '''
    if use_max:
        # return max index (Maximum A Posteriori character estimate)
        return numpy.argmax(preds)
    else:
        # choose index at random, weighted by preds
        np.random.choice(len(preds), p=preds)


def normalize_text(text):
    '''
    normalize text by first lowercasing it and then splitting
    it on text on whitespace and recombine the tokens using a 
    single space char.
    '''
    return ' '.join(text.split()).lower()


def get_text():
     # get raw text
    female_names_text, male_names_text = preprocess_names()
    jokes_text = preprocess_names()
    pnp_text = preprocess_pnp()
    return normalize_text(male_names_text)


def get_tensors(text, seq_len, stride, vocab_size):
    '''
    Return a tensor of shape (m, seq_len, vocab_size).  Each example is
    created from the text source by sliding a window of size seq_len across
    the text, using a stride of stride. At each step, characters in the window 
    are converted to integers using ISO Latin 1, converting any incompatible characters
    to encodable ones as needed.  The integers are one hot encoded.

    seq_len: int, the "time" dimension in the recurrent sense.
    The number of characters in the string.  Fixed for batch processing.
    stride: int, the stride/step length of the sliding window.
    '''    
    
    # convert chars to ints and back efficiently
    int_to_char = get_int_to_char(vocab_size)
    char_to_int = get_char_to_int(vocab_size)

    
    # convert to overlapping sequences and next char
    
    , to integers, to one-hot encoded vectors
    raw_text = male_names_text # shape: (,)
    text_x = text_to_sequences(raw_text, seq_len, stride) # shape: (m,)
    padded_x = [pad_text(t, seq_len) for t in text_x] # shape: (m, n_t)
    padded_y = [offset_by_one(t) for t in padded_x] # shape: (m, n_t)
    int_x = np.array([text_to_ints(t, char_to_int) for t in padded_x]) # shape (m, n_t)
    int_y = [text_to_ints(t, char_to_int) for t in padded_y] # shape (m, n_t)
    x = keras.utils.to_categorical(int_x, num_classes=vocab_size) # shape (m, n_t, n_x)
    y = keras.utils.to_categorical(int_y, num_classes=vocab_size) # shape (m, n_t, n_y)
    
    return x, y
    
     # convert string to list of ints, substituting chars that cannot be encoded
#     def str_to_ints(s):
#         return [i for i in s.replace('\u201c', '"').replace('\u201d', '"').encode('iso-8859-1')]
    
    # transform a list of lists of ints into an ndarray.
#     return np.array(text.apply(str_to_ints).tolist())


#     return text.apply(lambda x: x + (' ' * (length - len(x))))

#     pad_text(male.text, n_t)

#         return text.apply(lambda x: x[1:] + last_char)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
