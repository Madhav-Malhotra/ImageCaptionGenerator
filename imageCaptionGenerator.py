# -*- coding: utf-8 -*-

# Setting up Tensorflow (Using 1.9.0)
import tensorflow as tf
print("Tensorflow Version: " + str(tf.__version__));
print(tf.executing_eagerly());

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Other utilities
import re
import numpy as np
import os
import time
import json
from glob import glob
import pickle
from pathlib import Path


# Downloading the COCO image dataset
annotation_zip = tf.keras.utils.get_file('captions.zip', 
  cache_subdir = os.path.abspath('.'),
  origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
  extract = True);
annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json';

name_of_zip = 'train2014.zip';
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
  image_zip = tf.keras.utils.get_file(name_of_zip, 
    cache_subdir = os.path.abspath('.'),
    origin = 'http://images.cocodataset.org/zips/train2014.zip',
    extract = True);
  PATH = os.path.dirname(image_zip) + '/train2014/';
else:
  PATH = os.path.abspath('.') + '/train2014/';

print("Downloaded Data");

# Read COCO Captions file
with open(Path("annotations/captions_train2014.json"), 'r') as f:
    annotations = json.load(f);

# Storing Captions and Image Names in Lists
all_captions = [];
all_img_name_vector = [];

for annotation in annotations['annotations']:
    caption = '<start> ' + annotation['caption'] + ' <end>';
    image_id = annotation['image_id'];
    full_coco_image_path = os.path.abspath('.') + '/train2014/' + 'COCO_train2014_' + '%012d.jpg' % (image_id);
    
    all_img_name_vector.append(full_coco_image_path);
    all_captions.append(caption);

# Shuffling captions and image_names together
train_captions, img_name_vector = shuffle(all_captions,
  all_img_name_vector,
  random_state = 1);

# Selecting first 30,000 captions for training
num_examples = 100;
train_captions = train_captions[:num_examples];
img_name_vector = img_name_vector[:num_examples];

print("Training data ready");

def load_image(image_path):
    '''
    Processes images into right format for CNN classifier
    Input: str (image filepath)
    Output: Tensor [-1, 299, 299, 3] - 299 x 299 tensor without 
        a batch size set, holding values on RGB channel
    '''
    # Coverting image to tensor with RGB data
    img = tf.io.read_file(image_path);
    img = tf.image.decode_jpeg(img, channels = 3);
    # Resizing tensor to 299 x 299
    img = tf.image.resize(img, (299, 299));
    # Setting up CNN's input 
    img = tf.keras.applications.inception_v3.preprocess_input(img);
    return img, image_path;


# CNN used in preprocessing images
image_model = tf.keras.applications.InceptionV3(include_top = False, 
    weights = 'imagenet');
# Generating images with features extracted using CNN
new_input = image_model.input;
# Getting first hidden layer in caption generator
hidden_layer = image_model.layers[-1].output;
# Copying over CNN to local program with loaded weights
image_features_extract_model = tf.keras.Model(new_input, hidden_layer);

# getting the unique images
encode_train = sorted(set(img_name_vector));
image_dataset = tf.data.Dataset.from_tensor_slices(
    encode_train).map(load_image).batch(16);

print("Devil Loop");
        
for img, path in image_dataset:
  print('.', end='');
  # Pre-processing current batch of images with CNN   
  batch_features = image_features_extract_model(img);
  # Reshaping output to 8 x 8 x 2048 matrix
  batch_features = tf.reshape(batch_features, 
    (batch_features.shape[0], -1, batch_features.shape[3]));

  # Saving extracted features in images  
  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8");
    np.save(path_of_feature, bf.numpy());


def calc_max_length(tensor):
    '''
    Find the longest caption in the data
    Input: Tensor of captions
    Output: Int of number of words in longest captions
    '''
    return max(len(t) for t in tensor);

# Choosing top 30,000 words from vocabulary
mostCommon = 30000;
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = mostCommon, 
  oov_token = "<unk>", 
  filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ');
tokenizer.fit_on_texts(train_captions);
# Creating integer mappings of caption text
train_seqs = tokenizer.texts_to_sequences(train_captions);

#Creating a padding character in the word to index dictionary.
tokenizer.word_index['<pad>'] = 0;
# Padding each vector to max length of captions
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding = 'post');
# Calculating max length of integer mappings to store the attention weights
max_length = calc_max_length(train_seqs);

# Separating training data (80%) and validation data (20%)
img_name_train, img_name_val, captions_train, captions_val = train_test_split(img_name_vector, 
    cap_vector, 
    test_size = 0.2, 
    random_state = 0);
                                                                              
# Hyperparemeters for model
BATCH_SIZE = 64;
BUFFER_SIZE = 1000;
embedding_dim = 256;
units = 512;
vocab_size = len(tokenizer.word_index);
# Shape of vector extracted from CNN is (64, 2048)
features_shape = 2048;
attention_features_shape = 64;
 
def map_func(img_name, caption):
    '''
    Loading processed feature impages
    Input: Str (image filename)
    Input: Tensor (image caption)
    Output: Tensor (image data)
    Output: Tensor (image caption)
    '''
    img_tensor = np.load(img_name.decode('utf-8')+'.npy');
    return img_tensor, caption;

# Creating tensorflow dataset for model
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, captions_train));

# Using map to load the numpy files in parallel with 2 cores
dataset = dataset.map(lambda item1, item2: tf.py_func(
    map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls = 2);

# Shuffling and batching data
dataset = dataset.shuffle(BUFFER_SIZE);
dataset = dataset.batch(BATCH_SIZE);
dataset = dataset.prefetch(1);

def gru(units):
    '''
    Creating the RNN layer used by the captions generator
    Input: Int (number of neurons in layer)
    Output: Tensorflow RNN Layer
    '''
    return tf.keras.layers.GRU(units, 
       return_sequences = True, 
       return_state = True, 
       recurrent_activation = 'sigmoid', 
       recurrent_initializer = 'glorot_uniform');

class BahdanauAttention(tf.keras.Model):
  '''
  A specific type of algorithm for GRU RNN layer
  This reshapes tensors into appropriate shapes for the layers
  '''  
  def __init__(self, units):
    super(BahdanauAttention, self).__init__();
    # Weights used
    self.W1 = tf.keras.layers.Dense(units);
    self.W2 = tf.keras.layers.Dense(units);
    # Vector used 8x8x2048
    self.V = tf.keras.layers.Dense(1);
  
  def call(self, features, hidden):
    # Features(CNN_encoder output) shape - [batch_size, 64, embedding_dim] 
    # Hidden shape - [batch_size, hidden_size]
    
    # Hidden_with_time_axis shape - [batch_size, 1, hidden_size]
    hidden_with_time_axis = tf.expand_dims(hidden, 1);
    
    # Score shape - [batch_size, 64, hidden_size]
    # Reshaping to 64x2048 vector
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis));
    
    # Attention_weights shape - [batch_size, 64, 1]
    attention_weights = tf.nn.softmax(self.V(score), axis = 1);
    
    # context_vector shape after sum - [batch_size, hidden_size]
    context_vector = attention_weights * features;
    context_vector = tf.reduce_sum(context_vector, axis = 1);
    
    return context_vector, attention_weights;

class CNN_Encoder(tf.keras.Model):
    '''
    Encoder passes image features through fully-connected CNN layer
    '''
    
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__();
        # shape after layer is [batch_size, 64, embedding_dim]
        self.fully_connected = tf.keras.layers.Dense(embedding_dim);
        
    def call(self, x):
        # Passing data through layer
        x = self.fully_connected(x);
        x = tf.nn.relu(x);
        return x;
    
class RNN_Decoder(tf.keras.Model):
  '''
  This RNN predicts the next caption word for images
  '''
  def __init__(self, embedding_dim, units, vocab_size):
    # Setting up RNN
    super(RNN_Decoder, self).__init__();
    # Number of neurons used in layer shapes
    self.units = units;
    # Input layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim);
    # RNN layer
    self.gru = gru(self.units);
    # First processing layer
    self.fc1 = tf.keras.layers.Dense(self.units);
    # Output layer
    self.fc2 = tf.keras.layers.Dense(vocab_size);
    # Algorithm used with RNN
    self.attention = BahdanauAttention(self.units)
        
  def call(self, x, features, hidden):
    # Defining attention as separate model (from RNN)
    context_vector, attention_weights = self.attention(features, hidden);
    
    # x shape after passing through embedding layer - 
    # [batch_size, 1, embedding_dim]
    x = self.embedding(x);
    
    # x shape after concatenation - 
    # [batch_size, 1, embedding_dim + hidden_size]
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1);
    
    # Passing concatenated vector to GRU layer
    output, state = self.gru(x);
    
    # x shape - [batch_size, max_length, hidden_size]
    x = self.fc1(output);
    
    # x shape - [batch_size * max_length, hidden_size]
    x = tf.reshape(x, (-1, x.shape[2]));
    
    # output shape - [batch_size * max_length, vocab]
    x = self.fc2(x);

    return x, state, attention_weights;

  def reset_state(self, batch_size):
    '''
    Sets Tensor values to 0 to reset RNN state
    '''
    return tf.zeros((batch_size, self.units));

encoder = CNN_Encoder(embedding_dim);
decoder = RNN_Decoder(embedding_dim, units, vocab_size);

optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
    '''
    Calculating difference between predictions and correct results
    Calculated loss masked for padding
    '''
    mask = 1 - np.equal(real, 0);
    # Calculating loss with softmax algorithm
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = real, logits = pred) * mask;
    return tf.reduce_mean(loss);

# Array reset after multiple training rounds
loss_plot = [];

# Hyperparameter: Number of training epochs
EPOCHS = 20;

for epoch in range(EPOCHS):
    # To log initial state
    start = time.time();
    total_loss = 0;
    
    # Training with images in batches
    for (batch, (img_tensor, target)) in enumerate(dataset):
        loss = 0;
        
        # Initializing hidden state for each batch
        # Note: Prevents different image captions from being related
        hidden = decoder.reset_state(batch_size=target.shape[0]);

        # Start caption token passed to decoder
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1);
        
        with tf.GradientTape() as tape:
            features = encoder(img_tensor);
            
            for i in range(1, target.shape[1]):
                # Passing image features through decoder
                predictions, hidden, previous = decoder(dec_input, features, hidden);
                # Updating loss in current batch
                loss += loss_function(target[:, i], predictions);
                
                # Using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1);
        
        # Calculating model's total error
        total_loss += (loss / int(target.shape[1]));
        variables = encoder.variables + decoder.variables;
        # Cacluating changes required in model weights and biases
        gradients = tape.gradient(loss, variables) ;
        # Updating model weights and biases
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step());
        
        # Providing progress update with current batch
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
              batch, 
              loss.numpy() / int(target.shape[1])));

    # Storing epoch end loss value to plot later
    loss_plot.append(total_loss / len(cap_vector));
    
    # Providing progress update with current epoch
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
         total_loss/len(cap_vector)));
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start));

def evaluate(image):
    '''
    Checking model's performance with new data
    '''
    # Resetting attention and hidden state for evaluation
    attention_plot = np.zeros((max_length, attention_features_shape));
    hidden = decoder.reset_state(batch_size = 1);
    
    # Converting image to input tensor shape
    temp_input = tf.expand_dims(load_image(image)[0], 0);
    # Filling tensor with values
    img_tensor_val = image_features_extract_model(temp_input);
    # Reshaping tensor for input to model
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]));

    # Extracting features from image with CNN
    features = encoder(img_tensor_val);

    # Passing start caption to RNN
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0);
    result = [];

    for i in range(max_length):
        # Updating prediction, hidden layers, and attention
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden);
        
        # Reshaping attention tensor
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy();
        
        # Current predicted word in caption
        predicted_id = tf.argmax(predictions[0]).numpy();
        # Adding caption word to rest of caption
        result.append(tokenizer.index_word[predicted_id])

        # Returning finished caption
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot;

        # Passing predicted caption to RNN
        dec_input = tf.expand_dims([predicted_id], 0);
    
    # Returning predictions and attention
    attention_plot = attention_plot[:len(result), :];
    return result, attention_plot;