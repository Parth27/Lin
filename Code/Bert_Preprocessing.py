import tensorflow as tf
import pandas as pd
import os
import pickle
import bert
from bert import tokenization
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import numpy as np
import sys
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random
import warnings
import stanfordnlp
import re


def prepare_data(train_VP,test_VP,max_seq_length=None):
  VP_tokens = list(map(tokenizer.tokenize,train_VP))

  VP_tokens = list(map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], VP_tokens))

  VP_masks = list(map(lambda tok: [1]*len(tok), VP_tokens))

  VP_token_ids = list(map(tokenizer.convert_tokens_to_ids,VP_tokens))

  if not max_seq_length:
    max_seq_length = max([len(x) for x in VP_tokens])

  VP_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), VP_token_ids)
  VP_masks = map(lambda masks: masks + [0] * (max_seq_length - len(masks)), VP_masks)
  VP_segments = map(lambda tokens: [0]*max_seq_length, VP_tokens)

  VP_token_ids = np.array(list(VP_token_ids))
  VP_masks = np.array(list(VP_masks))
  VP_segments = np.array(list(VP_segments))

  test_VP_tokens = list(map(tokenizer.tokenize,test_VP))

  test_VP_tokens = list(map(lambda tok: ["[CLS]"] + tok[:max_seq_length-2] + ["[SEP]"], test_VP_tokens))

  test_VP_masks = list(map(lambda tok: [1]*len(tok), test_VP_tokens))

  test_VP_token_ids = list(map(tokenizer.convert_tokens_to_ids,test_VP_tokens))

  test_VP_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), test_VP_token_ids)
  test_VP_masks = map(lambda masks: masks + [0] * (max_seq_length - len(masks)), test_VP_masks)
  test_VP_segments = map(lambda tokens: [0]*max_seq_length, test_VP_tokens)

  test_VP_token_ids = np.array(list(test_VP_token_ids))
  test_VP_masks = np.array(list(test_VP_masks))
  test_VP_segments = np.array(list(test_VP_segments))
  return VP_token_ids,VP_masks,VP_segments,test_VP_token_ids,test_VP_masks,test_VP_segments

def prepare_sentences(data):
  max_sentence = 300

  tokens = list(map(tokenizer.tokenize,data['Sentence'].astype(str)))
  tokens = [x[:max_sentence-2] for x in tokens]

  tokens = list(map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], tokens))
  masks = list(map(lambda tok: [1]*len(tok), tokens))

  token_ids = list(map(tokenizer.convert_tokens_to_ids,tokens))

  token_ids = map(lambda tids: tids + [0] * (max_sentence - len(tids)), token_ids)
  masks = map(lambda masks: masks + [0] * (max_sentence - len(masks)), masks)
  segments = map(lambda tokens: [0]*max_sentence, tokens)

  token_ids = list(token_ids)
  masks = list(masks)
  segments = list(segments)
  print(len(segments))
  return token_ids,masks,segments

def create_features(train_VP,test_VP,train_context,test_context,token_ids,test_token_ids,masks,test_masks,segments,test_segments,max_seq_length=None):
  VP_token_ids,VP_masks,VP_segments,test_VP_token_ids,test_VP_masks,test_VP_segments = prepare_data(train_VP,test_VP,max_seq_length)
  context_ids = np.array([token_ids[x] for x in train_context])
  context_masks = np.array([masks[x] for x in train_context])
  context_segments = np.array([segments[x] for x in train_context])

  X_train_ids = np.concatenate((VP_token_ids,context_ids),axis=1)
  X_train_masks = np.concatenate((VP_masks,context_masks),axis=1)
  X_train_segments = np.concatenate((VP_segments,context_segments),axis=1)

  test_context_ids = np.array([test_token_ids[x] for x in test_context])
  test_context_masks = np.array([test_masks[x] for x in test_context])
  test_context_segments = np.array([test_segments[x] for x in test_context])

  X_test_ids = np.concatenate((test_VP_token_ids,test_context_ids),axis=1)
  X_test_masks = np.concatenate((test_VP_masks,test_context_masks),axis=1)
  X_test_segments = np.concatenate((test_VP_segments,test_context_segments),axis=1)
  return X_train_ids,X_train_masks,X_train_segments,X_test_ids,X_test_masks,X_test_segments

class BertLayer(tf.keras.layers.Layer):
  def __init__(self, n_fine_tune_layers=10, **kwargs):
    self.n_fine_tune_layers = n_fine_tune_layers
    self.trainable = True
    self.output_size = 768
    super(BertLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    self.bert = hub.Module(bert_path,trainable=self.trainable,name="bert_module")
    trainable_vars = self.bert.variables
    
    # Remove unused layers
    trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
    
    # Select how many layers to fine tune
    trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
    
    # Add to trainable weights
    for var in trainable_vars:
      self._trainable_weights.append(var)
    
    # Add non-trainable weights
    for var in self.bert.variables:
      if var not in self._trainable_weights:
        self._non_trainable_weights.append(var)
    
    super(BertLayer, self).build(input_shape)

  def call(self, inputs):
    inputs = [K.cast(x, dtype="int32") for x in inputs]
    input_ids, input_mask, segment_ids = inputs
    bert_inputs = dict(
        input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
    )
    result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
        "pooled_output"
    ]
    return result

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_size)

  def get_config(self):
    config = super().get_config().copy()
    config.update({'n_fine_tune_layers': self.n_fine_tune_layers})
    return config

def create_model(max_seq_length):
  in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
  in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
  in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
  bert_inputs = [in_id, in_mask, in_segment]

  # Instantiate the custom Bert Layer defined above
  bert_output = BertLayer(n_fine_tune_layers=10)(bert_inputs)

  # Build the rest of the classifier 
  dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
  pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

  model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

#Preprocess data
def preprocess(data,data_path):
    contractions = {}
    data['Sentence'] = data['Sentence'].astype(str)
    #List of contractions from Wikipedia
    with open(data_path+'Contractions','r') as f:
        contractions = dict([x.split(':') for x in f.read().strip().split('\n')])
        
    keys = contractions.keys()
    for key in list(keys):
        contractions[key] = contractions[key].strip()
        contractions[key[0].upper()+key[1:]] = contractions[key][0].upper()+contractions[key][1:]

    for key in contractions:
        data['Sentence'] = data['Sentence'].str.replace(key,contractions[key])

    for i in range(data.shape[0]):
        sent  = data.iloc[i]['Sentence']
        data.at[i,'Sentence'] = ' '.join(re.sub(r'"','',re.sub(r"['#@_%*()</>^&=]*[\-]*",'',sent)).split())

    for key in ('email','Email'):
        data['Sentence'] = data['Sentence'].str.replace(key,contractions[key])

    return data

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

data_path = '../Datasets/New_Files/'
tf.compat.v1.reset_default_graph()
vocab_file = data_path+'bert_vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
#dataset = sys.argv[1]
data = pd.read_excel(data_path+'Labeled_data.xlsx')
test_data = pd.read_excel(data_path+'Dataset_Emails.xlsx')
test_data = preprocess(test_data,data_path)

print(data.shape,test_data.shape)

with open(data_path+'test_all_VP.pickle','rb') as f:
    VP_data = pickle.load(f)
    
with open(data_path+'test_all_tasks.pickle','rb') as f:
    tasks = pickle.load(f)
    
with open(data_path+'test_all_context.pickle','rb') as f:
    context = pickle.load(f)

print(len(VP_data),len(context),len(tasks))

with open(data_path+'test_tasks.pickle','rb') as f:
  test_tasks = pickle.load(f)

with open(data_path+'test_VP.pickle','rb') as f:
  test_VP = pickle.load(f)

with open(data_path+'test_context.pickle','rb') as f:
  test_context = pickle.load(f)

with (tf.device('/CPU:0')):
    token_ids,masks,segments = prepare_sentences(data)
    test_token_ids,test_masks,test_segments = prepare_sentences(test_data)

    print('Initializing')
    train_VP = VP_data
    train_tasks = tasks
    train_context = context

    X_train_ids,X_train_masks,X_train_segments,X_test_ids,X_test_masks,X_test_segments = create_features(train_VP,test_VP,train_context,test_context,
                                                                                    token_ids,test_token_ids,masks,test_masks,segments,test_segments)

    subset = []
    negative_indices = [x for x in range(len(train_tasks)) if train_tasks[x] == 0]

    positive_indices = [x for x in range(len(train_tasks)) if train_tasks[x] == 1]
    subset_size = len(positive_indices)

    subset = sorted(random.sample(negative_indices,subset_size) + positive_indices)
    print(subset[:100])
    print(len(subset))

    X_train_ids = X_train_ids[subset,:]
    X_train_masks = X_train_masks[subset,:]
    X_train_segments = X_train_segments[subset,:]
    train_tasks = [train_tasks[x] for x in subset]

with open(data_path+'X_train_ids.pickle','wb') as f:
  pickle.dump(X_train_ids,f)

with open(data_path+'X_train_masks.pickle','wb') as f:
  pickle.dump(X_train_masks,f)

with open(data_path+'X_train_segments.pickle','wb') as f:
  pickle.dump(X_train_segments,f)

with open(data_path+'X_test_ids.pickle','wb') as f:
  pickle.dump(X_test_ids,f)

with open(data_path+'X_test_masks.pickle','wb') as f:
  pickle.dump(X_test_masks,f)

with open(data_path+'X_test_segments.pickle','wb') as f:
  pickle.dump(X_test_segments,f)

with open(data_path+'subset.pickle','wb') as f:
  pickle.dump(subset,f)

with open(data_path+'train_tasks.pickle','wb') as f:
  pickle.dump(train_tasks,f)