import tensorflow as tf
import pandas as pd
import os
import pickle
import bert
from bert import tokenization
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import numpy as np
import pickle
import os
import sys
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random

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

  return token_ids,masks,segments

def create_features(train_VP,test_VP,train_context,test_context,token_ids,masks,segments,max_seq_length=None):
  VP_token_ids,VP_masks,VP_segments,test_VP_token_ids,test_VP_masks,test_VP_segments = prepare_data(train_VP,test_VP,max_seq_length)
  context_ids = np.array([token_ids[x] for x in train_context])
  context_masks = np.array([masks[x] for x in train_context])
  context_segments = np.array([segments[x] for x in train_context])

  X_train_ids = np.concatenate((VP_token_ids,context_ids),axis=1)
  X_train_masks = np.concatenate((VP_masks,context_masks),axis=1)
  X_train_segments = np.concatenate((VP_segments,context_segments),axis=1)

  test_context_ids = np.array([token_ids[x] for x in test_context])
  test_context_masks = np.array([masks[x] for x in test_context])
  test_context_segments = np.array([segments[x] for x in test_context])

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

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

def create_split(train_idx,test_idx,data):
    train_data = [data[x] for x in train_idx]
    test_data = [data[x] for x in test_idx]
    return train_data,test_data

if __name__=='__main__':
  data_path = '../Datasets/'
  tf.compat.v1.reset_default_graph()
  random.seed(0)
  vocab_file = data_path+'bert_vocab.txt'
  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
  dataset = sys.argv[1]
  data = pd.read_excel(data_path+dataset+'_preprocessed_dataset.xlsx')

  with open(data_path+dataset+'_all_VP.pickle','rb') as f:
      VP_data = pickle.load(f)
      
  with open(data_path+dataset+'_all_tasks.pickle','rb') as f:
      tasks = pickle.load(f)
      
  with open(data_path+dataset+'_all_context.pickle','rb') as f:
      context = pickle.load(f)

  shuffled_idx = [x for x in range(len(VP_data))]
  random.shuffle(shuffled_idx)

  VP_data = [VP_data[x] for x in shuffled_idx]
  tasks = [tasks[x] for x in shuffled_idx]
  context = [context[x] for x in shuffled_idx]

  if dataset.lower() == 'chat':
    model = create_model(331)
    try:
        model.load_weights(data_path+'Bert_model_weights.h5')
    except:
        print('BERT model not trained yet')
        sys.exit()

    print('Loaded model successfully')
    print('Evaluating')
    token_ids,masks,segments = prepare_sentences(data)
    _,_,_,X_test_ids,X_test_masks,X_test_segments = create_features(VP_data,VP_data,context,context,token_ids,masks,segments,31)

    preds = model.predict([X_test_ids,X_test_masks,X_test_segments])

    preds = preds.tolist()
    chat_predictions = [1 if x[0]>=0.5 else 0 for x in preds]
    precision,recall,f1,_ = precision_recall_fscore_support(tasks,chat_predictions,average='binary')
    accuracy = accuracy_score(tasks,chat_predictions)
    print('BERT chat F1 score: {}'.format(f1))
    print('BERT chat Precision: {}'.format(precision))
    print('BERT chat Recall: {}'.format(recall))
    print('BERT chat Accuracy: {}'.format(accuracy))
    sys.exit()

  kf = KFold(n_splits=5,shuffle=False)
  token_ids,masks,segments = prepare_sentences(data)
  f1_scores = []
  accuracies = []
  precisions = []
  recalls = []
  best_f1 = 0
  best_model = None
  print('Initializing')
  for train_idx,test_idx in kf.split(VP_data,tasks):
      train_VP,test_VP = create_split(train_idx,test_idx,VP_data)
      train_context,test_context = create_split(train_idx,test_idx,context)
      train_tasks,test_tasks = create_split(train_idx,test_idx,tasks)

      X_train_ids,X_train_masks,X_train_segments,X_test_ids,X_test_masks,X_test_segments = create_features(train_VP,test_VP,train_context,test_context,token_ids,masks,segments)
      model = create_model(X_train_ids.shape[1])

      sess = tf.Session()
      initialize_vars(sess)

      class_weights = {0:1,1:int(len([x for x in train_tasks if x==0])/len([x for x in train_tasks if x==1]))}

      model.fit(
          [X_train_ids, X_train_masks, X_train_segments], 
          train_tasks,
          validation_data=([X_test_ids, X_test_masks, X_test_segments], test_tasks),
          epochs=20,
          batch_size=32,
          class_weight=class_weights
      )

      preds = model.predict([X_test_ids,X_test_masks,X_test_segments])

      preds = preds.tolist()
      preds = [1 if x[0]>=0.5 else 0 for x in preds]

      precision,recall,f1,_ = precision_recall_fscore_support(test_tasks,preds,average='binary')
      accuracy = accuracy_score(test_tasks,preds)

      precisions.append(precision)
      recalls.append(recall)
      f1_scores.append(f1)
      accuracies.append(accuracy)
      if f1>best_f1:
          best_f1 = f1
          best_model = model
      
      print('F1 scores: {}, average: {}'.format(f1_scores,sum(f1_scores)/5))
      print('Precisions: {}, average: {}'.format(precisions,sum(precisions)/5))
      print('Recalls: {}, average: {}'.format(recalls,sum(recalls)/5))
      print('Accuracies: {}, average: {}'.format(accuracies,sum(accuracies)/5))