import pandas as pd
import os
import pickle
import numpy as np
import sys
import random
import re
from transformers import BertTokenizer


def prepare_data(tokenizer,X_train,X_test,max_seq_length=None):
  X_tokens = []
  for i in range(len(X_train)):
    currTokens = tokenizer.encode(X_train[i],add_special_tokens=True)
    if max_seq_length and len(currTokens) > max_seq_length:
      currTokens = currTokens[:max_seq_length-1]+tokenizer.encode(['SEP'],add_special_tokens=False)
    X_tokens.append(currTokens)

  X_masks = list(map(lambda tok: [1]*len(tok), X_tokens))

  if not max_seq_length:
    max_seq_length = max([len(x) for x in X_tokens])

  X_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), X_tokens)
  X_masks = map(lambda masks: masks + [0] * (max_seq_length - len(masks)), X_masks)
  X_segments = map(lambda tokens: [0]*max_seq_length, X_tokens)

  X_token_ids = np.array(list(X_token_ids))
  X_masks = np.array(list(X_masks))
  X_segments = np.array(list(X_segments))

  test_X_tokens = []
  for i in range(len(X_test)):
    currTokens = tokenizer.encode(X_test[i],add_special_tokens=True)
    if max_seq_length and len(currTokens) > max_seq_length:
      currTokens = currTokens[:max_seq_length-1]+tokenizer.encode(['SEP'],add_special_tokens=False)
    test_X_tokens.append(currTokens)

  test_X_masks = list(map(lambda tok: [1]*len(tok), test_X_tokens))
  test_X_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), test_X_tokens)
  test_X_masks = map(lambda masks: masks + [0] * (max_seq_length - len(masks)), test_X_masks)
  test_X_segments = map(lambda tokens: [0]*max_seq_length, test_X_tokens)

  test_X_token_ids = np.array(list(test_X_token_ids))
  test_X_masks = np.array(list(test_X_masks))
  test_X_segments = np.array(list(test_X_segments))
  return X_token_ids,X_masks,X_segments,test_X_token_ids,test_X_masks,test_X_segments