import os
import pickle
import random
import re
import warnings

import numpy as np
import pandas as pd
import stanfordnlp

# Preprocess data
def preprocess(data):
    contractions = {}
    data['Sentence'] = data['Sentence'].astype(str)
    # List of contractions from Wikipedia
    with open('/home/parth/Lin/config/Contractions', 'r') as f:
        contractions = dict([x.split(':')
                             for x in f.read().strip().split('\n')])

    keys = contractions.keys()
    for key in list(keys):
        contractions[key] = contractions[key].strip()
        contractions[key[0].upper()+key[1:]
                     ] = contractions[key][0].upper()+contractions[key][1:]

    for key in contractions:
        data['Sentence'] = data['Sentence'].str.replace(key, contractions[key])

    remove = []
    for i in range(data.shape[0]):
        if pd.isnull(data.iloc[i]['Verbs']):
                remove.append(i)
        sent = data.iloc[i]['Sentence']
        data.at[i, 'Sentence'] = ' '.join(
            re.sub(r'"', '', re.sub(r"['#@_%*()</>^&=]*[\-]*", '', sent)).split())

    for key in ('email', 'Email', 'EMAIL'):
        data['Sentence'] = data['Sentence'].str.replace(
            key, contractions['email'])

    return data,remove

data = pd.read_excel('/home/parth/Lin/data/Dataset_email.xlsx')

data,remove = preprocess(data)
# data = data.drop(data.index[remove])
# data.index = list(range(data.shape[0]))
# print(data.shape)
data.to_excel('/home/parth/Lin/data/Preprocessed_Dataset_email.xlsx',index=False)