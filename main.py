import numpy as np
import pandas as pd
import nltk
import re
import pickle
import stanfordnlp
from nltk.corpus import verbnet
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
from nltk.stem.snowball import SnowballStemmer
import sys
from Code.DataProcessing import get_tasks_for_VP,extract_VP,preprocess
from Code.Lin import lin
import json


with open('Config/Rules.json') as f:
    rules = json.load(f)

themePredicates = rules['themePredicates']
agentPredicates = rules['agentPredicates']
compSemantics = rules['compSemantics']

model = lin(themePredicates,agentPredicates,compSemantics)
data = pd.read_excel('Data/Preprocessed_Dataset_email.xlsx')
data['Sentence'] = data['Sentence'].astype(str)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    predicted_tasks,predicted_verbs = model.extractTasks(data)
    print('Task extraction complete')

accuracy,precision,recall,f1 = model.evaluate(predicted_verbs,data)
print(accuracy,precision,recall,f1)