import argparse
import json
import pickle
import random
import re
import sys
import warnings

import nltk
import numpy as np
import pandas as pd
import stanfordnlp
from nltk.corpus import verbnet
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import KFold

from code.DataProcessing import extract_VP, get_tasks_for_VP, preprocess
from code.Lin import lin

if __name__ == '__main__':
    parser = argparse.ArgumentParser('main', description='Run Lin')
    parser.add_argument('--dataset', required=True,
                        action='store', help='Choose the dataset')
    args = parser.parse_args()

    with open('config/Rules.json') as f:
        rules = json.load(f)
    themePredicates = rules['themePredicates']
    agentPredicates = rules['agentPredicates']
    compSemantics = rules['compSemantics']

    if args.dataset.lower() in ('email', 'chat'):
        data = pd.read_excel('data/Preprocessed_Dataset_' +
                             args.dataset.lower()+'.xlsx')
    else:
        try:
            data = pd.read_excel(args.dataset)
        except:
            print("Dataset not found")
            sys.exit(0)

    model = lin(themePredicates, agentPredicates, compSemantics)
    data['Sentence'] = data['Sentence'].astype(str)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        predicted_tasks, predicted_verbs = model.extractTasks(data)
        print('Task extraction complete')

    if args.dataset.lower() in ('email', 'chat'):
        accuracy, precision, recall, f1 = model.evaluate(predicted_verbs, data)
        print(accuracy, precision, recall, f1)
