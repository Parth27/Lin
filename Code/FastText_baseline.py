import numpy as np
import pandas as pd
import re
import pickle
import os
import fasttext
import sys
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def shuffle_data(VP_data,tasks,context):
    random.seed(0)
    shuffled_idx = [x for x in range(len(VP_data))]
    random.shuffle(shuffled_idx)

    VP_data = [VP_data[x] for x in shuffled_idx]
    tasks = [tasks[x] for x in shuffled_idx]
    context = [context[x] for x in shuffled_idx]
    return VP_data,tasks,context


def create_split(train_idx,test_idx,data):
  train_data = [data[x] for x in train_idx]
  test_data = [data[x] for x in test_idx]
  return train_data,test_data


if __name__=='__main__':
    data_path = '../Datasets/'
    dataset = sys.argv[1]
    #Read data
    data = pd.read_excel(data_path+dataset+'_preprocessed_dataset.xlsx')

    with open(data_path+dataset+'_all_VP.pickle','rb') as f:
        VP_data = pickle.load(f)
        
    with open(data_path+dataset+'_all_tasks.pickle','rb') as f:
        tasks = pickle.load(f)
        
    with open(data_path+dataset+'_all_context.pickle','rb') as f:
        context = pickle.load(f)

    if dataset == 'chat':
        model = fasttext.load_model(data_path+'fasttext_model.bin')
        chat_predictions = []
        for i in range(len(VP_data)):
            chat_predictions.append(best_model.predict(VP_data[i]+' '+str(data.iloc[context[i]]['Sentence']))[0][0].split('__')[-1])

        chat_predictions = [int(x) for x in chat_predictions]
        precision,recall,f1,_ = precision_recall_fscore_support(tasks,chat_predictions,average='binary')
        accuracy = accuracy_score(tasks,chat_predictions)

        print('F1 score: {}'.format(f1))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('Accuracy: {}'.format(accuracy))
        sys.exit()

    VP_data,tasks,context = shuffle_data(VP_data,tasks,context)

    #5-fold cross validation
    kf = KFold(n_splits=5,shuffle=False)
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    best_f1 = 0
    best_model = None

    for train_idx,test_idx in kf.split(VP_data,tasks):
        train_VP,test_VP = create_split(train_idx,test_idx,VP_data)
        train_context,test_context = create_split(train_idx,test_idx,context)
        train_tasks,test_tasks = create_split(train_idx,test_idx,tasks)

        #Create training and testing data in format required by fasttext
        with open(data_path+'fasttext.train','wt') as f:
            for i,VP in enumerate(train_VP[:-1]):
                f.write('__label__'+str(train_tasks[i])+' '+VP.strip()+' '+data.iloc[train_context[i]]['Sentence'])
                f.write('\n')
            f.write('__label__'+str(train_tasks[-1])+' '+train_VP[-1].strip())

        with open(data_path+'fasttext.test','wt') as f:
            for i,VP in enumerate(test_VP[:-1]):
                f.write('__label__'+str(test_tasks[i])+' '+VP.strip()+' '+data.iloc[test_context[i]]['Sentence'])
                f.write('\n')
            f.write('__label__'+str(test_tasks[-1])+' '+test_VP[-1].strip())

        #Train model
        model = fasttext.train_supervised(input=data_path+"fasttext.train",epoch=25,lr=1.0,loss='softmax',wordNgrams=2)
        #Get predictions
        predictions = []
        for i in range(len(test_VP)):
            print(test_VP[i],str(data.iloc[test_context[i]]['Sentence']))
            predictions.append(model.predict(test_VP[i]+' '+str(data.iloc[test_context[i]]['Sentence']))[0][0].split('__')[-1])

        predictions = [int(x) for x in predictions]
        precision,recall,f1,_ = precision_recall_fscore_support(test_tasks,predictions,average='binary')
        accuracy = accuracy_score(test_tasks,predictions)
        
        #Save best model
        if f1>best_f1:
            best_f1 = f1
            best_model = model

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

        best_model.save_model(data_path+"fasttext_model.bin")

        print('F1 scores: {}, average: {}'.format(f1_scores,sum(f1_scores)/5))
        print('Precisions: {}, average: {}'.format(precisions,sum(precisions)/5))
        print('Recalls: {}, average: {}'.format(recalls,sum(recalls)/5))
        print('Accuracies: {}, average: {}'.format(accuracies,sum(accuracies)/5))