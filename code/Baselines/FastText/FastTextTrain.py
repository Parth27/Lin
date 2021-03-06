import os

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import KFold

from code.DataProcessing import create_split, processData


class FasttextTrainer:
    def __call__(self,dataset='email',num_epochs = 25,lr = 1.0):
        data = pd.read_excel('data/Preprocessed_Dataset_'+dataset+'.xlsx')
        VP_data, VP_df, tasks, context = processData(data, dataset)

        # 5-fold cross validation
        kf = KFold(n_splits=5, shuffle=False)
        f1_scores = []
        accuracies = []
        precisions = []
        recalls = []
        best_f1 = 0
        best_model = None
        fold_num = 1
        for train_idx, test_idx in kf.split(VP_data, tasks):
            print("========= Fold Number: {} ==========".format(fold_num))
            train_VP, test_VP = create_split(train_idx, test_idx, VP_data)
            train_context, test_context = create_split(
                train_idx, test_idx, context)
            train_tasks, test_tasks = create_split(train_idx, test_idx, tasks)

            # Create training and testing data in format required by fasttext
            with open('data/fasttext.train', 'wt') as f:
                for i, VP in enumerate(train_VP[:-1]):
                    f.write(
                        '__label__'+str(train_tasks[i])+' '+VP.strip()+' '+data.iloc[train_context[i]]['Sentence'])
                    f.write('\n')
                f.write('__label__' +
                        str(train_tasks[-1])+' '+train_VP[-1].strip())

            with open('data/fasttext.test', 'wt') as f:
                for i, VP in enumerate(test_VP[:-1]):
                    f.write(
                        '__label__'+str(test_tasks[i])+' '+VP.strip()+' '+data.iloc[test_context[i]]['Sentence'])
                    f.write('\n')
                f.write('__label__' +
                        str(test_tasks[-1])+' '+test_VP[-1].strip())

            # Train model
            model = fasttext.train_supervised(
                input="data/fasttext.train", epoch=num_epochs, lr=lr, loss='softmax', wordNgrams=2)
            # Get predictions
            predictions = []
            for i in range(len(test_VP)):
                print(test_VP[i], str(data.iloc[test_context[i]]['Sentence']))
                predictions.append(model.predict(
                    test_VP[i]+' '+str(data.iloc[test_context[i]]['Sentence']))[0][0].split('__')[-1])

            predictions = [int(x) for x in predictions]
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_tasks, predictions, average='binary')
            accuracy = accuracy_score(test_tasks, predictions)

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
            fold_num += 1
            os.remove('data/fasttext.train')
            os.remove('data/fasttext.test')
            
        print('F1 scores: {}, average: {}'.format(f1_scores, sum(f1_scores)/5))
        print('Precisions: {}, average: {}'.format(
            precisions, sum(precisions)/5))
        print('Recalls: {}, average: {}'.format(recalls, sum(recalls)/5))
        print('Accuracies: {}, average: {}'.format(
            accuracies, sum(accuracies)/5))
        return best_model, f1_scores, accuracies, precisions, recalls
