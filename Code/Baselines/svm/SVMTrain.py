import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from Code.DataProcessing import create_split, processData


class SVMTrainer:
    def __init__(self, dataset='email'):
        self.dataset = dataset
        self.data = pd.read_excel(
            'Data/Preprocessed_Dataset_'+self.dataset+'.xlsx')
        module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
        with tf.device('/cpu:0'):
            self.embed = hub.KerasLayer(module_url)

    def __call__(self):
        VP_data, tasks, context = processData(self.data, self.dataset)
        # K-fold cross validation
        kf = KFold(n_splits=5, shuffle=False)
        print('Started Training...')
        f1_scores = []
        accuracies = []
        precisions = []
        recalls = []
        best_model = None
        best_f1 = 0
        fold_num = 1
        for train_idx, test_idx in kf.split(VP_data, tasks):
            print("========= Fold Number: {} ==========".format(fold_num))
            train_VP, test_VP = create_split(train_idx, test_idx, VP_data)
            train_context, test_context = create_split(
                train_idx, test_idx, context)
            train_tasks, test_tasks = create_split(
                train_idx, test_idx, tasks)

            X_train, X_test = self.prep_data(
                train_VP, test_VP, train_context, test_context)
            model = SVC(class_weight='balanced')
            model.fit(X_train, train_tasks)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(test_tasks, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_tasks, predictions, average='binary')
            # Save best model based on f1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
            fold_num += 1

        print('F1 scores: {}, average: {}'.format(f1_scores, sum(f1_scores)/5))
        print('Precisions: {}, average: {}'.format(
            precisions, sum(precisions)/5))
        print('Recalls: {}, average: {}'.format(recalls, sum(recalls)/5))
        print('Accuracies: {}, average: {}'.format(
            accuracies, sum(accuracies)/5))
        return best_model, accuracies, f1_scores, precisions, recalls

    # Create Sentence embeddings using Universal Sentence Encoder
    def prep_data(self, train_VP, test_VP, train_context, test_context):
        # Create Sentense embeddings
        X_train = self.embed(train_VP).numpy()
        X_test = self.embed(test_VP).numpy()
        context_embeds = np.array(
            [self.embed(x).numpy().reshape((128,)) for x in train_context])
        X_train = np.concatenate((X_train, context_embeds), axis=1)
        test_context_embeds = np.array(
            [self.embed(x).numpy().reshape((128,)) for x in test_context])
        X_test = np.concatenate((X_test, test_context_embeds), axis=1)
        return X_train, X_test
