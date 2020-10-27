import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import stanfordnlp
from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn
import warnings
from sklearn.feature_selection import chi2,SelectKBest
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from sklearn.svm import SVC
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
import sys

#Create and train SVM model
def create_train_model(X_train,y_train):
    svm = SVC(class_weight='balanced')
    svm.fit(X_train,y_train)
    return svm

#Create Kfolds for cross validation
def create_kfold(VP_data,tasks,context):
    shuffled_idx = [x for x in range(len(VP_data))]
    random.seed(0)
    random.shuffle(shuffled_idx)

    VP_data = [VP_data[x] for x in shuffled_idx]
    tasks = [tasks[x] for x in shuffled_idx]
    context = [context[x] for x in shuffled_idx]

    print(len(context),len(tasks),len(VP_data))
    kf = KFold(n_splits=5,shuffle=False)
    return kf,VP_data,tasks,context

#Create Sentence embeddings using Universal Sentence Encoder
def prep_data(data,train_VP,test_VP,train_context,test_context):
    module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    with tf.device('/cpu:0'):
        embed = hub.KerasLayer(module_url)
        #Create Sentense embeddings
        X_train = embed(train_VP).numpy()
        X_test = embed(test_VP).numpy()
        
        sentence_embeds = embed(data['Sentence'].astype(str)).numpy()
        context_embeds = np.array([sentence_embeds[x].reshape((128,)) for x in train_context])
        print(context_embeds.shape)
        
        X_train = np.concatenate((X_train,context_embeds),axis=1)
        print(X_train.shape)
        
        test_context_embeds = np.array([sentence_embeds[x].reshape((128,)) for x in test_context])
        print(test_context_embeds.shape)
        
        X_test = np.concatenate((X_test,test_context_embeds),axis=1)
        print(X_test.shape)
    
    return X_train,X_test

#Split data based on training and test indexes
def split_data(train_idx,test_idx,data):
    train_data = [data[x] for x in train_idx]
    test_data = [data[x] for x in test_idx]
    return train_data,test_data

#Get predictions of model
def get_predictions(model,X_test):
    return model.predict(X_test)

#Evaluate model
def evaluate_model(y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    precision,recall,f1,_ = precision_recall_fscore_support(y_true,y_pred,average='binary')
    return accuracy,precision,recall,f1

if __name__ == '__main__':
    dataset = sys.argv[1]
    #Read data and VPs for selected dataset
    data_path = './Datasets/'
    data = pd.read_excel(data_path+dataset+'_preprocessed_dataset.xlsx')

    with open(data_path+dataset+'_all_VP.pickle','rb') as f:
        VP_data = pickle.load(f)
        
    with open(data_path+dataset+'_all_tasks.pickle','rb') as f:
        tasks = pickle.load(f)
        
    with open(data_path+dataset+'_all_context.pickle','rb') as f:
        context = pickle.load(f)

    if dataset.lower() == 'chat':
        #Load trained SVM model and evaluate on chat dataset
        try:
            with open(data_path+'svm_model.sav','rb') as f:
                svm = pickle.load(f)
        except:
            print('SVM model not trained yet')
            sys.exit()
        X_test,_ = prep_data(data,VP_data,VP_data,context,context)
        predictions = svm.predict(X_test)
        accuracy,precision,recall,f1 = evaluate_model(tasks,predictions)
        print('SVM chat F1 score: {}'.format(f1))
        print('SVM chat Precision: {}'.format(precision))
        print('SVM chat Recall: {}'.format(recall))
        print('SVM chat Accuracy: {}'.format(accuracy))
        #Save chat predictions
        with open(data_path+dataset+'_SVM_predictions.pickle','wb') as f:
            pickle.dump(predictions,f)
        sys.exit()

    kf,VP_data,tasks,context = create_kfold(VP_data,tasks,context)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    best_f1 = 0
    best_model = None

    #Train using 5-fold cross validation
    for train_idx,test_idx in kf.split(VP_data,tasks):
        train_VP,test_VP = split_data(train_idx,test_idx,VP_data)
        train_tasks,test_tasks = split_data(train_idx,test_idx,tasks)
        train_context,test_context = split_data(train_idx,test_idx,context)

        X_train,X_test = prep_data(data,train_VP,test_VP,train_context,test_context)
        svm = create_train_model(X_train,train_tasks)
        predictions = get_predictions(svm,X_test)
        accuracy,precision,recall,f1 = evaluate_model(test_tasks,predictions)
        #Save best model based on f1 score
        if f1>best_f1:
            best_f1 = f1
            best_Model = svm

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    #Save trained model
    with open(data_path+'svm_model.sav','wb') as f:
        pickle.dump(best_Model,f)

    #Print average results of 5-fold cross validation
    print('Average F-1 score: {}'.format(sum(f1_scores)/5))
    print('Average Precision: {}'.format(sum(precisions)/5))
    print('Average Recall: {}'.format(sum(recalls)/5))
    print('Average Accuracy: {}'.format(sum(accuracies)/5))