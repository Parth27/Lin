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
from collections import Counter
from nltk.corpus import stopwords
import sys

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

    for key in ('email','Email','EMAIL'):
        data['Sentence'] = data['Sentence'].str.replace(key,contractions['email'])

    return data

#Function to traverse dependency parse in BFS manner
def bfs(graph,rootNode):    
    ans = []
    queue = [rootNode]
    while(queue):
        node = queue.pop(0)
        ans.append(node)
        if node in graph:
            queue.extend(graph[node])
        
    return ans

#Function to extract VPs from dataset
def extract_VP(data):
    nlp = stanfordnlp.Pipeline(use_gpu=False)
    VP_data = []
    for i in range(data.shape[0]):
        Verb_Phrases = []
        main_verb = []
        if str(data.iloc[i]['Sentence']) == '':
            VP_data.append([])
            continue
        result = nlp(data.iloc[i]['Sentence'])
        for sentence in result.sentences:
            sent = sentence.words
            reverse_graph = {}
            dependencies = {}
            for parse in sentence.words:
                reverse_graph.setdefault(int(parse.governor),[])
                reverse_graph[int(parse.governor)].append(int(parse.index))
                reverse_graph.setdefault(int(parse.index),[])
                #Find root node
                if int(parse.governor) == 0:
                    rootNode = int(parse.index)
                
                dependencies.setdefault(int(parse.governor),{})
                dependencies.setdefault(int(parse.index),{})
                dependencies[int(parse.governor)].setdefault(parse.dependency_relation,[])
                dependencies[int(parse.governor)][parse.dependency_relation].append(parse)
                
            arr = bfs(reverse_graph,rootNode)
            parses = [sentence.words[y-1] for y in arr if y != 0]
            n = 0
            while(n<len(parses)):
                parse = parses[n]
                idx = int(parse.index)
                if 'VB' in parse.xpos:
                    #Get immediate descendants of verb
                    dependents = reverse_graph[idx]
                    dependents.append(idx)
                    Verb_Phrases.append(' '.join([sent[x].text for x in range(len(sent)) if int(sent[x].index) in dependents]))
                    main_verb.append(parse.text)
                n += 1
        VP_data.append(list(zip(Verb_Phrases,main_verb)))
        
        if i%500 == 0 and VP_data:
            print(i)
            print(VP_data[-1])
    return VP_data

#Function to separate out VPs and their respective annotated tasks
def get_tasks_for_VP(VP_data,data):
    tasks = [[0 for x in range(len(VP_data[i]))] for i in range(len(VP_data))]
    for i in range(len(VP_data)):
        if pd.isnull(data.iloc[i]['Verbs']):
            pass
        else:
            labels = data.iloc[i]['Task/Goal'].replace('[','').replace(']','').replace("'","")
            label = [x.strip() for x in labels.split(',')]
            #Only fetch words that are labeled as Task
            valid_labels = [x for x in range(len(label)) if label[x] == 'Task']

            verbs = data.iloc[i]['Verbs'].replace('[','').replace(']','').replace("'","").split(',')
            verbs = [verbs[x].strip().lower() for x in valid_labels]
            visited = dict([(x,False) for x in range(len(VP_data[i]))])
            for x in range(len(verbs)):
                for y in range(len(VP_data[i])):
                    VP = VP_data[i][y]
                    if visited[y] == True:
                        continue
                    if verbs[x] == VP[1].lower():
                        visited[y] = True
                        tasks[i][y] = 1

    tasks = [item for sublist in tasks for item in sublist]
    new_VP = []
    context = []
    for i in range(len(VP_data)):
        new_VP.extend([x[0] for x in VP_data[i]])
        context.extend([i for x in VP_data[i]])

    return tasks,new_VP,context

if __name__=='__main__':
    dataset = sys.argv[1]
    data_path = '../Datasets/'
    #Read selected dataset
    if dataset.lower() == 'email':
        data = pd.read_excel(data_path+'Dataset_Emails.xlsx')
    elif dataset.lower() == 'chat':
        data = pd.read_excel(data_path+'Dataset_Chat.xlsx')
    else:
        print('Incorrect dataset name, please enter "email" or "chat"')
        sys.exit()

    data = preprocess(data,data_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        VP_data = extract_VP(data)

    tasks,new_VP,context = get_tasks_for_VP(VP_data,data)

    #Save VPs and preprocessed dataset
    with open(data_path+dataset+'_all_tasks.pickle','wb') as f:
        pickle.dump(tasks,f)
        
    with open(data_path+dataset+'_all_VP.pickle','wb') as f:
        pickle.dump(new_VP,f)

    with open(data_path+dataset+'_all_VP_df.pickle','wb') as f:
        pickle.dump(VP_data,f)
        
    with open(data_path+dataset+'_all_context.pickle','wb') as f:
        pickle.dump(context,f)

    data.to_excel(data_path+dataset+'_preprocessed_dataset.xlsx',index=False)