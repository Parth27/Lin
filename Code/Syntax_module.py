import numpy as np
import pandas as pd
import nltk
import re
import pickle
import stanfordnlp
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
import sys

#Find descendants in DFS manner
def dfs(graph,node,arr,modify_dependencies = False,nsubj='',dependencies={}):
    #Pass nsubj to descendants who do not have it
    if modify_dependencies:
        dep = dependencies[node].get('nsubj',[])
        if not dep:
            dependencies[node]['nsubj'] = [nsubj]
        else:
            nsubj = dep[0]
    if node in graph.keys():
        for n in graph[node]:
            dfs(graph,n,arr,modify_dependencies,nsubj,dependencies)
            arr.append(n)
    return arr,dependencies

def bfs(graph,rootNode):    
    ans = []
    queue = [rootNode]
    while(queue):
        node = queue.pop(0)
        ans.append(node)
        if node in graph:
            queue.extend(graph[node])
        
    return ans

#Function to skip descendants
def skip_descendants(graph,idx,skip_words,dependencies,skip_conj = False,skip_comp = False):
    arr,_ = dfs(graph,idx,[])
    #Whether or not to skip conjunctions
    if not skip_conj:
        conj = [int(x.index) for x in dependencies[idx].get('conj',[])]
        arr = [x for x in arr if x not in conj]
    #Whether or not to skip clausal complements
    if not skip_comp:
        clausal_comp = [int(x.index) for x in dependencies[idx].get('xcomp',[]) if 'VB' in x.xpos]
        clausal_comp += [int(x.index) for x in dependencies[idx].get('ccomp',[]) if 'VB' in x.xpos]
        arr = [x for x in arr if x not in clausal_comp]
        
    skip_words.extend(arr)
    
    return skip_words

def check_subject(dependencies,idx):
    subj = [x.text.lower() if x!='' else '' for x in dependencies[idx].get('nsubj',[''])]
    valid_subjects = ['you','us','we','i','']
    subj_check = any(x in subj for x in valid_subjects)
    return subj_check

def extract_tasks(data):
    tasks = {}
    purpose = {}
    verbs = {}
    question_words = ['why','how','what','which','whose','who','whom','where','whether']
    for i in range(data.shape[0]):
        sent = str(data.iloc[i]['Sentence'])
        tasks[i] = []
        purpose[i] = []
        verbs[i] = []
        position = {}
        if sent == '':
            continue
        result = nlp(sent)
        j = 0
        
        for sentence in result.sentences:
            reverse_graph = {}
            dependencies = {}
            skip_words = []
            verb_ids = []
            
            for parse in sentence.words:
                position[int(parse.index)] = j
                j += 1
                reverse_graph.setdefault(int(parse.governor),[])

                reverse_graph[int(parse.governor)].append(int(parse.index))
                    
                if int(parse.governor) == 0:
                    rootNode = int(parse.index)
                
                dependencies.setdefault(int(parse.governor),{})
                dependencies.setdefault(int(parse.index),{})
                dependencies[int(parse.governor)].setdefault(parse.dependency_relation,[])
                dependencies[int(parse.governor)][parse.dependency_relation].append(parse)
                
            arr = bfs(reverse_graph,rootNode)
            parses = [sentence.words[y-1] for y in arr if y != 0]
            
            _,dependencies = dfs(reverse_graph,int(parses[0].index),[],True,'',dependencies)
            n = 0
            while(n<len(parses)):
                parse = parses[n]
                idx = int(parse.index)
                if idx in skip_words:
                    n += 1
                    continue

                #Skip questions
                advmod = [x.text.lower() for x in dependencies[idx].get('advmod',[])]
                question_check = any(item in advmod for item in question_words)
                if question_check:
                    skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,
                                                  skip_conj=True,skip_comp=True)
                    n += 1
                    continue

                #Skip descendants if root or current node is adjective or noun or past verbs
                if parse.xpos in ('JJ','JJR','JJS','NN','NNS','NNP','NNPS','VBN','VBD','VBZ'):
                    skip = True
                    if parse.xpos == 'VBN':
                        aux_pass = [x.xpos for x in dependencies[idx].get('aux:pass',[])]
                        if 'VB' in aux_pass:
                            skip = False
                            
                    if parse.xpos == 'JJ':
                        skip_comp = False
                    else:
                        skip_comp = True

                    if skip == True:
                        skip_words = skip_descendants(reverse_graph,idx,skip_words,
                                                      dependencies,skip_comp=skip_comp)
                    n += 1
                    continue

                aux_xpos = [x.xpos for x in dependencies[idx].get('aux',[])]
                past_check = any(item in aux_xpos for item in ['VBN','VBD'])
                if past_check:
                    skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,
                                                  skip_conj=True,skip_comp=True)
                    n += 1
                    continue

                if parse.xpos in ('VB','VBG','VBP','VBN') and idx not in skip_words:
                    if parse.xpos=='VBG':
                        #Skip descendants if VBG has a non-'VB' aux
                        if 'aux' not in dependencies[idx]:
                            pass

                        elif 'VB' not in aux_xpos and 'VBP' not in aux_xpos:
                            skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,
                                                          skip_conj=True,skip_comp=True)
                            n += 1
                            continue

                    #Subject check
                    subj_check = check_subject(dependencies,idx)
                    if not subj_check:
                        n += 1
                        continue

                    #Jump to the clausal complement of verb
                    clausal_comp = [x for x in dependencies[idx].get('xcomp',[]) if 'VB' in x.xpos]
                    clausal_comp += [x for x in dependencies[idx].get('ccomp',[]) if 'VB' in x.xpos]

                    if clausal_comp:
                        skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies)
                        n += 1
                        continue
                    
                    #Fetch verb object
                    obj = dependencies[int(parse.index)].get('obj',None)
                    task_obj = ''
                    if obj:
                        task_obj = obj[0].text


                    tasks[i].append((parse.text,task_obj))
                    verbs[i].append(parse.text)
                    verb_ids.append(idx)
                    skip_comp = True
                    skip_words = skip_descendants(reverse_graph,idx,skip_words,
                                                    dependencies,skip_comp=skip_comp)

                n += 1
        
    return tasks,verbs

#Function to link the predicted verbs to our Verb Phrases
def get_predictions(VP_df,data,predicted_verbs):
    predictions = [[0 for x in range(len(VP_df[i]))] for i in range(len(VP_df))]
    
    for i in range(len(VP_df)):
        if pd.isnull(data.iloc[i]['Verbs']):
            pass
        else:
            label = [x.strip() for x in str(data.iloc[i]['Task/Goal']).split(',')]
            valid_labels = [x for x in range(len(label)) if label[x] == 'Task']

            verbs = data.iloc[i]['Verbs'].replace('[','').replace(']','').replace("'","").split(',')
            verbs = [verbs[x].strip().lower() for x in valid_labels]
            
            visited = dict([(x,False) for x in range(len(VP_df[i]))])
            for x in range(len(predicted_verbs[i])):
                for y in range(len(VP_df[i])):
                    VP = VP_df[i][y]
                    if visited[y] == True:
                        continue
                    if predicted_verbs[i][x].lower() == VP[1].lower():
                        visited[y] = True
                        predictions[i][y] = 1

    predictions = [item for sublist in predictions for item in sublist]

    print(len(predictions))
    return predictions

def evaluate_model(y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    precision,recall,f1,_ = precision_recall_fscore_support(y_true,y_pred,average='binary')
    return accuracy,precision,recall,f1

if __name__=='__main__':
    data_path = '../Datasets/'

    nlp = stanfordnlp.Pipeline(use_gpu=False)
    dataset = sys.argv[1]
    data = pd.read_excel(data_path+dataset+'_preprocessed_dataset.xlsx')

    with open(data_path+dataset+'_all_VP.pickle','rb') as f:
        VP_data = pickle.load(f)
    
    with open(data_path+dataset+'_all_VP_df.pickle','rb') as f:
        VP_df = pickle.load(f)
        
    with open(data_path+dataset+'_all_tasks.pickle','rb') as f:
        tasks = pickle.load(f)
        
    with open(data_path+dataset+'_all_context.pickle','rb') as f:
        context = pickle.load(f)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        predicted_tasks,predicted_verbs = extract_tasks(data)

    predictions = get_predictions(VP_df,data,predicted_verbs)
    accuracy,precision,recall,f1 = evaluate_model(tasks,predictions)
    
    print('F1 score: {}'.format(f1))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('Accuracy: {}'.format(accuracy))

    #Save Predictions
    with open(data_path+dataset+'_Syntax_predictions.pickle','wb') as f:
        pickle.dump(predictions,f)