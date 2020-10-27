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
from Code.Extract_VerbPhrases import get_tasks_for_VP,extract_VP,preprocess

def dfs(graph,node,arr,modify_dependencies = False,nsubj='',dependencies={}):
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

def skip_descendants(graph,idx,skip_words,dependencies,skip_conj = False,skip_comp = False):
    arr,_ = dfs(graph,idx,[])
    if not skip_conj:
        conj = [int(x.index) for x in dependencies[idx].get('conj',[])]
        arr = [x for x in arr if x not in conj]
    
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

#With only Agent and Cause
def parse_frames(classid,found):
    valid_task = False
    agent_flag = False
    try:
        for frame in v3.frames(classid):
            found = True
            for s in frame['semantics']:
                for arg in s['arguments']:
                    if arg['type'] == 'ThemRole' and arg['value'] == 'Agent':
                        agent_flag = True
                        if s['predicate_value'] in agent_predicates:
                            valid_task = True
                            return valid_task,found,agent_flag
                        
            for s in frame['semantics']:
                for arg in s['arguments']:
                    if arg['type'] == 'ThemRole' and arg['value'] == 'Theme':
                        if s['predicate_value'] in theme_predicates:
                            valid_task = True
                            return valid_task,found,True
            
    except:
        pass
    return valid_task,found,agent_flag
    
def parse_action(lemma):
    flag = -1
    found_frames = False
    valid_task = False
    agent_flag = False
    
    if not v3.classids(lemma = lemma):
        flag = 0
        lemma = stemmer2.stem(lemma)
        if not v3.classids(lemma = lemma):
            return True,flag,True
    
    for vn_class in v3.classids(lemma):
        valid_task,found_frames,agent_flag = parse_frames(vn_class,found_frames)
        flag = 1
        if valid_task and found_frames:
            return valid_task,flag,agent_flag

    return valid_task,flag,agent_flag

def parse_for_comp(classid,found):
    comp_check = False
    try:
        for frame in v3.frames(classid):
            found = True
            for s in frame['semantics']:
                for arg in s['arguments']:
                    if arg['type'] == 'Event':
                        if s['predicate_value'] in semantics_for_comps:
                            comp_check = True
                            return comp_check,found
    except:
        pass
    return comp_check,found

def check_comp(parent_lemma,child_lemma):
    parent_task = False
    child_task = False
    found = False
    comp_check = False
    
    if not v3.classids(lemma = parent_lemma):
        parent_lemma = stemmer2.stem(parent_lemma)
        
    for classid in v3.classids(parent_lemma):
        comp_check,found = parse_for_comp(classid,found)    
        if found and comp_check:
            break
    
    if comp_check:
        valid_task,_,_ = parse_action(child_lemma)
        if valid_task:
            return True
        
    return False

def extract_tasks(data):
    tasks = {}
    verbs = {}
    labels = {}
    question_words = ['why','how','what','which','whose','who','whom','where','whether']
    for i in range(data.shape[0]):
        sent = str(data.iloc[i]['Sentence'])
        tasks[i] = []
        verbs[i] = []
        labels[i] = []
        if sent == '':
            continue
        result = nlp(sent)
        
        for sentence in result.sentences:
            reverse_graph = {}
            dependencies = {}
            skip_words = []
            verb_ids = []
            
            for parse in sentence.words:
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
                    skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,skip_conj=True,skip_comp=True)
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
                        skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,skip_comp=skip_comp)
                    n += 1
                    continue

                aux_xpos = [x.xpos for x in dependencies[idx].get('aux',[])]
                past_check = any(item in aux_xpos for item in ['VBN','VBD'])
                if past_check:
                    skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,skip_conj=True,skip_comp=True)
                    n += 1
                    continue

                if parse.xpos in ('VB','VBG','VBP','VBN') and idx not in skip_words:
                    if parse.xpos=='VBG':
                        #Skip descendants if VBG has a non-'VB' and non-'VBP' aux
                        if 'aux' not in dependencies[idx]:
                            pass

                        elif 'VB' not in aux_xpos and 'VBP' not in aux_xpos:
                            skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,skip_conj=True,skip_comp=True)
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
                        comp_flag = check_comp(parse.text,clausal_comp[0].text)

                        if comp_flag:
                            skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies)
                            n += 1
                            continue

                    #Check if task is valid or not
                    valid_task,info,agent_flag = parse_action(parse.text)
                    
                    #Fetch verb object
                    obj = dependencies[int(parse.index)].get('obj',None)
                    task_obj = ''
                    if obj:
                        task_obj = obj[0].text

                    if agent_flag:
                        skip_comp = False
                        if valid_task and parse.xpos != 'VBP' and 'not' not in advmod:
                            tasks[i].append((parse.text,task_obj))
                            verbs[i].append(parse.text)
                            labels[i].append('Task')
                            verb_ids.append(idx)
                            skip_comp = True
                        #Only skip descendant comps if task is valid
                        skip_words = skip_descendants(reverse_graph,idx,skip_words,dependencies,skip_comp=skip_comp)

                n += 1

        if i%500 == 0:
            print(i)
            print(verbs[i])
        
    return tasks,verbs,labels

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

    return predictions

def evaluate_model(y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    precision,recall,f1,_ = precision_recall_fscore_support(y_true,y_pred,average='binary')
    return accuracy,precision,recall,f1

if __name__=='__main__':
    data_path = '/home/parth/IBM Project/'
    #Agent Predicates
    agent_predicates = {'act','adopt','assess','attempt','body_motion','calculate','cause','charge',
                   'contact','cooperate','do','enforce','exert_force','financial_relationship',
                  'help','involve','motion','perform','search','social_interaction','spend','take_in',
                  'transfer','transfer_info','urge','use','utilize','withdraw','work'}

    #Semantics to check before going directly to the clausal complement of a verb
    semantics_for_comps = {'desire', 'attempt', 'begin', 'consider', 'allow', 'necessitate','exist'}

    #Theme Predicates
    theme_predicates = {'act','transfer','motion','contact','consider','utilize','rotational_motion','apply_material',
                    'exert_force','spend','convert','apply_heat','cooperate','enforce','search','help','calculate',
                    'perform','do','charge','transfer_info','use','withdraw','adjust','adopt','created_image',
                    'support','attempt','assess','involve','discover','indicate','social_interaction','take_in',
                    'confront','destroyed','mingled','rush'}

    
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    v3 = nltk.corpus.util.LazyCorpusLoader('verbnet3.3', nltk.corpus.reader.verbnet.VerbnetCorpusReader,r'(?!\.).*\.xml')
    #dataset = sys.argv[1]
    data1 = pd.read_excel(data_path+'All_emails1.xlsx')
    data2 = pd.read_excel(data_path+'All_emails2.xlsx')
    data3 = pd.read_excel(data_path+'All_emails3.xlsx')
    print(data1.shape,data2.shape,data3.shape)
    data = pd.concat([data1,data2,data3])
    data.index = list(range(data.shape[0]))
    print(data.shape)
    del data1
    del data2
    del data3

    # data = data.iloc[:200000]
    data = preprocess(data,data_path)
    print('data shape: ',data.shape)
    '''
    with open(data_path+dataset+'_all_VP.pickle','rb') as f:
        VP_data = pickle.load(f)
    
    with open(data_path+dataset+'_all_VP_df.pickle','rb') as f:
        VP_df = pickle.load(f)
        
    with open(data_path+dataset+'_all_tasks.pickle','rb') as f:
        tasks = pickle.load(f)
        
    with open(data_path+dataset+'_all_context.pickle','rb') as f:
        context = pickle.load(f)
    '''
    
    nlp = stanfordnlp.Pipeline(use_gpu=False)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        predicted_tasks,predicted_verbs,labels = extract_tasks(data)

    with open(data_path+'Predicted_verbs.pickle','wb') as f:
        pickle.dump(predicted_verbs,f)

    data['Verbs'] = [str(x) for x in list(predicted_verbs.values())]
    data['Task/Goal'] = [str(x) for x in list(labels.values())]
    data.to_excel(data_path+'Labeled_data.xlsx',index=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        VP_data = extract_VP(data)

    tasks,new_VP,context = get_tasks_for_VP(VP_data,data)
    print(tasks[:10],new_VP[:10])
    with open(data_path+'All_tasks.pickle','wb') as f:
        pickle.dump(tasks,f)
        
    with open(data_path+'All_VP.pickle','wb') as f:
        pickle.dump(new_VP,f)

    with open(data_path+'All_VP_df.pickle','wb') as f:
        pickle.dump(VP_data,f)
        
    with open(data_path+'All_context.pickle','wb') as f:
        pickle.dump(context,f)

    '''
    predictions = get_predictions(VP_df,data,predicted_verbs)
    accuracy,precision,recall,f1 = evaluate_model(tasks,predictions)
    
    print('Lin F1 score: {}'.format(f1))
    print('Lin Precision: {}'.format(precision))
    print('Lin Recall: {}'.format(recall))
    print('Lin Accuracy: {}'.format(accuracy))
    
    with open(data_path+dataset+'_lin_predictions.pickle','wb') as f:
        pickle.dump(predictions,f)
    '''