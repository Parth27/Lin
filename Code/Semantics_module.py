import numpy as np
import pandas as pd
import nltk
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


def bfs(graph,rootNode):    
    ans = []
    queue = [rootNode]
    while(queue):
        node = queue.pop(0)
        ans.append(node)
        if node in graph:
            queue.extend(graph[node])
        
    return ans

#Function to parse each frame of class
def parse_frames(classid,found):
    valid_task = False
    agent_flag = False
    try:
        for frame in v3.frames(classid):
            found = True
            #Check for Agent predicates
            for s in frame['semantics']:
                for arg in s['arguments']:
                    if arg['type'] == 'ThemRole' and arg['value'] == 'Agent':
                        agent_flag = True
                        if s['predicate_value'] in agent_predicates:
                            valid_task = True
                            return valid_task,found,agent_flag
                        
            #Check for Theme predicates
            for s in frame['semantics']:
                for arg in s['arguments']:
                    if arg['type'] == 'ThemRole' and arg['value'] == 'Theme':
                        if s['predicate_value'] in theme_predicates:
                            valid_task = True
                            return valid_task,found,True
            
    except:
        pass
    return valid_task,found,agent_flag

#Semantic Parsing of Verb in VerbNet   
def parse_action(lemma):
    found_frames = False
    valid_task = False
    agent_flag = False
    
    if not v3.classids(lemma = lemma):
        lemma = stemmer2.stem(lemma)
        if not v3.classids(lemma = lemma):
            return True,True
    
    for vn_class in v3.classids(lemma):
        valid_task,found_frames,agent_flag = parse_frames(vn_class,found_frames)
        if valid_task and found_frames:
            return valid_task,agent_flag

    return valid_task,agent_flag

def extract_tasks(data):
    tasks = {}
    verbs = {}
    question_words = ['why','how','what','which','whose','who','whom','where','whether']
    for i in range(data.shape[0]):
        sent = str(data.iloc[i]['Sentence'])
        tasks[i] = []
        verbs[i] = []
        if sent == '':
            continue
        result = nlp(sent)
        j = 0
        
        for sentence in result.sentences:
            reverse_graph = {}
            dependencies = {}
            skip_words = []
            
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
            
            n = 0
            while(n<len(parses)):
                parse = parses[n]
                idx = int(parse.index)
                if idx in skip_words:
                    n += 1
                    continue

                if parse.xpos in ('VB','VBG','VBP') and idx not in skip_words:
                    #Check if task is valid or not
                    valid_task,agent_flag = parse_action(parse.text)
                    
                    #Fetch verb object
                    obj = dependencies[int(parse.index)].get('obj',None)
                    task_obj = ''
                    if obj:
                        task_obj = obj[0].text

                    if agent_flag and valid_task and parse.xpos != 'VBP':
                        tasks[i].append((parse.text,task_obj))
                        verbs[i].append(parse.text)
                n += 1
        
    return tasks,verbs

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
    data_path = '../Datasets/'
    #Agent predicates
    agent_predicates = {'act','adopt','assess','attempt','body_motion','calculate','cause','charge',
                   'contact','cooperate','do','enforce','exert_force','financial_relationship',
                  'help','involve','motion','perform','search','social_interaction','spend','take_in',
                  'transfer','transfer_info','urge','use','utilize','withdraw','work'}

    #Theme predicates
    theme_predicates = {'act','transfer','motion','contact','consider','utilize','rotational_motion','apply_material',
                    'exert_force','spend','convert','apply_heat','cooperate','enforce','search','help','calculate',
                    'perform','do','charge','transfer_info','use','withdraw','adjust','adopt','created_image',
                    'support','attempt','assess','involve','discover','indicate','social_interaction','take_in',
                    'confront','destroyed','mingled','rush'}


    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    v3 = nltk.corpus.util.LazyCorpusLoader('verbnet3.3', nltk.corpus.reader.verbnet.VerbnetCorpusReader,r'(?!\.).*\.xml')
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

    nlp = stanfordnlp.Pipeline(use_gpu=False)
    
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
    with open(data_path+dataset+'_Semantic_predictions.pickle','wb') as f:
        pickle.dump(predictions,f)