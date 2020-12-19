import nltk
import numpy as np
import pandas as pd
import stanfordnlp
from nltk.corpus import verbnet
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import KFold

from code.DataProcessing import processData


class lin:
    def __init__(self, themePredicates, agentPredicates, compSemantics):
        self.vn = nltk.corpus.util.LazyCorpusLoader(
            'verbnet3', nltk.corpus.reader.verbnet.VerbnetCorpusReader, r'(?!\.).*\.xml')
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        # Change use_gpu to True if good GPU is available
        self.parser = stanfordnlp.Pipeline(use_gpu=False)
        # Theme Predicates
        self.themePredicates = themePredicates
        # Agent Predicates
        self.agentPredicates = agentPredicates
        # Semantics to check before going directly to the clausal complement of a verb
        self.compSemantics = compSemantics
        self.task_classes = set()
        self.agent_classes = set()
        self.comp_classes = set()
        for classid in self.vn.classids():
            valid_task, found, agent_flag = self.parse_frames(classid)
            if valid_task:
                self.task_classes.add(classid)
            if agent_flag:
                self.agent_classes.add(classid)
            if self.parse_for_comp(classid):
                self.comp_classes.add(classid)

    def dfs(self, graph, node, arr, modify_dependencies=False, nsubj='', dependencies={}):
        if modify_dependencies:
            dep = dependencies[node].get('nsubj', [])
            if not dep:
                dependencies[node]['nsubj'] = [nsubj]
            else:
                nsubj = dep[0]
        if node in graph.keys():
            for n in graph[node]:
                self.dfs(graph, n, arr, modify_dependencies,
                         nsubj, dependencies)
                arr.append(n)
        return arr, dependencies

    def bfs(self, graph, rootNode):
        ans = []
        queue = [rootNode]
        while(queue):
            node = queue.pop(0)
            ans.append(node)
            if node in graph:
                queue.extend(graph[node])

        return ans

    def skip_descendants(self, graph, idx, skip_words, dependencies, skip_conj=False, skip_comp=False):
        arr, _ = self.dfs(graph, idx, [])
        if not skip_conj:
            conj = [int(x.index) for x in dependencies[idx].get('conj', [])]
            arr = [x for x in arr if x not in conj]

        if not skip_comp:
            clausal_comp = [int(x.index) for x in dependencies[idx].get(
                'xcomp', []) if 'VB' in x.xpos]
            clausal_comp += [int(x.index)
                             for x in dependencies[idx].get('ccomp', []) if 'VB' in x.xpos]
            arr = [x for x in arr if x not in clausal_comp]

        skip_words.extend(arr)

        return skip_words

    def check_subject(self, dependencies, idx):
        subj = [x.text.lower() if x !=
                '' else '' for x in dependencies[idx].get('nsubj', [''])]
        valid_subjects = ['you', 'us', 'we', 'i', '']
        subj_check = any(x in subj for x in valid_subjects)
        return subj_check

    def parse_frames(self, classid, found = False):
        valid_task = False
        agent_flag = False
        try:
            for frame in self.vn.frames(classid):
                found = True
                for s in frame['semantics']:
                    for arg in s['arguments']:
                        if arg['type'] == 'ThemRole' and arg['value'] == 'Agent':
                            agent_flag = True
                            if s['predicate_value'] in self.agentPredicates:
                                valid_task = True
                                return valid_task, found, agent_flag

                for s in frame['semantics']:
                    for arg in s['arguments']:
                        if arg['type'] == 'ThemRole' and arg['value'] == 'Theme':
                            if s['predicate_value'] in self.themePredicates:
                                valid_task = True
                                return valid_task, found, True

        except:
            pass
        return valid_task, found, agent_flag

    def parse_action(self, lemma):
        found_frames = False
        valid_task = False
        agent_flag = False

        if not self.vn.classids(lemma=lemma):
            lemma = self.stemmer.stem(lemma)
            if not self.vn.classids(lemma=lemma):
                return True, True

        vn_classes = set(self.vn.classids(lemma))
        valid_task = True if (vn_classes & self.task_classes) else False
        agent_flag = True if (vn_classes & self.agent_classes) else False
        return valid_task, agent_flag

    def parse_for_comp(self, classid, found = False):
        comp_check = False
        try:
            for frame in self.vn.frames(classid):
                for s in frame['semantics']:
                    for arg in s['arguments']:
                        if arg['type'] == 'Event':
                            if s['predicate_value'] in self.compSemantics:
                                comp_check = True
                                return comp_check
        except:
            pass
        return comp_check

    def check_comp(self, parent_lemma, child_lemma):
        # found = False
        comp_check = False
        if not self.vn.classids(lemma=parent_lemma):
            parent_lemma = self.stemmer.stem(parent_lemma)

        vn_class = set(self.vn.classids(parent_lemma))
        comp_check = True if (vn_class & self.comp_classes) else False
        # for classid in self.vn.classids(parent_lemma):
        #     comp_check, found = self.parse_for_comp(classid, found)
        #     if found and comp_check:
        #         break

        if comp_check:
            valid_task, _ = self.parse_action(child_lemma)
            if valid_task:
                return True

        return False

    def get_predictions(self, VP_df, data, predicted_verbs):
        # Function to link the predicted verbs to our Verb Phrases
        predictions = [[0 for x in range(len(VP_df[i]))]
                       for i in range(len(VP_df))]

        for i in range(len(VP_df)):
            if pd.isnull(data.iloc[i]['Verbs']):
                pass
            else:
                label = [x.strip()
                         for x in str(data.iloc[i]['Task/Goal']).split(',')]
                valid_labels = [x for x in range(
                    len(label)) if label[x] == 'Task']
                verbs = data.iloc[i]['Verbs'].replace(
                    '[', '').replace(']', '').replace("'", "").split(',')
                verbs = [verbs[x].strip().lower() for x in valid_labels]
                visited = dict([(x, False) for x in range(len(VP_df[i]))])
                for x in range(len(predicted_verbs[i])):
                    for y in range(len(VP_df[i])):
                        VP = VP_df[i][y]
                        if visited[y]:
                            continue
                        if predicted_verbs[i][x].lower() == VP[1].lower():
                            visited[y] = True
                            predictions[i][y] = 1

        predictions = [item for sublist in predictions for item in sublist]
        return predictions

    def extractTasks(self, data):
        tasks = {}
        verbs = {}
        question_words = ['why', 'how', 'what', 'which',
                          'whose', 'who', 'whom', 'where', 'whether']
        for i in range(data.shape[0]):
            sent = str(data.iloc[i]['Sentence'])
            tasks[i] = []
            verbs[i] = []
            if sent == '':
                continue
            result = self.parser(sent)

            for sentence in result.sentences:
                reverse_graph = {}
                dependencies = {}
                skip_words = []
                verb_ids = []

                for parse in sentence.words:
                    reverse_graph.setdefault(int(parse.governor), [])

                    reverse_graph[int(parse.governor)].append(int(parse.index))

                    if int(parse.governor) == 0:
                        rootNode = int(parse.index)

                    dependencies.setdefault(int(parse.governor), {})
                    dependencies.setdefault(int(parse.index), {})
                    dependencies[int(parse.governor)].setdefault(
                        parse.dependency_relation, [])
                    dependencies[int(parse.governor)
                                 ][parse.dependency_relation].append(parse)

                arr = self.bfs(reverse_graph, rootNode)
                parses = [sentence.words[y-1] for y in arr if y != 0]

                _, dependencies = self.dfs(reverse_graph, int(
                    parses[0].index), [], True, '', dependencies)
                n = 0
                while(n < len(parses)):
                    parse = parses[n]
                    idx = int(parse.index)
                    if idx in skip_words:
                        n += 1
                        continue

                    # Skip questions
                    advmod = [x.text.lower()
                              for x in dependencies[idx].get('advmod', [])]
                    question_check = any(
                        item in advmod for item in question_words)
                    if question_check:
                        skip_words = self.skip_descendants(
                            reverse_graph, idx, skip_words, dependencies, skip_conj=True, skip_comp=True)
                        n += 1
                        continue

                    # Skip descendants if root or current node is adjective or noun or past verbs
                    if parse.xpos in ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBN', 'VBD', 'VBZ'):
                        skip = True
                        if parse.xpos == 'VBN':
                            aux_pass = [
                                x.xpos for x in dependencies[idx].get('aux:pass', [])]
                            if 'VB' in aux_pass:
                                skip = False

                        if parse.xpos == 'JJ':
                            skip_comp = False
                        else:
                            skip_comp = True

                        if skip == True:
                            skip_words = self.skip_descendants(
                                reverse_graph, idx, skip_words, dependencies, skip_comp=skip_comp)
                        n += 1
                        continue

                    aux_xpos = [
                        x.xpos for x in dependencies[idx].get('aux', [])]
                    past_check = any(
                        item in aux_xpos for item in ['VBN', 'VBD'])
                    if past_check:
                        skip_words = self.skip_descendants(
                            reverse_graph, idx, skip_words, dependencies, skip_conj=True, skip_comp=True)
                        n += 1
                        continue

                    if parse.xpos in ('VB', 'VBG', 'VBP', 'VBN') and idx not in skip_words:
                        if parse.xpos == 'VBG':
                            # Skip descendants if VBG has a non-'VB' and non-'VBP' aux
                            if 'aux' not in dependencies[idx]:
                                pass
                            elif 'VB' not in aux_xpos and 'VBP' not in aux_xpos:
                                skip_words = self.skip_descendants(
                                    reverse_graph, idx, skip_words, dependencies, skip_conj=True, skip_comp=True)
                                n += 1
                                continue
                        # Subject check
                        subj_check = self.check_subject(dependencies, idx)
                        if not subj_check:
                            n += 1
                            continue

                        # Jump to the clausal complement of verb
                        clausal_comp = [x for x in dependencies[idx].get(
                            'xcomp', []) if 'VB' in x.xpos]
                        clausal_comp += [x for x in dependencies[idx].get(
                            'ccomp', []) if 'VB' in x.xpos]

                        if clausal_comp:
                            comp_flag = self.check_comp(
                                parse.text, clausal_comp[0].text)

                            if comp_flag:
                                skip_words = self.skip_descendants(
                                    reverse_graph, idx, skip_words, dependencies)
                                n += 1
                                continue

                        # Check if task is valid or not
                        valid_task, agent_flag = self.parse_action(
                            parse.text)

                        # Fetch verb object
                        obj = dependencies[int(parse.index)].get('obj', None)
                        task_obj = ''
                        if obj:
                            task_obj = obj[0].text

                        if agent_flag:
                            skip_comp = False
                            if valid_task and parse.xpos != 'VBP' and 'not' not in advmod:
                                tasks[i].append((parse.text, task_obj))
                                verbs[i].append(parse.text)
                                verb_ids.append(idx)
                                skip_comp = True
                            # Only skip descendant comps if task is valid
                            skip_words = self.skip_descendants(
                                reverse_graph, idx, skip_words, dependencies, skip_comp=skip_comp)

                    n += 1

        return tasks, verbs

    def evaluate(self, predicted_verbs, data, dataset='email'):
        VP_data, VP_df, tasks, _ = processData(data, dataset, shuffle=False)
        predictions = self.get_predictions(VP_df, data, predicted_verbs)
        accuracy = accuracy_score(tasks, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            tasks, predictions, average='binary')
        return accuracy, precision, recall, f1
