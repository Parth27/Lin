import datetime
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from transformers import BertConfig, BertModel, BertTokenizer

from Code.Baselines.bert.Bert import BertBaseline
from Code.Baselines.bert.Bert_Preprocessing import prepare_data
from Code.DataProcessing import processData,create_split


class BertTrainer:
    def __init__(self,dataset='email'):
        self.dataset = dataset
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def __call__(self, num_epochs=20, batch_size=64, lr=0.00001, cross_val=True):
        self.loadDataset()
        VP_data, tasks, context = processData(self.data,self.dataset)
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
            X_train_ids, X_train_masks, X_train_segments, X_test_ids, X_test_masks, X_test_segments = prepare_data(
                train_VP, test_VP, max_seq_length=50)

            train_sent_ids, train_sent_masks, train_sent_segments, test_sent_ids, test_sent_masks, test_sent_segments = prepare_data(
                train_context, test_context, max_seq_length=150)
            class_weights = torch.tensor(np.array([1, len([x for x in train_tasks if x == 0])/len(
                [x for x in train_tasks if x == 1])])).float().to(self.device)

            X_train_ids = torch.tensor(X_train_ids)
            X_train_masks = torch.tensor(X_train_masks)
            y_train = torch.tensor(np.array(train_tasks))
            train_sent_ids = torch.tensor(train_sent_ids)
            train_sent_masks = torch.tensor(train_sent_masks)

            X_test_ids = torch.tensor(X_test_ids)
            X_test_masks = torch.tensor(X_test_masks)
            y_test = torch.tensor(np.array(test_tasks))
            test_sent_ids = torch.tensor(test_sent_ids)
            test_sent_masks = torch.tensor(test_sent_masks)

            self.model = BertBaseline()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)

            loss_values = []
            for epoch_i in range(num_epochs):
                print("")
                print(
                    '======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
                print('Training...')
                # Measure how long the training epoch takes.
                t0 = time.time()
                # Reset the total loss for this epoch.
                total_loss = 0
                self.model.train(True)
                # For each batch of training data...
                for step in range(math.ceil(X_train_ids.shape[0]/batch_size)):
                    # Progress update every 40 batches.
                    if step % 20 == 0 and not step == 0:
                        # Calculate elapsed time in minutes.
                        elapsed = format_time(time.time() - t0)
                        print('  Batch {:>5,}   Elapsed: {:}.'.format(
                            step, elapsed))

                    b_input_ids = X_train_ids[step *
                                              batch_size:(step*batch_size)+batch_size].to(device)
                    b_input_mask = X_train_masks[step *
                                                 batch_size:(step*batch_size)+batch_size].to(device)
                    b_sent_ids = train_sent_ids[step *
                                                batch_size:(step*batch_size)+batch_size].to(device)
                    b_sent_masks = train_sent_masks[step *
                                                    batch_size:(step*batch_size)+batch_size].to(device)
                    b_labels = y_train[step *
                                       batch_size:(step*batch_size)+batch_size].long().to(device)

                    self.model.zero_grad()
                    outputs = self.model(input_sentence=b_sent_ids, sentence_mask=b_sent_masks,
                                         current_VP=b_input_ids, VP_mask=b_input_mask)

                    loss = loss_func(outputs, b_labels)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                avg_train_loss = total_loss / \
                    math.ceil(X_train_ids.shape[0]/batch_size)

                loss_values.append(avg_train_loss)
                print("")
                print("  Average training loss: {0:.2f}".format(
                    avg_train_loss))
                print("  Training epoch took: {:}".format(
                    self.format_time(time.time() - t0)))

            accuracy, f1, prec, rec = self.evaluate(
                X_test_ids, X_test_masks, test_sent_ids, test_sent_masks, test_tasks, y_test, batch_size)
            #Save best model
            if f1>best_f1:
                best_f1 = f1
                best_model = self.model

            f1_scores.append(f1)
            accuracies.append(accuracy)
            precisions.append(prec)
            recalls.append(rec)
            fold_num += 1
        return best_model, f1_scores, accuracies, precisions, recalls

    def loadDataset(self):
        self.data = pd.read_excel('Data/Preprocessed_Dataset_'+self.dataset+'.xlsx')
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

    def evaluate(self, X_test_ids, X_test_masks, test_sent_ids, test_sent_masks, test_tasks, y_test, batch_size=64):
        print("Running Evaluation...")
        t0 = time.time()
        self.model.eval()
        all_predictions = []

        for evalStep in range(math.ceil(X_test_ids.shape[0]/batch_size)):
            # Add batch to GPU
            b_input_ids = X_test_ids[evalStep *
                                     batch_size:(evalStep*batch_size)+batch_size].to(device)
            b_input_mask = X_test_masks[evalStep*batch_size:(
                evalStep*batch_size)+batch_size].to(device)
            b_labels = y_test[evalStep *
                              batch_size:(evalStep*batch_size)+batch_size].long().to(device)
            b_sent_ids = test_sent_ids[evalStep*batch_size:(
                evalStep*batch_size)+batch_size].to(device)
            b_sent_masks = test_sent_masks[evalStep*batch_size:(
                evalStep*batch_size)+batch_size].to(device)

            with torch.no_grad():
                outputs = self.model(input_sentence=b_sent_ids, sentence_mask=b_sent_masks,
                                     current_VP=b_input_ids, VP_mask=b_input_mask)

            logits = outputs.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            all_predictions.extend([np.argmax(x) for x in logits])

        prec, rec, f1, _ = precision_recall_fscore_support(
            test_tasks, all_predictions, average='binary')
        accuracy = accuracy_score(test_tasks, all_predictions)
        # Report the final results
        print("  Accuracy: {0:.2f}".format(accuracy))
        print("  Precision: {0:.2f}".format(prec))
        print("  Recall: {0:.2f}".format(rec))
        print("  F1 score: {0:.2f}".format(f1))
        print("  Evaluation took: {:}".format(format_time(time.time() - t0)))
        return accuracy, f1, prec, rec
