import argparse
import pickle

import torch

from Code.Baselines.bert.BertTrain import BertTrainer
from Code.Baselines.FastText.FastTextTrain import FasttextTrainer
from Code.Baselines.svm.SVMTrain import SVMTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train', description='Train model')
    parser.add_argument('--model', required=True,
                        action='store', help='Choose the model')

    args = parser.parse_args()
    if args.model == 'bert':
        trainer = BertTrainer()
        model, accuracies, f1_scores, precisions, recalls = trainer()
        torch.save(model.state_dict(), 'Data/TrainedModels/bert_model.pt')

    elif args.model == 'fasttext':
        trainer = FasttextTrainer()
        model, accuracies, f1_scores, precisions, recalls = trainer()
        model.save_model("Data/TrainedModels/fasttext_model.bin")

    elif args.model == 'svm':
        trainer = SVMTrainer()
        model, accuracies, f1_scores, precisions, recalls = trainer()
        with open('Data/TrainedModels/svm_model.sav', 'wb') as f:
            pickle.dump(model, f)

    else:
        print("Unrecognized model")
        print("Please enter bert/fasttext/svm")
