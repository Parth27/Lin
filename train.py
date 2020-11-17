import argparse
import pickle

import torch

from code.Baselines.bert.BertTrain import BertTrainer
from code.Baselines.FastText.FastTextTrain import FasttextTrainer
from code.Baselines.svm.SVMTrain import SVMTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train', description='Train model')
    parser.add_argument('--model', required=True,
                        action='store', help='Choose the model')

    args = parser.parse_args()
    if args.model == 'bert':
        trainer = BertTrainer()
        model, accuracies, f1_scores, precisions, recalls = trainer()
        torch.save(model.state_dict(), 'data/TrainedModels/bert_model.pt')

    elif args.model == 'fasttext':
        trainer = FasttextTrainer()
        model, accuracies, f1_scores, precisions, recalls = trainer()
        model.save_model("data/TrainedModels/fasttext_model.bin")

    elif args.model == 'svm':
        trainer = SVMTrainer()
        model, accuracies, f1_scores, precisions, recalls = trainer()
        with open('data/TrainedModels/svm_model.sav', 'wb') as f:
            pickle.dump(model, f)

    else:
        print("Unrecognized model")
        print("Please enter bert/fasttext/svm")
