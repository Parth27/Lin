import argparse

from Code.Baselines.bert.BertTrain import BertTrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train', description='Train model')
    parser.add_argument('--model', required=False,
                        action='store', help='Choose the model')

    args = parser.parse_args()
    if args.model == 'bert':
        trainer = BertTrain()
        model, accuracies, f1_scores, precisions, recalls = trainer()
