# Lin: Unsupervised task extraction from textual communication
Unsupervised task extraction from textual communication, presented at COLING 2020

## Setup Instructions
* Download and extract this repo
* Run "pip install -r requirements.txt" for installing all dependencies

## Training baseline models
For training baseline models use command  "python train.py --model=MODEL".
MODEL can be one of bert/fasttext/svm. The selected model is then trained on email dataset.
Training is done using 5-fold cross validation and the model with best f1 score is stored in Data/TrainedModels folder.
The average accuracy, precision, recall and f1 score are reported.

## Evaluating trained models
