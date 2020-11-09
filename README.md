# Lin: Unsupervised task extraction from textual communication
Unsupervised task extraction from textual communication, presented at COLING 2020

## Setup Instructions
* Download and extract this repo
* Install dependencies
```bash
pip install -r requirements.txt
```
* Download stanfordnlp english library
```python
import stanfordnlp
stanfordnlp.download('en')
```
* Download verbnet 3 and stopwords in nltk
```python
import nltk
nltk.download('verbnet3')
nltk.download('stopwords')
```

## How to run Lin

### Running Lin on given datasets

### Running Lin on your dataset

## Evaluating Lin

## Training baseline models
```bash
python train.py --model=<MODEL>
```
`MODEL` can be one of bert/fasttext/svm. The selected model is then trained on email dataset.
Training is done using 5-fold cross validation and the model with best f1 score is stored in `Data/TrainedModels`.
The average accuracy, precision, recall and f1 score are reported.

## Evaluating trained models
