# Lin: Unsupervised task extraction from textual communication
Unsupervised task extraction from textual communication, presented at COLING 2020

## Lin Performance
In our experiments, `Lin` outperforms strong baselines of `BERT`, `FastText` and `Universal Sentence Encoder + SVM` across multiple domains

Results on email dataset
| Model    | Accuracy | Precision | Recall | F1 score |
| ---      | ---      | ---    | ---      | ---      |
| SVM + USE| 89.35    | 54.40   | 82.42 | 65.51 |
| FastText | 69.53    | 69.95   | 68.62 | 69.25 |
| BERT | 89.17    | 74.82   | 82.85 | 78.58 |
| Lin Syntax | 93.34    | 74.48   | 69.80 | 72.06 |
| Lin Semantics | 91.08    | 58.62   | 93.36 | 72.01 |
| **Lin** | **95.36**    | **83.82**   | **77.29** | **80.42** |

Results on chat dataset (baselines trained on email dataset)
| Model    | Accuracy | Precision | Recall | F1 score |
| ---      | ---      | ---    | ---      | ---      |
| SVM + USE| 83.42    | 70.09   | 72.11 | 71.09 |
| FastText | 78.80    | 71.66   | 41.34 | 52.43 |
| BERT | 92.68    | 85.32   | 89.42 | 87.32 |
| Lin Syntax | 92.12    | 87.12   | 84.61 | 85.85 |
| Lin Semantics | 89.40    | 74.07   | 96.15 | 83.68 |
| **Lin** | **94.85**    | **94.73**   | **86.53** | **90.45** |
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
```bash
python main.py --dataset=<DATASET>
```
`DATASET` can be either `chat` or `email`.
On the provided dataset, the code will extract tasks, store them in the same excel file under column `Tasks` and report accuracy, precision, recall and f1 score.

### Running Lin on your dataset
```bash
python main.py --dataset=<path to your dataset>
```
Please note that your dataset must be an **excel file** with a `Sentence` column that has the sentences from which tasks are to be extracted.
Running `Lin` on custom dataset will also extract and store tasks the same way, but will not report accuracy, precision, recall and f1 score.

## Training baseline models
```bash
python train.py --model=<MODEL>
```
`MODEL` can be either `bert`, `fasttext` or `svm`. The selected model is then trained on email dataset.
Training is done using 5-fold cross validation and the model with best f1 score is stored in `data/TrainedModels`.
The average accuracy, precision, recall and f1 score are reported.
Training models multiple times will overwrite previously trained model.

## Evaluating trained models
```bash
python evaluate.py --model=<MODEL> --dataset=<DATASET>
```
`DATASET` can be either `chat` or `email`. Metrics reported are accuracy, precision, recall and f1 score.
