import torch
from transformers import BertConfig, BertModel


class BertBaseline(torch.nn.Module):
  def __init__(self, num_classes = 2):
    super(BertBaseline,self).__init__()
    self.LinearLayer1 = torch.nn.Linear(768*2,256)
    self.LinearLayer2 = torch.nn.Linear(256,1)
    self.ReLULayer = torch.nn.ReLU()
    self.bertBase = BertModel.from_pretrained('bert-base-uncased')

  def forward(self, input_sentence, sentence_mask, current_VP, VP_mask):
    BertOutput1 = self.bertBase(input_sentence,attention_mask=sentence_mask)[1]
    BertOutput2 = self.bertBase(current_VP,attention_mask=VP_mask)[1]

    # Concat BERT outputs for sentence and VP
    concat = torch.cat((BertOutput1,BertOutput2),dim=1)

    y = self.LinearLayer1(concat)
    y = self.ReLULayer(y)
    y = self.LinearLayer2(y)
    return y