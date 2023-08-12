import os
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BartForSequenceClassification, BartTokenizer
from torch.nn import functional as F


class EmailClassifier(object):
    def __init__(self, model):
        self._model = model
        self.class_lables = ['unhappy', 'happy', 'positive', 'negative', 'neutral']

    @classmethod
    def from_path(cls, model_dir):
        model_file = os.path.join(model_dir, 'model.pt')
        model = torch.load(model_file)
        return cls(model)

    def predict(self, sentence, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
        inputs = tokenizer.batch_encode_plus([sentence] + self.class_lables,
                                             return_tensors='pt',
                                             pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = self._model(input_ids, attention_mask=attention_mask)[0]
        sentence_emb = output[:1].mean(dim=1)
        label_emb = output[1:].mean(dim=1)

        similarities = F.cosine_similarity(sentence_emb, label_emb)

        result = dict(zip(self.class_lables, similarities.tolist()))
        #predicted = sorted(result.items(), key=operator.itemgetter(1),
        #                  reverse=True)

        return result
