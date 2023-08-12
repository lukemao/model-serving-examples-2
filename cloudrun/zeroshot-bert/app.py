from flask import Flask, request, jsonify
import pickle
import subprocess
import requests
import os
import datetime as dt
import json

from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F

from sklearn.metrics.pairwise import cosine_similarity
import torch

app = Flask(__name__)


@app.route('/run-zerohot', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    response = run_zeroshot(data)

    return response


def run_zeroshot(data):

    ################# run zeroshot ############

    response = ''

    sentence = data['sentence']
    labels = data['labels'].split(',')
    try:
        try:
            print('retrieve saved model')
            tokenizer = AutoTokenizer.from_pretrained('./sentence-bert_tokenizer')
            model = AutoModel.from_pretrained('./sentence-bert_model')
        except:
            print('loading and save models')
            tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
            model = AutoModel.from_pretrained('deepset/sentence_bert')
            tokenizer.save_pretrained('./sentence-bert_tokenizer')
            model.save_pretrained('./sentence-bert_model')

        print('Tokenizing')
        # Encode and tokenize the sentence and labels
        inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                             return_tensors='pt',
                                             pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        print('Generating Embeddings')
        # Inference the embedding using the pre-trained model
        output = model(input_ids, attention_mask=attention_mask)[0]
        sentence_emb = output[:1].mean(dim=1)
        label_emb = output[1:].mean(dim=1)


        print('Similarity Comparison')
        similarities = F.cosine_similarity(sentence_emb, label_emb)
        closest = similarities.argsort(descending=True)
        print('Assigned Labels: ')
        for ind in closest:
            response += f'{labels[ind]} \t similarity: {round(similarities[ind].item(), 2)}\n'
        del output,model,sentence_emb,label_emb,similarities,closest,inputs,input_ids,attention_mask
        torch.cuda.empty_cache()
    except Exception as e:
        response = 'Something went wrong'
        print(e)

    return response


if __name__ == '__main__':
    # Model is loaded when the API is launched
    app.run(debug=True, host='0.0.0.0')