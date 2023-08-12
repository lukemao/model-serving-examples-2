from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import torch
import os
model_dir = '.'


def predict(text, labels):
    model_file = os.path.join(model_dir, 'model.pt')
    model = torch.load(model_file)
    response = ''
    try:
        # Tokenizing
        # Encode and tokenize the sentence and labels
        tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
        # model = AutoModel.from_pretrained('deepset/sentence_bert')
        inputs = tokenizer.batch_encode_plus([text] + labels,
                                             return_tensors='pt',
                                             pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        print('Generating Embeddings')
        # Generating Embeddings
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
    except:
        response = 'Something went wrong'
    return response