from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

import logging

def text_labelling_sb(model, text, labels):
    
    response = ''
    try:
        logging.info('Loading Models: '+ model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
    
        logging.info('Tokenizing')
        # Encode and tokenize the sentence and labels
        inputs = tokenizer.batch_encode_plus([text] + labels,
                                             return_tensors='pt',
                                             pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        logging.info('Generating Embeddings')
        # Inference the embedding using the pre-trained model
        output = model(input_ids, attention_mask=attention_mask)[0]
        sentence_emb = output[:1].mean(dim=1)
        label_emb = output[1:].mean(dim=1)
        
        
        logging.info('Similarity Comparison')
        similarities = F.cosine_similarity(sentence_emb, label_emb)
        closest = similarities.argsort(descending=True)
        logging.info('Assigned Labels: ')
        for ind in closest:
            response += f'{labels[ind]} \t similarity: {round(similarities[ind].item(), 2)}\n'
        del output,model,sentence_emb,label_emb,similarities,closest,inputs,input_ids,attention_mask
        torch.cuda.empty_cache()
    except Exception as e:
        response = 'Something went wrong'
        logging.excepts(str(e))
    return response 

def text_labelling_st(model, text, labels):
    
    response = ''
    try:
        logging.info('Loading Models: '+ model)
        model = SentenceTransformer(model)
        
        logging.info('Generating Embeddings')
        sentence_embeddings     = model.encode([text])
        labels_embeddings       = model.encode(labels)
        
        logging.info('Similarity Comparison')
        cosine_sim = cosine_similarity(sentence_embeddings, labels_embeddings).flatten()
        related_doc_indices = cosine_sim.argsort()[:-len(labels)-1:-1]
        for ind in related_doc_indices:
            logging.info(labels[ind]+' \t similarity: '+str(round(cosine_sim[ind], 2))+'\n')
            response += labels[ind]+' \t similarity: '+str(round(cosine_sim[ind], 2))+'\n'
        del model, sentence_embeddings, labels_embeddings, cosine_sim, related_doc_indices 
        torch.cuda.empty_cache()
    except Exception as e:
        response = 'Something went wrong'
        logging.excepts(str(e))
    return response 
        