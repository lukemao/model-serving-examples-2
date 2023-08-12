def email_labelling(request):
    request_json = request.get_json()
    
    sentence = request_json['email']
    labels = request_json['labels']
    response = 'Sentence submitted for labelling \n' + str(sentence)+'\n'
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
    model = AutoModel.from_pretrained('deepset/sentence_bert')

    # Encode and tokenize the sentence and labels
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # Inference the embedding using the pre-trained model
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_emb = output[:1].mean(dim=1)
    label_emb = output[1:].mean(dim=1)
    
    from torch.nn import functional as F
    similarities = F.cosine_similarity(sentence_emb, label_emb)
    closest = similarities.argsort(descending=True)
    print('Assigned Labels')
    for ind in closest:
        response += f'label: {labels[ind]}    \t similarity: {round(similarities[ind].item(), 2)} \n'
    return response