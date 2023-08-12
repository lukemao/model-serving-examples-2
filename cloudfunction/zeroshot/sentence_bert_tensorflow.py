from transformers import AutoTokenizer, AutoModel
from transformers import BartForSequenceClassification, BartTokenizer
from torch.nn import functional as F

tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')

sentence = '''

Hi, 

I had very good experience in using the service.
I have recently made use of Virtual Shopping to help me to rearrang home delivery for my order.
Because I want to change the delivery date, can you share the instructions.

Kindly regards,
Lu
'''


labels = ['unhappy', 'happy', 
           'positive', 'negative', 'neutral']


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

# now find the labels with the highest cosine similarities to
# the sentence
print('Sentiment Analysis')
similarities = F.cosine_similarity(sentence_emb, label_emb)
closest = similarities.argsort(descending=True)
for ind in closest:
    print(f'label: {labels[ind]}    \t similarity: {round(similarities[ind].item(), 2)}')
    
'''
:: Output: 
    
:: Sentiment Analysis
:: label: happy             similarity: 0.13
:: label: positive          similarity: 0.04
:: label: unhappy           similarity: -0.06
:: label: neutral           similarity: -0.12
:: label: negative          similarity: -0.17
'''