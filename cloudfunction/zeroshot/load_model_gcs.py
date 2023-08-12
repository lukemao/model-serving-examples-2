from transformers import AutoTokenizer, AutoModel
from transformers import BartForSequenceClassification, BartTokenizer
from torch.nn import functional as F

tokenizer = AutoTokenizer.from_pretrained('https://console.cloud.google.com/storage/browser/landing-data-models/transformers/')
tokenizer = AutoTokenizer.from_pretrained('https://console.cloud.google.com/storage/browser/landing-data-models/transformers/')
#model = AutoModel.from_pretrained('deepset/sentence_bert')