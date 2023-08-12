from transformers import AutoTokenizer, TFAutoModel
from transformers import BartForSequenceClassification, BartTokenizer
import tensorflow as tf


class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""

    def __init__(self, model):

        self._model = model
        self.class_lables = ['unhappy', 'happy', 'positive', 'negative',
                             'neutral']

    def predict(self, instances, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
        inputs = tokenizer.batch_encode_plus([sentence] + self.class_lables,
                                             return_tensors='pt',
                                             pad_to_max_length=True)
        tf_outputs = model(inputs)

        label_emb = tf.reduce_mean(tf_outputs[0][1:], 1)
        sentence_emb = tf.reduce_mean(tf_outputs[0][:1], 1)

        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
                                                       reduction=tf.keras.losses.Reduction.NONE)

        similarities = cosine_loss(sentence_emb, label_emb).numpy()

        result = dict(zip(labels, similarities))
        #predicted = sorted(result.items(), key=operator.itemgetter(1))

        return result

    @classmethod
    def from_path(cls, model_dir):
        model = TFAutoModel.from_pretrained(model_dir)
        return cls(model)
