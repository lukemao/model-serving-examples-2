from flask import Flask, request, jsonify
import pickle
import subprocess
import requests
import os
import datetime as dt
import json

import tensorflow as tf

from transformers import (
    TFT5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

app = Flask(__name__)


@app.route('/run-zerohot-t5', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    response = run_zeroshot(data)

    return response


def run_zeroshot(data):

    ################# run zeroshot ############

    response = ''

    sentence = data['sentence']
    model_name = data['model_name']
    try:

        config = T5Config.from_pretrained(model_name)
        try:
            print('retrieve saved model')
            tokenizer = T5Tokenizer.from_pretrained('./'+model_name+'_tokenizer')
            model = TFT5ForConditionalGeneration.from_pretrained('./'+model_name+'_model', config=config)
        except:
            print('loading and save models')
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = TFT5ForConditionalGeneration.from_pretrained(model_name, config=config)
            tokenizer.save_pretrained('./'+model_name+'_tokenizer')
            model.save_pretrained('./'+model_name+'_model')


        task_specific_config = getattr(model.config, "task_specific_params", {})
        translation_config = task_specific_config.get("translation_en_to_de", {})
        model.config.update(translation_config)

        preprocess_text = sentence.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text

        input_ids = tokenizer.encode(t5_prepared_Text, return_tensors="tf")
        outputs = model.generate(input_ids=input_ids,
                                 max_length=50,)

        response += '{ output:'+tokenizer.decode(outputs[0])+'\n}'

    except Exception as e:
        response = 'Something went wrong'
        print(e)

    return response


if __name__ == '__main__':
    # Model is loaded when the API is launched
    app.run(debug=True, host='0.0.0.0')