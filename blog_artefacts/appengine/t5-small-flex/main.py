# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_app]

import logging

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

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

app.config['SECRET_KEY'] = 'c2af6efe3d00cfb311ed3e3bf814c3c2e8face7845c58fbeb4684e7d94b924a2'


@app.route('/run-zerohot-t5', methods = ['POST'])
def zeroshot():    # Get the data from the POST request.
    data = request.get_json(force=True)
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
        print(response)
    except Exception as e:
        response = 'Something went wrong'
        print(e)
    return response

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python38_app]
