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

from flask import Flask, render_template, flash
from labellingForm import LabellingForm
from zeroShot import text_labelling_sb, text_labelling_st


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

app.config['SECRET_KEY'] = 'c2af6efe3d00cfb311ed3e3bf814c3c2e8face7845c58fbeb4684e7d94b924a2'


@app.route('/')
def home():
    """Return a friendly HTTP greeting."""
    return render_template('home.html', title='GFT ML')


@app.route('/nlp', methods = ['GET', 'POST'])
def zeroshot():
    form = LabellingForm()
    if form.validate_on_submit():
        labels = form.inputLabels.data.split(',')
        print('Labels: ', labels)
        response = None
        if form.model.data == 'st':
            response = text_labelling_st('bert-base-nli-mean-tokens', form.inputText.data, labels)
        else: 
            response = text_labelling_sb('deepset/sentence_bert', form.inputText.data, labels)
        if response is not None:
            form.results.data = f'{response}'
        
    return render_template('nlp.html', title='Labelling Service', form=form)


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
