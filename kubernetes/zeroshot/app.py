import os

import logging
import sys

from flask import Flask, render_template, flash
from labellingForm import LabellingForm
from zeroShot import text_labelling_sb, text_labelling_st

app = Flask(__name__)

app.config['SECRET_KEY'] = 'c2af6efe3d00cfb311ed3e3bf814c3c2e8face7845c58fbeb4684e7d94b924a2'

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


import google.cloud.logging
logging_client = google.cloud.logging.Client()
cloud_log = logging_client.logger('zeroshot')
logging_client.get_default_handler()
logging_client.setup_logging()
   

@app.route('/')
def home():
    target = os.environ.get('TARGET', 'World')
    return render_template('home.html', title='GFT ML')



@app.route('/nlp', methods = ['GET', 'POST'])
def zeroshot():
    form = LabellingForm()
    if form.validate_on_submit():
        labels = form.inputLabels.data.split(',')
        logging.info('Labels: '+ str(labels))
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
    logging.excepts('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))