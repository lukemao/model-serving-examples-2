from flask import Flask, jsonify, make_response, request
from model import predict
app = Flask(__name__)


@app.route('/score', methods=['POST'])
def score():
    labels = request.json['label']
    sentence = request.json['sentence']
    response = predict(sentence, labels)
    return make_response(jsonify({'score': response}))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))