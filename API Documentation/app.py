from flask import Flask, jsonify, request
import os
from os.path import join
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
from text_cleaner_module import cleansing_text
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__)) 
UPLOAD_FOLDER = join(basedir, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "API Documentation for Data Processing and Modelling using NN & LSTM ☕️",
        "description": "***",
        "version": "1.0.0"
    }
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)


# Load total_data
with open('lstm/resources/total_data.pickle', 'rb') as fp:
    total_data = pickle.load(fp)

max_features = 100000
sentiment = ['negative', 'neutral', 'positive']
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(total_data)

#LSTM
file = open("lstm/resources/x_pad_sequences.pickle", "rb")
feature_file_from_lstm = pickle.load(file)
file.close()
model_file_from_lstm = load_model("lstm/model/model.h5")

#NN
count_vect = pickle.load(open("nn/resources/feature.p","rb"))
model_NN = pickle.load(open("nn/model/model.p","rb"))

@app.route('/', methods=['GET'])
def root():
    return "API untuk Data Processing & Modelling menggunakan NN & LSTM ☕"

@swag_from("docs/nn.yml", methods=['POST'])
@app.route('/nn', methods=['POST'])
def text_processing():
    original_text = request.form.get('text')
    cleaned_text = cleansing_text(original_text)

    #Vectorizing 
    text = count_vect.transform([cleaned_text])
    #Predict sentiment
    get_sentiment = model_NN.predict(text)[0]
    
    json_response = {
        'status_code': 200,
        'description': 'Results of NN model',
        'data': {
            'original text':original_text,
            'cleaned text':cleaned_text,
            'sentiment': get_sentiment.tolist()  # Convert to list
        }
    }
    return jsonify(json_response)

@swag_from("docs/nnCSV.yml", methods=['POST'])
@app.route("/nnCSV", methods=['POST'])
def nnCSV():
    original_file = request.files['file']
    filename = secure_filename(original_file.filename)
    filepath = 'static/' + filename
    original_file.save(filepath)
    
    df = pd.read_csv(filepath, header=0)
    
    sentiment_results = []

    for text in df.iloc[:, 0]:
        original_text = text
        cleaned_text = cleansing_text(original_text)

        text = count_vect.transform([cleaned_text])  #Vectorizing 
        get_sentiment = model_NN.predict(text)[0]   #Predict sentiment

        sentiment_results.append({
            'text': original_text,
            'cleaned text':cleaned_text,
            'sentiment': get_sentiment.tolist()  # Convert to list
            })

    json_response = {
        'status_code': 200,
        'description': 'Results of NN model',
        'data': sentiment_results
    }

    return jsonify(json_response)

@swag_from("docs/lstm.yml", methods=['POST'])
@app.route("/lstm", methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing_text(original_text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': 'Results of LSTM model',
        'data': {
            'text':original_text,
            'cleaned text':text[0],
            'sentiment': get_sentiment
        }
    }
    return jsonify(json_response)

@swag_from("docs/lstmCSV.yml", methods=['POST'])
@app.route("/lstmCSV", methods=['POST'])
def lstmCSV():
    original_file = request.files['file']
    filename = secure_filename(original_file.filename)
    filepath = 'static/' + filename
    original_file.save(filepath)
    
    df = pd.read_csv(filepath, header=0)
    
    sentiment_results = []

    for text in df.iloc[:, 0]:
        original_text = text
        cleaned_text = cleansing_text(original_text)
        tokenizer.fit_on_texts(cleaned_text)
        feature = tokenizer.texts_to_sequences([cleaned_text])
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        sentiment_results.append({
            'text': original_text,
            'cleaned text':cleaned_text,
            'sentiment': get_sentiment
            })


    json_response = {
    'status_code': 200,
    'description': 'Results of LSTM model',
    'data': sentiment_results
    }

    return jsonify(json_response)

if __name__ == '__main__':
    app.run(debug=True)
