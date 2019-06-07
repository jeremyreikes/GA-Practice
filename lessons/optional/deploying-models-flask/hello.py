from flask import Flask, render_template
app = Flask(__name__)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/SMSSpamCollection.txt', sep='\t', header=None)
df.columns = ['target', 'msg']
y = df['target']
X = df['msg']

df_html = df.to_html()

cvec = TfidfVectorizer(stop_words='english', max_features = 300)
X = cvec.fit_transform(X)
clf = MultinomialNB()
clf.fit(X, y)

import flask
@app.route('/table')
def table():
    return render_template('table.html', table = df_html)
@app.route('/is_spam', methods=["GET"])
def is_spam():
    msg = pd.Series(flask.request.args['msg'])
    X_new = cvec.transform(msg)
    score = clf.predict(X_new)
    results = {'prediction': score[0]}
    return flask.jsonify(results)



@app.route('/')
def hello():
    return '''
    <body>
    <h2> Hello World! </h2>
    </body>
    '''
# name = 'Jacob'
@app.route('/greet/<name>')
def greet(name):
    return render_template('index.html', name = name)

if __name__ == '__main__':
    app.run(debug = True)