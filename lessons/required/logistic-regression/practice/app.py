from flask import Flask, render_template
app = Flask(__name__)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.metrics import accuracy_score


import pandas as pd
bank = pd.read_csv('../data/bank.csv')
jobs = bank.job.value_counts()
X_num = bank[['age', 'duration']]
X_dum = pd.get_dummies(bank[['loan', 'marital', 'job']], drop_first=True)
X = pd.concat([X_num, X_dum], axis = 1)
# convert selected features do dummies
y = bank.y
# set the model
clf = LogisticRegression(solver = 'newton-cg')
# set x and y

# train test splot
X_train, X_test, y_train, y_test = train_test_split(X, y)
# fit model
clf.fit(X_train, y_train)
preds = cross_val_predict(clf, X_test, y_test, cv = 5)
score = accuracy_score(y_test, preds)

@app.route("/")
def hello():
    return render_template('report.html', a = score)

if __name__ == '__main__':
    app.run(debug=True)