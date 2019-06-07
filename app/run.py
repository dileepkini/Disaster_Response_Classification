import json
import plotly
import pandas as pd

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

app = Flask(__name__)

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """
    A function that tokenizes raw text into individual tokens by
    removing urls and non-alphabetic characters, splitting by whitespace and converting to lower case.
    The function also removes stop words and lemmatizes the words.
    
    Args:
        text (str): raw text string which is to be tokenized
    
    Returns:
        list: A list of strings, each string representing a token
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, " ")
        
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens if w not in stops]
    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/model.pickle")

# test data
X = df['message'].values
Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
Y_pred = model.predict(X_test)
cols = df.columns[4:]
score_dict={}
for i, col in enumerate(cols):
    y_true = Y_test[:,i]
    y_pred = Y_pred[:,i]
    score_dict[col] = f1_score(y_true, y_pred)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()
    category_pct = (category_counts * 100 / df.shape[0]).sort_values(ascending=False)
    
    category_scores = pd.Series(score_dict).sort_values(ascending=False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_pct.index,
                    y=category_pct.values
                )
            ],
            
            'layout': {
                'title':'Percentage of data points representing each category',
                'yaxis': {
                    'title': 'Percentage of Data points',
                },
                'xaxis':{
                    'title': 'Categories',
                    'tickangle': -45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_scores.index,
                    y=category_scores.values,
                )
            ],
            'layout': { 
                'title':'Scores on sampled data',
                'xaxis': {
                    'title':'Categories',
                    'tickangle':-45
                },
                'yaxis': {
                    'title':'F1-Score'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
