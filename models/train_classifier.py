import re
import click
import pickle
import pandas as pd
from sqlalchemy import create_engine


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()
    df = pd.read_sql('DisasterResponse', conn)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    categories = df.columns[4:]
    return X, Y, categories


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, " ")
        
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens if w not in stops]
    return tokens


def build_model(X_train, y_train):
    pipeline = Pipeline([('count', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier())))])
    
    parameters = {
        'clf__estimator__base_estimator__max_depth' : [None, 2],
        'clf__estimator__base_estimator__max_features' : [None, 'auto']
    }

    cv = GridSearchCV(pipeline, parameters, scoring='f1_weighted')
    cv.fit(X_train, y_train)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        test_col = y_test[:,i]
        pred_col = y_pred[:,i]
        print(col.upper())
        print(classification_report(test_col, pred_col))


@click.command()
@click.argument('database_filename', type=click.Path(exists=True))
@click.argument('output_filename', type=click.Path())
def main(database_filename, output_filename):
    X, Y, category_names = load_data(database_filename)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    
    model = build_model(X_train, Y_train)

    pickle.dump(model, open(output_filename, 'wb'))
    
    evaluate_model(model, X_test, Y_test, category_names)

if __name__ == '__main__':
    main()
