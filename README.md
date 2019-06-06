# Disaster_Response_Classification

## Overview

This project contains an ETL and Machine Learning Pipeline for classifying Disaster Response messages. We have been provided with labeled data, in which the input space is messages received during a disaster situation, and the lables are the different categories each corresponding to a different division/department of an aid organization. Our goal in this project is to train a model that takes messages as text input (in natural language) and outputs likely labels for them thus helping us to route requests efficiently and accurately on a large scale.

The project contains 3 main pieces:
1) ETL pipeline: which takes in the raw data and applies techniques like tokenization/lemmatization to make it ready for consumption by a ML pipeline.
2) Machine Learning pipeline: which consumes the cleaned data, and trains a AdaBoost classifier using Bag-of-Words and TF-IDF on the input text as features.
3) A Flask Web Aplication: which showcases the data and the trained model.

## Contents
```
├── README.md                         # this file
├── app
│   ├── run.py                        # script for running web app
│   └── templates
│       ├── go.html                   # classification result page of web app
│       └── master.html               # main page of web app
├── data
│   ├── DisasterResponse.db           # database containing cleaned data
│   ├── disaster_categories.csv       # raw input file contaning labels
│   ├── disaster_messages.csv         # raw input file containing input messages
│   └── process_data.py               # script for cleaning and storing data
└── models
    ├── train_classifier.py           # script for training and storing classifier
    └── model.pickle                  # python pickle file containing trained classifier
```

## Instructions

- To run ETL pipeline that cleans data and stores in database

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    
- To train, test and save the ML model

    `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`
    
- To run the web application, and go to http://0.0.0.0:3001/

    `python app/run.py`
