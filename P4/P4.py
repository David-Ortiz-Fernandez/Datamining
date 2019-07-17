

from pymongo import MongoClient
from flask import Flask , render_template , request, url_for
from flask import request
from random import randint
import csv
import pprint
import pandas as pd
import numpy as np
from langdetect import detect
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import indicoio
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



indicoio.config.api_key= 'e25d37d8025290017afc431eab2c062b'
client = MongoClient()
app = Flask(__name__)
db = client.words_database

@app.route('/')
def form():
    return render_template("form_train.html")

@app.route('/train/', methods=['POST'])
def train():
    name=request.form['yourname']
    words=request.form['yourwords']
    result = db.words_database.insert_one(
    {"name": name,
     "words": words}
    )
    saveDatabase()
    return render_template('form_action.html', name=name, words=words)

@app.route('/deleteDb/')
def deleteMongoDb():
    client.drop_database("words_database")
    return render_template('delete.html')

@app.route('/login/')
def log():
    return render_template("form_login.html")

@app.route('/validate/',methods=['POST'])
def validate():
    # Preparing the data.
    data_train = pd.read_csv('train.csv')
    X = data_train[['language','totalL','meanLW','sentiment','numwords']]
    Y = data_train['name']

    name=request.form['yourname']
    words=request.form['yourwords']
    row = [{'name': name, 'words':words}]

    df = pd.DataFrame(row)

    output = convertData(df.iloc[0])

    print ( output)
    with open('login.csv','w') as csvfile:
        fieldnames = ['name','language','totalL','meanLW','sentiment','numwords']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()

    f = open('login.csv','a')
    try:
        writer = csv.writer(f)
        writer.writerow(output)
    finally:
        f.close()
    data_test = pd.read_csv('login.csv')
    del data_test['name']

    # The algorithm
    #KNeighborsClassifier(3)
    classifier =RandomForestClassifier(n_estimators=10)
    classifier.fit(X,Y)
    result = classifier.predict(data_test)
    print (result )
    score = cross_val_score(classifier,X,Y,cv=3,n_jobs=1)
    print(score.mean())

    return render_template("result.html",result=result)

def saveDatabase():
    cursor = db.words_database.find({},{'_id':0,'name':1,'words':1})
    with open('some.csv','w') as outfile:
        fields = ['name','words']
        writer = csv.DictWriter(outfile,fieldnames=fields)
        writer.writeheader()
        for x in cursor:
            writer.writerow(x)

def convertData(x):
    data = x
    name = data['name']

    lang = indicoio.language(data['words'])

    if lang['English'] > lang['Spanish']:
        language='english'
    if lang['English'] < lang['Spanish']:
        language='spanish'

    sent = round(indicoio.sentiment(data['words']),2)
    words = data['words'].split()
    numwords = len(words)
    totalL = 0
    for j in words:
        totalL+=len(j)

    meanLW=round((totalL/numwords),2)

    #Encoding Total-lenght
    # Values :
    # 0 : short < 20
    # 1 : medium > 20
    # 2 : long > 40
    if totalL<20:
        totalL=0
    if totalL>40:
        totalL=2
    if totalL>20:
        totalL=1
    #Encoding numwords
    if numwords<5:
        numwords=0
    if numwords>10:
        numwords=2
    if numwords>=5:
        numwords=1

    #Encoding sentiment
    #Values :
    # 1 : Possitive
    # 2 : Negative
    # 0 : Neutral

    if sent>0.6:
        sent=1
    if sent<0.4:
        sent=2
    if sent<1:
        sent=0

    if language == 'english':
        language=1
    if language == 'spanish':
        language=0
        sent=0

    cad = [name,language,totalL,meanLW,sent,numwords]
    return cad

@app.route('/transform/')
def dostuff():
    with open('train.csv','w') as csvfile:
        fieldnames = ['name','language','totalL','meanLW','sentiment','numwords']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()

    data_train = pd.read_csv("some.csv")
    rows= len(data_train.index)
    for i in range(rows):
        data = data_train.iloc[i]
        output = convertData(data)
        f = open('train.csv','a')
        try:
            writer = csv.writer(f)
            writer.writerow(output)
        finally:
            f.close()


    data_train = pd.read_csv('train.csv')

    # Encoding  the data.

    #Encoding names.
    #encoding = LabelEncoder()
    #encoding.fit(data_train['name'].values)
    #names = encoding.transform(data_train['name'].values)
    #data_train['name']=names

    print(data_train)

    HEADER = '''
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta http-equiv="X-UA-Compatible" content="ie=edge">
      <title>Document</title>
    </head>
    <body>
    '''

    FOOTER = '''
        </body>
    </html>
    '''

    with open('templates/transform.html','w') as f:
        f.write(HEADER)
        f.write(data_train.to_html(classes='data_train'))
        f.write(FOOTER)

    return render_template("transform.html")


app.run(host='0.0.0.0')
