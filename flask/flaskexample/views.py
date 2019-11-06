from flask import render_template
from flask import request, redirect, url_for
from flaskexample import app
from flaskexample.a_Model import ModelIt
import pandas as pd
import numpy as np
from recommender import CosSimilarityRecommender
import googlemaps
import time

gmaps = googlemaps.Client(key='#')

dct = {}

#Load list of all topics
topics = list(np.load('topics.npy'))

#Load the data for the similarity
mn = np.load('mn.npy')
sig = np.load('sig.npy')
similarity = np.load('similarity.npy')


#Find the Cos Similarity matrix
csr = CosSimilarityRecommender(similarity, mn, sig)

#take top 20 most popular hobbies, select three of those at random for initial recommendation
top_hobbies = topics[:20]
offered_hobbies = np.random.choice(top_hobbies, 3)

@app.route('/')
@app.route('/index')
def index():
    return redirect(url_for('hobby_input'))

@app.route('/presentation')
def presentation():
    return '''
           <html>
            <head>
              <title>Home Page</title>
            </head>
            <body>
              <a href=https://docs.google.com/presentation/d/1FerxZDk3CzyuyF0-si5z4Dduru7kUKuBp1mh-TSgFrs/edit?usp=sharing>Presentation</a>
            </body>
           </html>
         '''

@app.route('/input')
def hobby_input():
    global offered_hobbies
    offered_hobbies = np.random.choice(top_hobbies, 3, replace=False)
    return render_template("input.html", hobbies = offered_hobbies)

@app.route('/output')
def hobby_output():
    #this dictionary will containt hobbies user rated
    global dct
    options = {"up":1, "down":0}
    global offered_hobbies
    #check which hobbies were selected
    #put 1 for the one rated positively, 0 for one rated negatively
    #unrated hobbies don't go to dct
    for hobby in offered_hobbies:
        try:
            value = request.args.get(hobby.replace(' ', '-'))
            dct[hobby] = options[value]
        except:
            pass
    try:
        positionMuseum = request.args.get('museum')
        print(positionMuseum)
    except:
        positionMuseum = None

    #run the recommender
    the_result = ModelIt(value, user=dct, topics=topics, csr=csr, mn=mn, sig=sig)
    #new top 3 recommendations
    offered_hobbies = list(the_result.keys())

    #Search for activities in the area
    positions = []
    for hobby in offered_hobbies:
        qu = gmaps.places_nearby(location=(42.3501, -71.0496), radius=2000, keyword=hobby) #Boston South Station is the default
        positions.append(qu['results'][:3])

    return render_template("output.html", hobbies = offered_hobbies, positions = positions)
