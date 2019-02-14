import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from numpy import random
# libraries for UBER API
from uber_rides.session import Session
from uber_rides.client import UberRidesClient
# libraries for take the capture date
import time
from datetime import datetime, timedelta
# libraries for capture data each 4 min
import threading
import requests

import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from sqlalchemy import func
from sqlalchemy import distinct

from flask import Flask, render_template, request
from flask import Flask, jsonify

import threading
#################################################
# Database Setup
#################################################
engine4 = create_engine("sqlite:///UberPrices.sqlite")
engine = create_engine("sqlite:///UberPricesNew.sqlite")
engine2 = create_engine("sqlite:///ClosebyPlaces.sqlite")
engine3 = create_engine("sqlite:///LyftPricesNew.sqlite")
# reflect an existing database into a new model
Base = automap_base()
Base2 = automap_base()
Base3 = automap_base()
Base4 = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)
Base2.prepare(engine2, reflect=True)
Base3.prepare(engine3, reflect=True)
Base4.prepare(engine4, reflect=True)
# Save reference to the table
UberPrices = Base4.classes.uberPrices
UberPricesNew = Base.classes.uberPricesNew
ClosebyPlaces = Base2.classes.closebyPlaces
LyftPricesNew = Base3.classes.lyftPricesNew
# Create our session (link) from Python to the DB
session = Session(engine)
session2 = Session(engine2)
session3 = Session(engine3)
session4 = Session(engine4)
################################################
# ML setup
################################################
import pickle
# Load from file
uber_pkl_filename = "uber_model.pkl" 
with open(uber_pkl_filename, 'rb') as file:  
    regr_uber = pickle.load(file)
    
lyft_pkl_filename = "lyft_model.pkl" 
with open(lyft_pkl_filename, 'rb') as file1:  
    regr_lyft = pickle.load(file1)
    
uberd_pkl_filename = "uberd_modeldu.pkl"  
with open(uberd_pkl_filename, 'rb') as file2:  
    modeldu = pickle.load(file2)
    
lyftd_pkl_filename = "lyftd_modeldl.pkl"  
with open(lyftd_pkl_filename, 'rb') as file3:  
    modeldl = pickle.load(file3)

# Calculate the accuracy score and predict target values
# score = pickle_model.score(Xtest, Ytest)  
# print("Test score: {0:.2f} %".format(100 * score))  
# Ypredict = pickle_model.predict(Xtest)  
#################################################
# real value
#################################################
places = [
  { "name": "Centennial Park",
  "location": [33.7603474,-84.3957012]},
  { "name": "Buckhead Bars",
  "location": [33.8439849,-84.3789694]},
  { "name": "Inman Park",
  "location": [33.7613676,-84.3623401]},
  { "name": "Stone Mountain",
  "location": [33.8053189,-84.1477255]},
  { "name": "Six Flags",
  "location": [33.7706408,-84.5537186]},
  { "name": "Statefarm Arena",
 "location": [33.7572891,-84.3963244]},
 { "name": "Zoo Atlanta",
 "location": [33.7327032,-84.3846396]},
 { "name": "Atlanta High Museum",
 "location": [33.7900632,-84.3877407]},
 { "name": "Fox Theater",
 "location": [33.7724591,-84.3879697]},
 { "name": "Virginia Highlands",
 "location": [33.7795027,-84.3757217]},
 { "name": "Shops at Buckhead",
 "location": [33.838031,-84.3821468]},
 { "name": "Emory University",
 "location": [33.7925239,-84.3261929]},
 { "name": "Georgia State University",
 "location": [33.7530724,-84.3874759]},
 { "name": "Spelman College",
 "location": [33.7463641,-84.4144874]},
 { "name": "Edgewood Bars",
 "location": [33.7544814,-84.3745674]},
  {"name": "Hartsfield Jackson Airport",
  "location": [33.6407282,-84.4277001]},
   {"name":"SunTrust Park",
   "location":[33.8908211,-84.4678309]},
   {"name":"Mercedes Benz Stadium",
   "location":[33.7554483,-84.400855]},
   {"name":"Lenox Square Mall",
    "location":[33.8462925,-84.3621419]},
   {"name":"Piedmont Park",
   "location":[33.7850856,-84.373803]},
   {"name":"Hall",
   "location":[34.3063924,-83.9791498]},
   {"name":"Forsyth",
   "location":[33.0369172,-83.9534595]},
   {"name":"Cherokee",
   "location":[34.2431482,-84.5984968]},
   {"name":"Bartow",
   "location":[34.2443826,-84.9857677]},
   {"name":"Paulding",
   "location":[33.928889,-85.0268574]},
   {"name":"Douglas",
   "location":[33.6899886,-84.8846799]},
   {"name":"Coweta",
   "location":[33.3516087,-84.8962848]},
   {"name":"Fayette",
   "location":[33.4039244,-84.6445094]},
   {"name":"Spalding",
   "location":[33.2658534,-84.4391148]},
   {"name":"Butts",
   "location":[33.3126177,-84.105586]},
   {"name":"Newton",
   "location":[33.5559292,-84.0049789]},
   {"name":"Walton",
   "location":[33.7635877,-83.8840724]},
   {"name":"Gwinett",
   "location":[33.960546,-84.178047]},
   {"name":"Fulton",
   "location":[33.8446039,-84.7543888]},
   {"name":"Cobb",
   "location":[33.9126755,-84.6972663]},
   {"name":"Clayton",
   "location":[33.5008779,-84.491477]},
   {"name":"Henry",
   "location":[33.4727666,-84.278534]},
   {"name":"Rockdale",
   "location":[33.6561613,-84.1887709]},
   {"name":"Dekalb",
   "location":[33.7929946,-84.327053]}]

from uber_rides.session import Session
ubersession = Session(server_token="TZ9aAN7GMzp49djfXoMil2HJ7XxCs0Zwo8EWXd88")

from uber_rides.client import UberRidesClient
client = UberRidesClient(ubersession)

def realtime_lyft_results(dest,ptype):
    for place in places:
        if place["name"] == dest:
            lat = place["location"][0]
            lon = place["location"][1]
            
    url="https://api.lyft.com/v1/cost?start_lat=33.7762&start_lng=-84.3895&end_lat=" +str(lat)+ "&end_lng="+str(lon)
    estimates = requests.get(url).json()["cost_estimates"]
    # print(estimates)
    for estimate in estimates:
        if estimate["display_name"] == ptype:
            low = (estimate["estimated_cost_cents_min"])//100
            high = (estimate["estimated_cost_cents_max"])//100
            avg = (low + high) / 2
            return avg
        
def realtime_uber_results(dest,ptype):
    for place in places:
        if place["name"] == dest:
            lat = place["location"][0]
            lon = place["location"][1]
            
    response = client.get_price_estimates(
            start_latitude=33.7762,
            start_longitude=-84.3895,
            end_latitude=lat,
            end_longitude=lon
        )
            
    estimates = response.json.get('prices')
    # print(estimates)
    for estimate in estimates:
        if estimate["display_name"] == ptype:
            low = (estimate["low_estimate"])
            high = (estimate["high_estimate"])
            avg = (low + high) / 2
            return avg

def convert_type(ptype):
    ptypes = []
    if ptype == "Black":
        ptypes = ["Black","Lux Black"]
    if ptype == "Black SUV":
        ptypes = ["Black SUV","Lux Black XL"]
    if ptype == "Select":
        ptypes = ["Select","Lux"]
    if ptype == "UberPool":
        ptypes = ["UberPool","Shared"]
    if ptype == "UberX":
        ptypes = ["UberX","Lyft"]
    if ptype == "UberXL":
        ptypes = ["UberXL","Lyft XL"]
    return ptypes



#################################################
# selections
#############################################3###
dist_dict = {'Atlanta High Museum': 0, 'Bartow':1,
       'Buckhead Bars':2, 'Butts':3, 'Centennial Park':4,
       'Cherokee':5, 'Clayton':6, 'Cobb':7, 'Coweta':8,
       'Dekalb':9, 'Douglas':10, 'Edgewood Bars':11,
       'Emory University':12, 'Fayette':13, 'Forsyth':14,
       'Fox Theater':15, 'Fulton':16, 'Georgia State University':17,
       'Gwinett':18, 'Hall':19, 'Hartsfield Jackson Airport':20,
       'Henry':21, 'Inman Park':22, 'Lenox Square Mall':23,
       'Mercedes Benz Stadium':24, 'Newton':25, 'Paulding':26,
       'Piedmont Park':27, 'Rockdale':28, 'Shops at Buckhead':29,
       'Six Flags':30, 'Spalding':31, 'Spelman College':32,
       'Statefarm Arena':33, 'Stone Mountain':34, 'SunTrust Park':35,
       'Virginia Highlands':36, 'Walton':37, 'Zoo Atlanta':38,
       '00:00':39, '01:00':40, '02:00':41, '03:00':42, '04:00':43,
       '05:00':44, '06:00':45, '07:00':46, '08:00':47, '09:00':48,
       '10:00':49, '11:00':50, '12:00':51, '13:00':52, '14:00':53,
       '15:00':54, '16:00':55, '17:00':56, '18:00':57, '19:00':58,
       '20:00':59, '21:00':60, '22:00':61, '23:00':62}





def predict_uber_distance(dest,time):
    dist_pred_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for key,value in dist_dict.items():
        if key == dest:
            dist_pred_array[value]=1
        if key == time:
            dist_pred_array[value]=1
    print(dist_pred_array)
    dist = modeldu.predict([dist_pred_array])
    return dist[0][0]

def predict_lyft_distance(dest,time):
    dist_pred_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for key,value in dist_dict.items():
        if key == dest:
            dist_pred_array[value]=1
        if key == time:
            dist_pred_array[value]=1
    dist = modeldl.predict([dist_pred_array])
    return dist[0][0]
    
uber_pred_dict = {'dist':0,'Atlanta High Museum': 1, 'Bartow':2,
       'Buckhead Bars':3, 'Butts':4, 'Centennial Park':5,
       'Cherokee':6, 'Clayton':7, 'Cobb':8, 'Coweta':9,
       'Dekalb':10, 'Douglas':11, 'Edgewood Bars':12,
       'Emory University':13, 'Fayette':14, 'Forsyth':15,
       'Fox Theater':16, 'Fulton':17, 'Georgia State University':18,
       'Gwinett':19, 'Hall':20, 'Hartsfield Jackson Airport':21,
       'Henry':22, 'Inman Park':23, 'Lenox Square Mall':24,
       'Mercedes Benz Stadium':25, 'Newton':26, 'Paulding':27,
       'Piedmont Park':28, 'Rockdale':29, 'Shops at Buckhead':30,
       'Six Flags':31, 'Spalding':32, 'Spelman College':33,
       'Statefarm Arena':34, 'Stone Mountain':35, 'SunTrust Park':36,
       'Virginia Highlands':37, 'Walton':38, 'Zoo Atlanta':39,
       'Black':40, 'Black SUV':41, 'Select':42,
       'UberPool':43, 'UberX':44, 'UberXL':45,
       '00:00':46, '01:00':47, '02:00':48, '03:00':49, '04:00':50,
       '05:00':51, '06:00':52, '07:00':53, '08:00':54, '09:00':55,
       '10:00':56, '11:00':57, '12:00':58, '13:00':59, '14:00':60,
       '15:00':61, '16:00':62, '17:00':63, '18:00':64, '19:00':65,
       '20:00':66, '21:00':67, '22:00':68, '23:00':69}    

lyft_pred_dict = {'dist':0,'Atlanta High Museum': 1, 'Bartow':2,
       'Buckhead Bars':3, 'Butts':4, 'Centennial Park':5,
       'Cherokee':6, 'Clayton':7, 'Cobb':8, 'Coweta':9,
       'Dekalb':10, 'Douglas':11, 'Edgewood Bars':12,
       'Emory University':13, 'Fayette':14, 'Forsyth':15,
       'Fox Theater':16, 'Fulton':17, 'Georgia State University':18,
       'Gwinett':19, 'Hall':20, 'Hartsfield Jackson Airport':21,
       'Henry':22, 'Inman Park':23, 'Lenox Square Mall':24,
       'Mercedes Benz Stadium':25, 'Newton':26, 'Paulding':27,
       'Piedmont Park':28, 'Rockdale':29, 'Shops at Buckhead':30,
       'Six Flags':31, 'Spalding':32, 'Spelman College':33,
       'Statefarm Arena':34, 'Stone Mountain':35, 'SunTrust Park':36,
       'Virginia Highlands':37, 'Walton':38, 'Zoo Atlanta':39,
       'Lux':40, 'Lux Black':41,
       'Lux Black XL':42, 'Lyft':43,
       'Lyft XL':44, 'Shared':45,
       '00:00':46, '01:00':47, '02:00':48, '03:00':49, '04:00':50,
       '05:00':51, '06:00':52, '07:00':53, '08:00':54, '09:00':55,
       '10:00':56, '11:00':57, '12:00':58, '13:00':59, '14:00':60,
       '15:00':61, '16:00':62, '17:00':63, '18:00':64, '19:00':65,
       '20:00':66, '21:00':67, '22:00':68, '23:00':69} 
    
    


def select_type(ptype):
    if ptype == "Black":
        pred_array[40] = 1
    if ptype == "Black SUV":
        pred_array[41] = 1
    if ptype == "Select":
        pred_array[42] = 1
    if ptype == "UberPool":
        pred_array[43] = 1
    if ptype == "UberX":
        pred_array[44] = 1
    if ptype == "UberXL":
        pred_array[45] = 1
        



#################################################
# Flask Setup
#################################################
app = Flask(__name__)


#################################################
# Flask Routes
#################################################

#@app.route("/")
#def welcome():
 #   """List all available api routes."""
 #   return (
 #       f"Available Routes:<br/>"
 #       f"/api/v1.0/names<br/>"
 #       f"/api/v1.0/passengers"
 #   )

#@app.route("/api/v1.0/names")
#def names():
   # """Return a list of all passenger names"""
    # Query all passengers
    #results = session.query(UberPrices.place, UberPrices.lat, UberPrices.lon, UberPrices.dist, UberPrices.display_name, 
    #                        UberPrices.product_id, UberPrices.duration, UberPrices.estimate).all()

    # Convert list of tuples into normal list
    #all_names = list(np.ravel(results))

    #return jsonify(results)


#@app.route("/api/v1.0/passengers")
#def passengers():
    #"""Return a list of passenger data including the name, age, and sex of each passenger"""
    # Query all passengers
    #results = session.query(Passenger).all()

    # Create a dictionary from the row data and append to a list of all_passengers
    #all_passengers = []
    #for passenger in results:
    #    passenger_dict = {}
    #    passenger_dict["name"] = passenger.name
     #   passenger_dict["age"] = passenger.age
    #    passenger_dict["sex"] = passenger.sex
    #    all_passengers.append(passenger_dict)

    #return jsonify(all_passengers)

results = session4.query(UberPrices.place, UberPrices.lat, UberPrices.lon, UberPrices.dist, UberPrices.display_name, 
                            UberPrices.product_id, UberPrices.duration, UberPrices.estimate, UberPrices.time).all()
# count distinct "name" values
myplaces = session4.query(distinct(UberPrices.place)).order_by(UberPrices.dist).all()
mytimes = session4.query(distinct(UberPrices.time)).all()
mytypes = session4.query(distinct(UberPrices.display_name)).all()

x = session4.query(UberPrices.duration).all()
v = session4.query(UberPrices.dist).all()

chartdata = session4.query(UberPrices.place, UberPrices.duration, UberPrices.high_estimate, UberPrices.low_estimate, UberPrices.dist, UberPrices.time, UberPrices.display_name).all()
#chartdata2 = session.query(UberPrices.place, UberPrices.duration, UberPrices.high_estimate, UberPrices.low_estimate, UberPrices.dist, UberPrices.time, UberPrices.display_name).all()

nearbyplaces = session2.query(ClosebyPlaces.place, ClosebyPlaces.mlat, ClosebyPlaces.mlon, ClosebyPlaces.lat, ClosebyPlaces.lon, ClosebyPlaces.name, ClosebyPlaces.vicinity).all()
#below not working
#barchartdata = Session.query(UberPrices.place, UberPrices.duration, UberPrices.display_name, UberPrices.high_estimate, UberPrices.low_estimate, UberPrices.dist, #UberPrices.time).group_by(UberPrices.place).group_by(UberPrices.time).all()

#c.count = Session.query(func.count(Person.id)).scalar()
 
#c.avg = Session.query(func.avg(Person.id).label('average')).scalar()
       
#c.sum = Session.query(func.sum(Person.id).label('sum')).scalar()
        
#c.max = Session.query(func.max(Person.id).label('max')).scalar() 
        
#c.coutg = Session.query(func.count(Person.id).label('count'), Person.name ).group_by(Person.name).all()
pred_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
@app.route("/placesinfo")
def placesinfo():
    #results = session.query(UberPrices.place, UberPrices.lat, UberPrices.lon, UberPrices.dist, UberPrices.display_name, 
    #                       UberPrices.product_id, UberPrices.duration, UberPrices.estimate).all()
    # data = db.session.query(uberPrices).all()
    return jsonify(nearbyplaces)

@app.route("/chartsinfo")
def chartsinfo():
    total_chart_info = [ myplaces, mytimes, mytypes, chartdata]
    
    return jsonify(total_chart_info)


@app.route("/data")
def data():
    #results = session.query(UberPrices.place, UberPrices.lat, UberPrices.lon, UberPrices.dist, UberPrices.display_name, 
    #                       UberPrices.product_id, UberPrices.duration, UberPrices.estimate).all()
    data = jsonify(results)
    
    # data = db.session.query(uberPrices).all()
    return jsonify(results)
    #return render_template("index.html")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/map")
def map():
    return render_template("map.html")

@app.route("/prediction1", methods=['GET','POST'])
def prediction1():
    
    if request.method == 'POST':
        dest = request.form['select-key']
        time = request.form['select-time']
        uber_dist = predict_uber_distance(dest,time)
        print(uber_dist)
        lyft_dist = predict_lyft_distance(dest,time)
        print(lyft_dist)
        pred_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # pr = regr.predict([[1.32,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        dest = request.form['select-key']
        # select_place(dest)
        if dest == "Atlanta High Museum":
            pred_array[1] = 1
        if dest == "Bartow":
            pred_array[2] = 1
        if dest == "Buckhead Bars":
            pred_array[3] = 1
        if dest == "Butts":
            pred_array[4] = 1
        if dest == "Centennial Park":
            pred_array[5] = 1
        if dest == "Cherokee":
            pred_array[6] = 1
        if dest == "Clayton":
            pred_array[7] = 1
        if dest == "Cobb":
            pred_array[8] = 1
        if dest == "Coweta":
            pred_array[9] = 1
        if dest == "Dekalb":
            pred_array[10] = 1
        if dest == "Douglas":
            pred_array[11] = 1
        if dest == "Edgewood Bars":
            pred_array[12] = 1
        if dest == "Emory University":
            pred_array[13] = 1
        if dest == "Fayette":
            pred_array[14] = 1
        if dest == "Forsyth":
            pred_array[15] = 1
        if dest == "Fox Theather":
            pred_array[16] = 1
        if dest == "Fulton":
            pred_array[17] = 1
        if dest == "Georgia State University":
            pred_array[18] = 1
        if dest == "Gwinett":
            pred_array[19] = 1
        if dest == "Hall":
            pred_array[20] = 1
        if dest == "Hartsfield Jackson Airport":
            pred_array[21] = 1
        if dest == "Henry":
            pred_array[22] = 1
        if dest == "Inman Park":
            pred_array[23] = 1
        if dest == "Lenox Square Mall":
            pred_array[24] = 1
        if dest == "Mercedes Benz Stadium":
            pred_array[25] = 1
        if dest == "Newton":
            pred_array[26] = 1
        if dest == "Paulding":
            pred_array[27] = 1
        if dest == "Piedmont Park":
            pred_array[28] = 1
        if dest == "Rockdale":
            pred_array[29] = 1
        if dest == "Shops at Buckhead":
            pred_array[30] = 1
        if dest == "Six Flags":
            pred_array[31] = 1
        if dest == "Spalding":
            pred_array[32] = 1
        if dest == "Spelman College":
            pred_array[33] = 1
        if dest == "Statefarm Arena":
            pred_array[34] = 1
        if dest == "Stone Mountain":
            pred_array[35] = 1
        if dest == "SunTrust Park":
            pred_array[36] = 1
        if dest == "Virginia Highlands":
            pred_array[37] = 1
        if dest == "Walton":
            pred_array[38] = 1
        if dest == "Zoo Atlanta":
            pred_array[39] = 1
        time = request.form['select-time']
        # select_time(time)
        if time == "00:00":
            pred_array[46] = 1
        if time == "01:00":
            pred_array[47] = 1
        if time == "02:00":
            pred_array[48] = 1
        if time == "03:00":
            pred_array[49] = 1
        if time == "04:00":
            pred_array[50] = 1
        if time == "05:00":
            pred_array[51] = 1
        if time == "06:00":
            pred_array[52] = 1
        if time == "07:00":
            pred_array[53] = 1
        if time == "08:00":
            pred_array[54] = 1
        if time == "09:00":
            pred_array[55] = 1
        if time == "10:00":
            pred_array[56] = 1
        if time == "11:00":
            pred_array[57] = 1
        if time == "12:00":
            pred_array[58] = 1
        if time == "13:00":
            pred_array[59] = 1
        if time == "14:00":
            pred_array[60] = 1
        if time == "15:00":
            pred_array[61] = 1
        if time == "16:00":
            pred_array[62] = 1
        if time == "17:00":
            pred_array[63] = 1
        if time == "18:00":
            pred_array[64] = 1
        if time == "19:00":
            pred_array[65] = 1
        if time == "20:00":
            pred_array[66] = 1
        if time == "21:00":
            pred_array[67] = 1
        if time == "22:00":
            pred_array[68] = 1
        if time == "23:00":
            pred_array[69] = 1
        uber_pred_array = pred_array.copy()
        lyft_pred_array = pred_array.copy()
        ptype = request.form['select-type']
        print(ptype)
        # select_type(ptype)
        if ptype == "Black":
            uber_pred_array[40] = 1
            lyft_pred_array[41] = 1
        if ptype == "Black SUV":
            uber_pred_array[41] = 1
            lyft_pred_array[42] = 1
        if ptype == "Select":
            uber_pred_array[42] = 1
            lyft_pred_array[40] = 1
        if ptype == "UberPool":
            uber_pred_array[43] = 1
            lyft_pred_array[45] = 1
        if ptype == "UberX":
            uber_pred_array[44] = 1
            lyft_pred_array[43] = 1
        if ptype == "UberXL":
            uber_pred_array[45] = 1
            lyft_pred_array[44] = 1
            
        uber_pred_array[0] = uber_dist
        lyft_pred_array[0] = lyft_dist
        print(pred_array)
        print(uber_pred_array)
        print(lyft_pred_array)
        pr_uber = regr_uber.predict([uber_pred_array])
        pr_lyft = regr_lyft.predict([lyft_pred_array])
        if pr_uber[0] < pr_lyft[0]:
            result = "Uber"
        else:
            result = "Lyft"
        projectpath = [dest,time,ptype]  
        resultlist = [result,pr_uber[0],pr_lyft[0], projectpath]
        ########################
        # realtime results
        ########################
        ptypes = convert_type(ptype)
        print(dest)
        print(ptypes)
        lyft_estimate = realtime_lyft_results(dest,ptypes[1])
        print(lyft_estimate)
        uber_estimate = realtime_uber_results(dest,ptypes[0])
        realtime = [uber_estimate, lyft_estimate]
        
        
        return render_template("prediction1.html", resultlist=resultlist, realtime = realtime)
    else:
        projectpath = "Plese select destination, time and type of ride to make a prediction"
        result = ""
        resultlist = [projectpath,result,result]
        return render_template("prediction1.html", resultlist=resultlist)

    

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    posts = Post.query.all()
    form = PostForm()
    if form.validate_on_submit():
        new_post = Post(request.form['title'], request.form['body'])
        db.session.add(new_post)
        db.session.commit()
        return redirect('/')
    return render_template('index.html', posts=posts, form=form)


@app.route("/chart2")
def chart2():
    # Create a dictionary from the row data and append to a list of chartinfo
    chartinfo = []
    for data in chartdata:
        data_dict = {}
        data_dict["place"] = data.place
        data_dict["duration"] = data.duration
        data_dict["high_estimate"] = data.high_estimate
        data_dict["low_estimate"] = data.low_estimate
        data_dict["distance"] = data.dist
        data_dict["time"] = data.time
        chartinfo.append(data_dict)
    #rows = session.query(Person).count()
    times = []
    for time in mytimes:
        times.append(time[0])
    
    places = []
    for place in myplaces:
        places.append(place[0])
    
    types = []
    for type in mytypes:
        types.append(type[0])
        
        
    
    #return jsonify(all_passengers)
    return render_template("chart2.html", chartdata = chartdata, chartinfo = chartinfo, places = places, times = times, types = types)




if __name__ == "__main__":
    app.run(debug=True)