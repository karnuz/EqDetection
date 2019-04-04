import os
import sys
import csv
import obspy
import math
from obspy import read
import pandas as pd
import pickle
import numpy as np
#from torch.utils.data import DataLoader
#import torch
#from torch.utils.data import TensorDataset
import itertools
from obspy import core

""""
   Run the following commands:
    
from processdata import ProcessData
o = ProcessData()
len(o.filtered_stations)
data = o.oneMonthData()
for s in data: print(len(s['station']))

stations_for_events = [len(set(s['station'])) for s in data]
traces_for_events = [len(s['station']) for s in data]


Taking,
speed = 1 km/s
time = 20 secs (-2 to 18secs)

# total station = 2363
# filtered stations = 73

for 49 events-
# traces_for_events =
[130, 121, 30, 108, 105, 109, 100, 105, 58, 130, 103, 85, 121, 110, 88, 88, 108, 70, 111, 85, 105, 73, 108, 58, 88, 108, 85, 121, 122, 88, 105, 130, 64, 115, 58, 121, 85, 109, 108, 64, 58, 121, 58, 107, 85, 109, 76, 58, 79]

# unique stations_for_events =
[20, 19, 5, 17, 17, 18, 15, 17, 9, 20, 16, 13, 19, 17, 14, 14, 17, 11, 18, 13, 17, 11, 17, 9, 14, 18, 13, 19, 18, 14, 17, 20, 10, 19, 9, 19, 13, 18, 17, 10, 9, 19, 9, 17, 13, 18, 11, 9, 12]

    """

class ProcessData:
    def __init__(self, filepath = "../cahuilla_events.txt"):
        self.events_path = os.path.realpath(filepath)
        
        self.boundingBox()
        self.filter_stations_in_region()

    def loadData(self):
        eventFile = open(self.events_path,"r")
        buf = csv.reader(eventFile)
        x = next(buf)[0].split(" ")
                
        path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
        print('hello')
        st = read(path)
        headers_x = list(st[0].stats.keys())
        headers_y = ['depth','latitude','longitude']
        
        #traces: to strore event waves
        dfy = pd.DataFrame(columns = headers_y)
        
        #to store location of events
        dfx =  pd.DataFrame(columns=headers_x)
            
        for line in buf:
            try:
                x = line[0].split(" ")
                path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
                print(path)
                if os.path.isfile(path):
                    st = read(path)
                    for i, trace in enumerate(st):
                        items = list(trace.stats.values())
                        dfy.loc[len(dfy)] = [x[10],x[7],x[8]]
                        dfx.loc[len(dfx)] = items
            except:
                continue

        self.latitudeMax = dfy['latitude'].max()
        self.latitudeMin = dfy['latitude'].min()
        self.longitudeMax = dfy['longitude'].max()
        self.longitudeMin = dfy['longitude'].min()
        self.depthMax = dfy['depth'].max()
        self.depthMin = dfy['depth'].min()
        self.dfx = dfx
        self.dfy = dfy

    def oneMonthData(self):
        dfs =[]
        for filename in os.listdir('../eqdata/2017/201701'):
            filead = '../eqdata/2017/201701/'+filename
            st = read(filead)
            headers_x = list(st[0].stats.keys()) # ['sampling_rate', 'delta', 'starttime','endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format']
            dfx = pd.DataFrame(columns = headers_x)
            for i in range(len(st)):
                items = list(st[i].stats.values())
                # If trace is from a filtered station
                if items[7] in self.filtered_stations:
                    dfx.loc[len(dfx)] = items
            dfs.append(dfx)
    
        return dfs

    def boundingBox(self, filepath = "../cahuilla_events.txt"):
        eventFile = open(self.events_path,"r")
        buf = csv.reader(eventFile)
        
        headers_y = ['depth','latitude','longitude']
        #to store location of events
        dfy = pd.DataFrame(columns = headers_y)
        
        for line in buf:
            x = line[0].split(" ")
            dfy.loc[len(dfy)] = [x[10],x[7],x[8]]
        
        self.latitudeMax = dfy['latitude'].max()
        self.latitudeMin = dfy['latitude'].min()
        self.longitudeMax = dfy['longitude'].max()
        self.longitudeMin = dfy['longitude'].min()
        self.depthMax = dfy['depth'].max()
        self.depthMin = dfy['depth'].min()
            
        print(self.latitudeMax, self.latitudeMin, self.longitudeMax,
              self.longitudeMin, self.depthMax, self.depthMin)
    
    
    def filter_stations_in_region(self, stations_file= "../scsn_stations.txt"):
        file = open(stations_file, "r")
        buf = csv.reader(file)
        headers = next(buf)[0].split()
        print(headers)
        self.filtered_stations = []
        
        # Latitude: 1 deg = 110.574 km
        # Longitude: 1 deg = 111.320*cos(latitude)
        # Considering the speed of the wave = 1 km/s and the sampling time is 20secs, distance = 10Km
        latitude_margin = (2 * 20) / 110.574
        longitude_margin = (2 * 20) / (math.cos(math.radians(latitude_margin)) * 111.32)
        
        # 608 stations out of 2363 stations
        
        i=0
        for line in buf:
            i+=1
            x = line[0].split(" ")
            if float(x[2]) <= (float(self.latitudeMax) + latitude_margin) and \
                float(x[2]) >= (float(self.latitudeMin) - latitude_margin) and \
                float(x[3]) <= (float(self.longitudeMax) + longitude_margin) and \
                float(x[3]) <= (float(self.longitudeMin) - longitude_margin):
                self.filtered_stations.append(x[1])
        print('total stations: ', i)
            
        return self.filtered_stations

    def plotMap(self, points=[]):
        
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        fig = plt.figure(figsize=(8, 8))
        m = Basemap(projection='lcc', resolution=None,
                    width=8E6, height=8E6,
                    lat_0=45, lon_0=-100,)
        m.etopo(scale=0.5, alpha=0.5)

        # Map (long, lat) to (x, y) for plotting
        bounding_box = [[self.latitudeMin,self.longitudeMin],
                        [self.latitudeMin,self.longitudeMax],
                        [self.latitudeMax,self.longitudeMax],
                        [self.latitudeMax,self.longitudeMin]]
        for lat, long in bounding_box[:1]:
            plt.plot(long, lat, 'ok', markersize=10)
            plt.text(long, lat, ' box', fontsize=12);

        for lat, long in points:
            plt.plot(long, lat, 'ok', markersize=5)
            plt.text(long, lat, ' Seattle', fontsize=12);


    def getDimensions():
        return [[self.latitudeMin, self.latitudeMax],
                [self.longitudeMin, self.longitudeMax],
                [self.depthMin, self.depthMax]]

    def filterStream():
        for filename in os.listdir('../eqdata/2017/201701/'+filename):
            st = read(filead)
            filterdf = st.select(station.isin(sts))

            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                items = list(st[i].stats.values())
                dfx.loc[len(dfx)] = items
            dfs.append(dfx)
        return dfs    
    
    
    sts = ['PALA', 'DNR', 'LKH', 'RDM', 'PFO', 'LVA2', 'TRO', 'PLM', 'B082', 'WMC', 'PGA', 'B093', 'KNW', 'FRD', 'POB2', 'BAC', 'MOR', 'SND', 'HMT2', 'MTG', 'DGR', 'B086', 'BZN', 'GVDA', 'CRY', 'CSH', 'B087']
    
    def getStationTraces(self):
        sts = ['PALA', 'DNR', 'LKH', 'RDM', 'PFO', 'LVA2', 'TRO',
               'PLM', 'B082', 'WMC', 'PGA', 'B093', 'KNW', 'FRD',
               'POB2', 'BAC', 'MOR', 'SND', 'HMT2', 'MTG', 'DGR',
               'B086', 'BZN', 'GVDA', 'CRY', 'CSH', 'B087']
        filedir = '../eqdata/2017/201701/'
        #os.mkdir(filedir+'new')
        for filename in os.listdir('../eqdata/2017/201701'):
            try:
                if not os.path.isdir(filename):
                    filead = '../eqdata/2017/201701/'+filename
                    st = read(filead)
                    st2 = core.stream.Stream(traces = [])
                    for i in range(len(st)):
                        if st[i].stats['station'] in sts:
                            st2.append(st[i])
                    st2.write(filedir+'/new/'+filename,format='MSEED')
            except:
                print(filename)

    def readstationfiles(self):
        dfs = []
        for filename in os.listdir('../eqdata/2017/201701/new'):
            filead = '../eqdata/2017/201701/new/'+filename
            st = read(filead)
            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                items = list(st[i].stats.values())
                dfx.loc[len(dfx)] = items
            dfs.append(dfx)
        return dfs

    def oneMonthFilterData(self):
        dfs = []
        for filename in os.listdir('../eqdata/2017/201701/new'):
            filead = '../eqdata/2017/201701/new/'+filename
            st = read(filead)
            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                items = list(st[i].stats.values())
                dfx.loc[len(dfx)] = items
            dfs.append(dfx)
        return dfs

    def oneMonthFilterDataDonwSample(self):
        dfs = []
        for filename in os.listdir('../eqdata/2017/201701/new'):
            filead = '../eqdata/2017/201701/new/'+filename
            st = read(filead)
            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                if st[i].stats['sampling_rate']==200:
                   st[i].decimate(10)
                if st[i].stats['sampling_rate'] == 40:
                    st[i].decimate(2)
                if st[i].stats['sampling_rate'] == 100:
                    st[i].decimate(5)
            st.write(filead,format='MSEED')

    def remove(self):
        for filename in os.listdir('../eqdata/2017/201701/new'):
            filead = '../eqdata/2017/201701/new/'+filename
            st = read(filead)
            if not st.count()==201:
                os.remove(filead)
        return 0

