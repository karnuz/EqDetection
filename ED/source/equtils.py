
import os
import sys
import csv
import obspy
from obspy import read
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import TensorDataset
import itertools
from obspy.core import UTCDateTime


class DataLoad:
    def __init__(self,eventfilepath = "../cahuilla_events.txt"):
        self.eventfileP = os.path.realpath("../cahuilla_events.txt")

    def loadData(self):
        eventFile = open(self.eventfileP,"r")
        buf = csv.reader(eventFile)
        x = next(buf)[0].split(" ")
                
        path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
        print('hello')
        st = read(path)
        headers_x = list(st[0].stats.keys())[0:10]
        print(headers_x)
        headers_y = ['depth','latitude','longitude']
        dfy = pd.DataFrame(columns = headers_y)
        dfx =  pd.DataFrame(columns = headers_x)
        data = []
        label = []
        for line in buf:
            try:
                x = line[0].split(" ")
                path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/new/" + x[6]
                dt = UTCDateTime(x[0]+'-'+x[1]+'-'+x[2]+'T'+x[3]+':'+x[4]+':'+x[5])
                if os.path.isfile(path):
                    st = read(path)
                    st = st.slice(dt-2,dt+18)
                    dp = np.empty([len(st),len(st[0].data)])
                    lp = [float(x[10]),float(x[7]),float(x[8])]
                    #print(len(st))
                    for i in range(len(st)):
                        dp[i] = st[i].data
                        items = list(st[i].stats.values())
                        values = items[0:10]
#                        values.append(items[9])
                        #print(values)
                        dfy.loc[len(dfy)] = [x[10],x[7],x[8]]
                        dfx.loc[len(dfx)] = values
                    label.append(lp)
                    data.append(dp)
                    
            except Exception as ex:
                print(ex)

        self.latitudeMax = float(dfy['latitude'].max())
        self.latitudeMin = float(dfy['latitude'].min())
        self.longitudeMax = float(dfy['longitude'].max())
        self.longitudeMin = float(dfy['longitude'].min())
        self.depthMax = float(dfy['depth'].max())
        self.depthMin = float(dfy['depth'].min())
        self.dfx = dfx
        self.dfy = dfy
        return np.asarray(data),label # return matrix of dfx and dfy ??

    def getDimensions(self):
        return [[self.depthMin,self.depthMax],[self.latitudeMin,self.latitudeMax],[self.longitudeMin,self.longitudeMax]]
