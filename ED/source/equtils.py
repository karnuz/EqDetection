
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
        headers_x = list(st[0].stats.keys())
        headers_y = ['depth','latitude','longitude']
        dfy = pd.DataFrame(columns = headers_y)
        #    items = list(st[0].stats.values())
        dfx =  pd.DataFrame(columns=headers_x)
            
        for line in buf:
            try:
                x = line[0].split(" ")
                path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
                print(path)
                if os.path.isfile(path):
                    st = read(path)
                    for i in range(len(st)):
                        items = list(st[i].stats.values())
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
        return dfx,dfy # return matrix of dfx and dfy ??

    def getDimensions():
        return [[self.latitudeMin,self.latitudeMax],[self.longitudeMin,self.longitudeMax],[self.depthMin,self.depthMax]]
