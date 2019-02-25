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
from obspy import core

class ProcessData:
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

    def oneMonthData(self):
        dfs = []
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701'):
            filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/'+filename
            st = read(filead)
            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                items = list(st[i].stats.values())
                dfx.loc[len(dfx)] = items
            dfs.append(dfx)
        return dfs

    def getDimensions():
        return [[self.latitudeMin,self.latitudeMax],[self.longitudeMin,self.longitudeMax],[self.depthMin,self.depthMax]]


    def filterStream():
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701'):
            filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/'+filename
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
        sts = ['PALA', 'DNR', 'LKH', 'RDM', 'PFO', 'LVA2', 'TRO', 'PLM', 'B082', 'WMC', 'PGA', 'B093', 'KNW', 'FRD', 'POB2', 'BAC', 'MOR', 'SND', 'HMT2', 'MTG', 'DGR', 'B086', 'BZN', 'GVDA', 'CRY', 'CSH', 'B087']
        filedir = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/'
        #os.mkdir(filedir+'new')
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701'):
            try:
                if not os.path.isdir(filename):
                    filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/'+filename
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
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new'):
            filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new/'+filename
            st = read(filead)
            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                items = list(st[i].stats.values())
                dfx.loc[len(dfx)] = items
            dfs.append(dfx)
        return dfs

    def oneMonthFilterData(self):
        dfs = []
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new'):
            filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new/'+filename
            st = read(filead)
            dfx = pd.DataFrame(columns = ['sampling_rate', 'delta', 'starttime', 'endtime', 'npts', 'calib', 'network', 'station', 'location', 'channel', 'mseed', '_format'])
            for i in range(len(st)):
                items = list(st[i].stats.values())
                dfx.loc[len(dfx)] = items
            dfs.append(dfx)
        return dfs

    def oneMonthFilterDataDonwSample(self):
        dfs = []
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new'):
            filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new/'+filename
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
        for filename in os.listdir('/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new'):
            filead = '/Users/himanshusharma/karnuz/Rose/ED/eqdata/2017/201701/new/'+filename
            st = read(filead)
            if not st.count()==201:
                os.remove(filead)
        return 0
