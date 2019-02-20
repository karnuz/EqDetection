import os
import sys
import csv
import obspy
from obspy import read
import pandas as pd


class DataLoader:
    def __init__(eventfilepath = "../cahuilla_events.txt"):
        self.eventfileP = os.path.realpath(eventfilepath)

    def loadData(self):
        eventFile = open(eventfileP,"r")
        buf = csv.reader(eventFile)
        x = next(buf)[0].split(" ")
        
        path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
        print('hello')
        st = read(path)
        headers_x = list(st[0].stats.keys())
        headers_y = ['latitude','longitude','depth']
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
                        dfy.loc[len(dfy)] = [x[7],x[8],x[10]]
                        df.loc[len(dfx)] = items
            except:
                continue
                print(path)

        self.latitudeMax = dfy['latitude'].max()
        self.latitudeMin = dfy['latitude'].min()
        self.longitudeMax = dfy['longitude'].max()
        self.longitudeMin = dfy['longitude'].min()
        self.depthMax = dfy['depth'].max()
        self.depthMin = dfy['depth'].min()
        self.dfx = dfx
        self.dfy = dfy
        return df # return matrix of dfx and dfy ??

    def getDimensions():
        return [[self.latitudeMin,self.latitudeMax],[self.longitudeMin,self.longitudeMax],[self.depthMin,self.depthMax]]

#if __name__ == "__main__":
#    main()
