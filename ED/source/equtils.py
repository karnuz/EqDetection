import os
import sys
import csv
import obspy
from obspy import read
import pandas as pd


class DataLoader
def loadData():

    eventfilepath = os.path.realpath("../cahuilla_events.txt")
    eventFile = open(eventfilepath,"r")
    buf = csv.reader(eventFile)
    x = next(buf)[0].split(" ")
    
    path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
    print('hello')
    st = read(path)
    headers = list(st[0].stats.keys())
    headers.append('latitude')
    headers.append('longitude')
    headers.append('depth')
    items = list(st[0].stats.values())
    df =  pd.DataFrame(columns=headers)
    df.append(items)
    
    for line in buf:
        try:
            x = line[0].split(" ")
            path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
            print(path)
            if os.path.isfile(path):
                st = read(path)
                for i in range(len(st)):
                    items = list(st[i].stats.values())
                    items.append(x[7])
                    items.append(x[8])
                    items.append(x[10])
                    df.loc[len(df)] = items
        except:
            continue
            print(path)
    return df

#if __name__ == "__main__":
#    main()
