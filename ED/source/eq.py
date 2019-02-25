import torch
import torch.nn as nn
import torch.nn.functional as F
from equtils import DataLoad
import numpy as np


class EQ:

    def __init__(self):
        self.r = 25   #radius of gaussian distribution
        self.dimensions = [30,30,30]
        
    def makeGrid(self,region, dimensions):
        grid = np.empty(dimensions)
        depths, depthstep = np.linspace(region[0][0],region[0][1], dimensions[0],endpoint=False,retstep=True)
        self.depths = depths + depthstep/2
        latitudes, latstep = np.linspace(region[1][0],region[1][1], dimensions[1],endpoint=False,retstep=True)
        self.latitudes = latitudes + latstep/2
        longitudes, longstep = np.linspace(region[2][0],region[2][1], dimensions[2],endpoint=False,retstep=True)
        self.longitudes = longitudes + longstep/2
##        for i in range(dimensions[0]): #depth
##            for j in range(dimensions[1]): #latitude
##                for k in range(dimensions[2]): #longitude
##                    grid[i][j][k] = (depths[i],latitudes[j],longitudes[k])
##        self.grid = grid

    def labelProb(self,hypocenter,dimensions):
        #region [[latitudemin,latitudemax],[longitudemin,longitudemax],[depthmin,depthmax]]
        #dimensions: dimension of the discreteblock in which we calculate probability
        pArray = np.empty([dimensions[0],dimensions[1],dimensions[2]])
        for i in range(dimensions[0]): #depth
            for j in range(dimensions[1]): #latitude
                for k in range(dimensions[2]): #longitude
                    pArray[i][j][k] = self.pMag(hypocenter,(self.depths[i],self.latitudes[j],self.longitudes[k]))
        return pArray
        
    def pMag(self,hypocenter, point):
        d = self.sqdistance(hypocenter, point)
        return np.exp(-d/self.r)
        

    def sqdistance(self,point1, point2):
        #point[0]:depth, point[1]=phi  point[2]=theta
        R = 6378.137
        x1,y1,z1 = (R-point1[0])*np.cos(point1[1])*np.cos(point1[2]),(R-point1[0])*np.cos(point1[1])*np.sin(point1[2]),(R-point1[0]*np.sin(point1[1]))
        x2,y2,z2 = (R-point2[0])*np.cos(point2[1])*np.cos(point2[2]),(R-point2[0])*np.cos(point2[1])*np.sin(point2[2]),(R-point2[0]*np.sin(point2[1]))
        print(x1,x2,y1,y2,z1,z2)
        xd = x2-x1
        yd = y2-y1
        zd = z2-z1
        return xd*xd + yd*yd + zd*zd


    #def makeLabels():
        ##read /Users/himanshusharma/karnuz/Rose/files
        ## load label data
        ## make a 1d array of probGrid

##    class Lambda(nn.Module):
##        def __init__(self, func):
##            super().__init__()
##            self.func = func
##
##        def forward(self, x):
##            return self.func(x)


    def preprocess(self,x):
        return x.view(3, 2772)


    def getEqModel(self):

        model = nn.Sequential(
            Lambda(preprocess),
            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,4)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,4)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,4)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3,4)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample((4,2)),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample((2,2)),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample((2,2)),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 30, kernel_size=3, padding=1),
            Lambda(lambda x: x.view(-1,128,80))
        )
        return model
        

    def entropy_loss(self,input,target):
        return torch.nn.functional.binary_cross_entropy(input, target)

    def loss_batch(self,model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        return loss.item(), len(xb)


    def fit(self,epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            print(epoch, val_loss)


    def setitup(self):
        self.makeGrid(self.region,self.dimensions)
        labelpd = []
        for i in range(len(self.yb)):
            pd = self.labelProb(self.yb[i],self.dimensions)
            labelpd.append(pd)
        self.ypd = labelpd

        #dt = DataLoad()
        #dfx, dfy = dt.loadData()

        #region = [(dt.latitudeMin,dt.latitudeMax),(dt.longitudeMin,dt.longitudeMax),(dt.depthMin,dt.depthMax)]
        #self.makegrid(region,self.dimensions)

        #yb = []
        #for i in range(len(dfy)):
         #   pd = self.labelProb(df.iloc[i].values,self.dimensions)
          #  yb.append(pd)
        #self.yb = yb
        #self.xb = dfx.as_matrix()

    def getReadyData(self):
        dt = DataLoad()
        self.xb, self.yb = dt.loadData()
        self.region = dt.getDimensions()
        

    if __name__ == "main":

        epochs = 10
        lr = 0.0001
        

        model = getEqModel()
        opt = optim.SGD(model.parameters(),lr=lr)

        train_ds = TensorDataset(self.xb,self.ypd)
        valid_ds = TensorDataset(self.xb,self.ypd)

        train_dl = DataLoader(train_ds, batch_size=8)
        valid_dl = DataLoader(valid_ds, batch_size=16)
        
#        train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

        loss_func = entropy_loss

        fit(epochs, model, loss_func, opt, train_dl, valid_dl)




























        

