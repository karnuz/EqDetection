import torch
import torch.nn as nn
import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)

        self.maxPool1 = nn.MaxPool2d((1,4))

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)

        self.maxPool2 = nn.MaxPool2d((1,4))

        self.conv5 = nn.Conv2d(128,256,3,padding=1)
        self.conv6 = nn.Conv2d(256,256,3,padding=1)

        self.maxPool3 = nn.MaxPool2d((2,4))

        self.conv7 = nn.Conv2d(256,512,3,padding=1)
        self.conv8 = nn.Conv2d(512,512,3,padding=1)
        
        self.maxPool4 = nn.MaxPool2d((3,4))

        self.conv9 = nn.Conv2d(512,1024,3,padding=1)
        self.conv10 = nn.Conv2d(1024,1024,3,padding=1)

        self.upsample1 = nn.Upsample((4,2))
        
        self.conv11 = nn.Conv2d(1024,512,3,padding=1)
        self.conv12 = nn.Conv2d(512,512,3,padding=1)
        self.conv13 = nnConv2d(512,512,3,padding=1)

        self.upsampe2 = nn.Upsample((2,2))
        
        self.conv14 = nn.Conv2d(512,256,3,padding=1)
        self.conv15 = nn.Conv2d(256,256,3,padding=1)
        self.conv16 = nn.Conv2d(256,256,3,padding=1)

        self.upsample3 = nn.Upsample((2,2))
        
        self.conv17 = nn.Conv2d(256,128,3,padding=1)
        self.conv18 = nn.Conv2d(128,64,3,padding=1)
        self.conv19 = nn.Conv2d(64,64,3,padding=1)
        self.conv20 = nn.Conv2d(64,32,3,padding=1)
        self.conv21 = nn.Conv2d(32,30,3,padding=1)

def labelProb(hypocenter, region, dimensions):
    pArray = np.array(dimensions)
    grid = np.array(dimensions)
    latitudes, latstep = np.linspace(region[0][0],region[0][1], dimension[0],endpoint=False,retstep=True) 
    longitudes, longstep = np.linspace(region[1][0],region[1][1], dimension[1],endpoint=False,retstep=True)
    depths, depthstep = np.linspace(region[2][0],region[2][1], dimension[2],endpoint=False,retstep=True)
    for i in range(dimensions[2]):
        for j in range(dimensions[0]):
            for k in range(dimensions[1]):
                grid[i][j][k] = (depths[i]+depthstep,latitudes[j]+latstep,longitudes[k]+longstep)
                pArray[i][j][k] = self.pMag(hypocenter,grid[i][j][k])
    
def pMag(hypocenter, point):
    d = self.sqdistance(hypocenter, point)
    return np.exp(-d/r)
    

def sqdistance(point1, point2):
    x1,y1,z1 = (R-point1[0])*cos(point1[1])*cos(point1[2]),(R-point1[0])*cos(point1[1])*sin(point1[2]),(R-point1[0]*sin(point1[1]))
    x2,y2,z2 = (R-point2[0])*cos(point2[1])*cos(point2[2]),(R-point2[0])*cos(point2[1])*sin(point2[2]),(R-point2[0]*sin(point2[1]))
    xd = x2-x1
    yd = y2-y1
    zd = z2-z1
    return xd*xd + yd*yd + zd*zd



def makeLabels():
    ##read /Users/himanshusharma/karnuz/Rose/files
    ## load label data
    ## make a 1d array of probGrid


in __name__ == "main":


    
    target = labelProb()

    model = nn.Sequential(
        Lambda(preprocess),
        nn.Conv2d(3, 64, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d((1,4))
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

    opt = optim.SGD(model.parameters(),lr=lr)
    opt(,loss_func(),)
    

def loss_func = torch.nn.functional.binary_cross_entropy(input, target) 

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(1, 2772)


if __name__ == "main":

    x_train = np.array()
    eventFile = open("../cahuilla_events.txt","r")
    for line in csv.reader(eventFile):
        x = line[0].split(" ")
        path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
        st = read(path)
        x_i = []
        for i in range(len(st)):
            x_i.append(st[i])
        x_train.append(x_i)



