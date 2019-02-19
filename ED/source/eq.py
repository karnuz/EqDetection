import torch
import torch.nn as nn
import torch.nn.functional as F



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

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(1, 2772)


def getEqModel():

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
    return model
    

def loss_func = torch.nn.functional.binary_cross_entropy(input, target)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


if __name__ == "main":

    x_train = np.array()
    eventFile = open("../cahuilla_events.txt","r")
    buf = csv.reader(eventFile)
    x = buf[0].split(" ")
    path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
    st = read(path)
    headers = list(st[0].stats.keys())
    items = list(st[0].stats.values())
    df =  pd.DataFrame(columns=headers)
    df.append(items)
    
    for line in buf:
        x = line[0].split(" ")
        path = "../eqdata/" + x[0] + "/" + x[0] + x[1] + "/" + x[6]
        st = read(path)
        items = list(st[0].stats.values())
        df.append(items)
        

        x_train = np.zeros((164,2520))
        for i in range(len(st)):
            x_i.append(st[i])
        x_train.append(x_i)

    model = getEqModel()
    opt = optim.SGD(model.parameters(),lr=lr)

