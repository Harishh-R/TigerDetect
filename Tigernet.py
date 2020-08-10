from mxnet.gluon import nn
from mxnet import np,npx
npx.set_np()

class TigerResnet(nn.HybridBlock):
    
    def __init__(self,n):
        
        super(TigerResnet, self).__init__()
        self.n1=n[0]
        self.n2=n[1]
        self.n3=n[2]
        self.n4=n[3]
        self.conv_1 = nn.Conv2D(self.n1,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2D(self.n2,kernel_size=5,padding=2)
        self.conv_3 = nn.Conv2D(self.n3,kernel_size=7,padding=3)
        self.conv_4 = nn.Conv2D(self.n4,kernel_size=5,padding=2,strides=4)
        self.conv_5 = nn.Conv2D(self.n4,kernel_size=1,strides=4)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        self.bn4 = nn.BatchNorm()
        self.bn4=nn.BatchNorm()
    
    def hybrid_forward(self, F, x, *args, **kwargs):
        
        y = F.npx.relu(self.bn1(self.conv_1(x)))
        y = F.npx.relu(self.bn2(self.conv_2(y)))
        y = F.npx.relu(self.bn3(self.conv_3(y)))
        y=self.bn4(self.conv_4(y))
        x=self.conv_5(x)
        return F.npx.relu(y+x)

def get_tigernet():
    one_net=nn.HybridSequential()
    two_net=nn.HybridSequential()
    
    net=nn.HybridSequential()
    channels=[[120,140,160,180],[200,240,280,360],[400,440,480,500],[540,560,650,712]]
    one_net.add(TigerResnet(channels[0]),TigerResnet(channels[1]),TigerResnet(channels[2]))
    two_net.add(TigerResnet(channels[3]))
    
    net.add(one_net,two_net)
    return net
