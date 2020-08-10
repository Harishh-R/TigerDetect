import mxnet
from mxnet import nd,ndarray,np,npx
npx.set_np()
from mxnet.gluon import nn
import mxnet as mx
from Tigernet import *
from mxnet import np,npx
npx.set_np()

class TigerSSD(nn.HybridBlock):

    def __init__(self,bs):
        super(TigerSSD,self).__init__()
        self.sizes = [[0.22,0.32,0.46,0.54],[0.63,0.73,0.83,0.92]]
        self.ratios = [[0.5,0.8,2,1.4]] * 2
        self.num_anchors=len(self.sizes[0])+len(self.ratios[0])-1
        self.num_classes=1
        self.bs=bs
        self.class_predict=nn.HybridSequential()
        self.bbox_predict=nn.HybridSequential()
        self.features=get_tigernet()
        for i in range(2):
            self.class_predict.add(self.class_predictor())
            self.bbox_predict.add(self.bbox_predictor())
            
    def class_predictor(self):
        s=nn.HybridSequential()
        s.add(nn.Conv2D(kernel_size=5,padding=2,channels=(self.num_classes+1)*self.num_anchors),nn.BatchNorm(),nn.Activation('softrelu'))
        return s
    
    def bbox_predictor(self):
        b=nn.HybridSequential()
        b.add(nn.Conv2D(self.num_anchors*4,kernel_size=5,padding=2),nn.BatchNorm(),nn.Activation('softrelu'))
        return b

    def hybrid_forward(self, F, x, *args, **kwargs):
        feature=[]
        for block in self.features:
            x=block(x)
            feature.append(x)
        
        cls_preds=[F.npx.batch_flatten(F.np.transpose(cp(feat),(0,2,3,1))) for feat,cp in zip(feature,self.class_predict)]
        bbox_preds=[F.npx.batch_flatten(F.np.transpose(bp(feat),(0,2,3,1))) for feat,bp in zip(feature,self.bbox_predict)]
        anchors=[F.np.reshape(F.npx.multibox_prior(feat,size,ratio),(1,-1)) for feat,size,ratio in zip(feature,self.sizes,self.ratios)]
        cls_preds=F.np.reshape(F.np.concatenate(seq=(cls_preds),axis=1),(self.bs,-1,self.num_classes+1))
        bbox_preds=F.np.reshape(F.np.concatenate(seq=(bbox_preds),axis=1),(self.bs,-1))
        anchors=F.np.concatenate(seq=(anchors),axis=1)
        anchors=F.np.reshape(anchors,(1,-1,4))
        return anchors,cls_preds,bbox_preds
