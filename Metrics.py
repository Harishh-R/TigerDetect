import mxnet
from mxnet.gluon import nn
from mxnet import np

class Accumulator:
    
    def __init__(self,n):
        self.l=[0]*n
        self.c=0
    
    def add(self,values):
        self.l=[self.l[i]+val for i,val in enumerate(values)]
        self.c+=1
    
    def get(self):
        return self.l
    
    def __getitem__(self,i):
        return self.l[i]/self.c
        
class F1Score(nn.HybridBlock):
      
    def __init__(self):
        super().__init__()
    
    def hybrid_forward(self,F,x):
        pred=x[0]
        true=x[1]
        true_positive=float((pred*true).sum())
        false_positive=float(((true==0)*pred).sum())
        false_negative=float(((true==1)*(pred==0)).sum())
        if ((true_positive+false_negative)==0 or (true_positive+false_positive)==0):
            return 0
        precision=(true_positive)/(true_positive+false_positive)
        recall=(true_positive)/(true_positive+false_negative)
        if (precision+recall==0):
            return 0
        else:
            f1score=(2*(precision*recall))/(precision+recall)
            return float(format(f1score,'.2g'))
