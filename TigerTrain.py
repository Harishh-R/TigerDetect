from gluoncv.data.base import VisionDataset
import os
import warnings
import glob
import logging
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from mxnet.gluon import Trainer
from mxnet.gluon.data.vision import transforms
from TigerData import *
from TigerLoss import *
from Metrics import Accumulator,F1Score
from Tigernet import *
from TigerSSD import *
from gluoncv.data.transforms.bbox import *
import matplotlib.pyplot as plt
from mxnet import np,nd
from TigerPredict import *
from mxnet.gluon.data import DataLoader
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn
from TigerData import *
from TigerLoss import *
from Metrics import Accumulator,F1Score
from google.colab import files
from Tigernet import *
from TigerSSD import *
npx.set_np()

root="/content/TigerDetect"
batch_size=8
train_data=data(True,root,batch_size)
test_data=data(False,root,batch_size)

device=mxnet.gpu(0)

model=TigerSSD(batch_size)
#model.initialize(init=mxnet.init.Xavier(),ctx=device)
model.load_parameters('model0te',ctx=device)
trainer=Trainer(model.collect_params(),'nag',{'learning_rate':0.01,'wd':0.001})

lr_decay_epoch=[45,69,82,98]

num_epochs=100

print("Size of training data :",len(train_data))
print("Size of test data :",len(test_data))

for batch in train_data:
    x,y=batch[0].as_in_ctx(device),batch[1].as_in_ctx(device)
    break
    
a,c,b=model(x)
losses=LossBox(c.shape[0:2])

animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train_acc','test_acc','train_loss','test_loss'])
animator2 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train_bboxmae','test_bboxmae'])
f1animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['f1_train','f1_test'])
for epoch in range(num_epochs):
    
    train_f1_score,test_f1_score=0,0
  
    if epoch in lr_decay_epoch:
       trainer.set_learning_rate(trainer.optimizer.lr/7)
    
    for data in train_data:
        image=data[0].as_in_ctx(device)
        label=data[1].as_in_ctx(device)
        
        with autograd.record():
            anchors,class_predictions,bbox_predictions=model(image)
            bbox_labels,bbox_masks,class_labels=npx.multibox_target(anchors,label,class_predictions.transpose(0,2,1))
            weight=np.ones(shape=bbox_labels.shape)*4
            weight=weight*bbox_masks
            loss=(losses.calculate_loss(class_predictions,class_labels,bbox_predictions,bbox_labels,bbox_masks,1,weight)).sum()
      
        loss.backward()
        trainer.step(batch_size)

    class_acc_train=losses.evaluateclass(class_predictions,class_labels)

    if class_acc_train>0.85:
       model.save_parameters('1stagetrain'+str(class_acc_train)
    

    bbox_mae_train=losses.evaluatebbox(bbox_predictions,bbox_labels,bbox_masks)
    class_p=(class_predictions.argmax(axis=-1)).reshape(-1)
    class_l=class_labels.reshape(-1)
    train_loss=loss.item()
    train_f1_score=f1((class_p,class_l))
    
     
    for data in test_data:
        image=data[0].as_in_ctx(device)
        label=data[1].as_in_ctx(device)
        anchors,class_predictions,bbox_predictions=model(image)
        bbox_labels,bbox_masks,class_labels=npx.multibox_target(anchors,label,class_predictions.transpose(0,2,1))
        weight=np.ones(shape=bbox_labels.shape)
        weight=weight*bbox_masks
        test_losses=(losses.calculate_loss(class_predictions,class_labels,bbox_predictions,bbox_labels,bbox_masks,0,weight)).sum()
        
    class_acc_test=losses.evaluateclass(class_predictions,class_labels)

    if class_acc_test>0.8:
       model.save_parameters('1stagetest'+str(class_acc_test)
    
    
    bbox_mae_test=losses.evaluatebbox(bbox_predictions,bbox_labels,bbox_masks)
    test_loss=test_losses.item()
    class_p=(class_predictions.argmax(axis=-1)).reshape(-1)
    class_l=class_labels.reshape(-1)
    test_f1_score=f1((class_p,class_l))

    if test_loss-train_loss>6:
       trainer.optimizer.wd=trainer.optimizer.wd*2
    

    animator2.add(epoch+1,(bbox_mae_train,bbox_mae_test))
    animator.add(epoch+1,(class_acc_train,class_acc_test,train_loss,test_loss))
    f1animator.add(epoch+1,(train_f1_score,test_f1_score))
model.save_parameters('model')
files.download('model')

predict('/content/tiger-Siberian.jpg',model,device,0.7,image_arg=True)
