from mxnet.gluon import nn,loss
from mxnet import nd,npx,np
import mxnet
npx.set_np()
classs_loss=loss.SoftmaxCrossEntropyLoss()
bbox_loss=loss.L2Loss()

class LossBox:
      def __init__(self,shapes):
          self.shapes=shapes
          self.weight=(np.ones(shape=self.shapes+(1,))*6).as_in_ctx(mxnet.gpu(0))
    
      
      def calculate_loss(self,class_preds,class_labels,bbox_preds,bbox_labels,bbox_masks,train,weight):
          if train==1:
             weights=self.weight*(class_labels==1).reshape(self.shapes+(1,))
             weights=weights+3
          else:
             weights=(self.weight)/6
          loss_class=classs_loss(class_preds,class_labels,weights)
          weight=weight.as_in_ctx(mxnet.gpu(0))
          loss_bbox=bbox_loss(bbox_preds*bbox_masks,bbox_labels*bbox_masks,weight)
          return loss_bbox+loss_class

      def evaluateclass(self,class_preds,class_labels):
    	    predictions=class_preds.argmax(axis=-1)
    	    return ((predictions==class_labels).mean()).item()

      def evaluatebbox(self,bbox_preds,bbox_labels,bbox_masks):
          return ((np.abs((bbox_labels-bbox_preds)*bbox_masks)).mean()).item()
