from TigerData import *
import mxnet 
from mxnet import image,np,npx
npx.set_np()
from mxnet.gluon.data.vision.transforms import Resize,ToTensor
from d2l import mxnet as d2l
from PIL import Image, ImageDraw

def predict(image,model,ctx,k):
    w,h=256,256
    im1=np.array(image)
    im2=image
    im1=Resize(256)(im1)
    im1=ToTensor()(im1)
    im2=Image.fromarray(im2)
    draw=ImageDraw.Draw(im2)
    x=np.expand_dims(im1,axis=0)
    anchors,class_preds,bbox_preds=model(x.as_in_ctx(ctx))
    cls_probs=npx.softmax(class_preds).transpose(0,2,1)
    output = npx.multibox_detection(cls_probs, bbox_preds, anchors,nms_threshold=0.5)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    output=output[0,idx]
    output=output[:k]
    for row in output:
        bbox = row[2:6] * np.array((w, h, w, h), ctx=row.ctx)
        draw.rectangle((bbox[0],bbox[1],bbox[2],bbox[3]), fill=None, outline=(255, 255, 255))
    return im2
