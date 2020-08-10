from TigerData import *
import mxnet 
from mxnet import image,np,npx
npx.set_np()
from mxnet.gluon.data.vision.transforms import Resize
from d2l import mxnet as d2l
from PIL import Image, ImageDraw

def predict(im,model,ctx,threshold,k,image_arg=True):
    w,h=256,256
    if image_arg:
        images=image.imread(im)
        im=image.imread(im)
        im=image_transform(im)
        im=im.as_np_ndarray()
    else:
        images=im.transpose(1,2,0)
        im=im.as_np_ndarray()
    images=Resize(256)(images)
    images=Image.fromarray(images.asnumpy())
    draw = ImageDraw.Draw(images)
    x=np.expand_dims(im,axis=0)
    anchors,class_preds,bbox_preds=model(x.as_in_ctx(ctx))
    cls_probs=npx.softmax(class_preds).transpose(0,2,1)
    output = npx.multibox_detection(cls_probs, bbox_preds, anchors,nms_threshold=0.7)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    output=output[0,idx]
    print(output)
    for row in output:
#         score = float(row[1])
#         if score < threshold:
#             continue
        bbox = row[2:6] * np.array((w, h, w, h), ctx=row.ctx)
        draw.rectangle((bbox[0],bbox[1],bbox[2],bbox[3]), fill=None, outline=(255, 255, 255))
    return images
