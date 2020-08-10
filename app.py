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
from mxnet.gluon.data import DataLoader
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn
import numpy
from TigerData import *
from TigerLoss import *
from PIL import Image
import io
from Metrics import Accumulator,F1Score
import streamlit as st
from Tigernet import *
from tigerpredict import *
from TigerSSD import *
import gdown
npx.set_np()
st.set_option('deprecation.showfileUploaderEncoding', False)


st.title("Tiger Detection")

st.text("Downloading model")
k=gdown.download('https://drive.google.com/uc?id=1YZ_2193qZmPtRVpdU5WY7VMp6RU1sgfb')
st.text("Download finished")

if k:
   model=TigerSSD(1)
   model.load_parameters('1ststagetest80.9978991746902466',ctx=mxnet.gpu(0))

model.hybridize()

s=st.file_uploader("Choose an image",type=['png','jpg'])
if s:
   print(type(s))
   im4 = Image.open(io.BytesIO(s.read()))
   im4=numpy.array(im4)
   option=st.selectbox("Choose the number of bounding boxes",('1','2','3','4','5'))
   option=int(option)
   im=predict(im4,model,mxnet.gpu(0),option)
   st.image(im)
