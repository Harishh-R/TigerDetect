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
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.bbox import *
from mxnet import np,nd
from mxnet.gluon.data import DataLoader

mean=np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
std=np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

class TigerDetection(VisionDataset):
    CLASSES = ['tiger']

    def __init__(self, root,train=True,transform=None, index_map=None, preload_label=True):
        super(TigerDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self.train=train
        self._items = self._load_items(root,train)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'Images', '{}.jpg')
        self.index_map = index_map or dict([(classes,i) for i,classes in zip(range(len(self.classes)),self.classes)])
        self._label_cache = self._preload_labels() if preload_label else None
        
    @property
    def classes(self):
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label.copy()

    def _load_items(self, root,train):
        if train:
            name='train'
        else:
            name='val'
        ids = []
        file = os.path.join(root, name + '.txt')
        with open(file, 'r') as f:
            ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
                label.append([cls_id,xmin, ymin, xmax, ymax])
            except AssertionError as e:
                logging.warning("Invalid label at %s, %s", anno_path, e)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]
        
def custom_transformations(*sample):
    img=sample[0]
    bbox=sample[1]
    box=resize(bbox[:,1:],(img.shape[1],img.shape[0]),(256,256))
    bbox[:,1:]=box
    box=flip(bbox[:,1:],flip_x=True,size=(256,256))
    bbox[:,1:]=box
    bbox[:]/=256
    img=transforms.Resize(256)(img)
    img=transforms.RandomFlipLeftRight()(img)
    img=transforms.Cast()(img)
    img=transforms.ToTensor()(img)
    img[:]-=mean
    img[:]/=std
    return (img,bbox)

def image_transform(img):
    img=transforms.Resize(256)(img)
    img=transforms.Cast()(img)
    img=transforms.ToTensor()(img)
    img[:]-=mean
    img[:]/=std
    return img
    
def default_batchify(data):
    image=[]
    label=[]
    for i in range(len(data)):
        image.append(data[i][0].as_np_ndarray())
        label.append(data[i][1])
    for i in range(len(label)):
        m=np.ones(shape=(10,5))*-1
        s=(label[i].shape[0])
        m[:s]=label[i]
        label[i]=m
    image=np.stack(image)
    label=np.stack(label)
    return [image,label]

def data(train,root,bs):
    return DataLoader(TigerDetection(root,train).transform(custom_transformations),batch_size=bs,batchify_fn=default_batchify,last_batch='discard')
    

