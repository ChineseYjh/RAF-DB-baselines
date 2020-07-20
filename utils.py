import os
import sys
import pickle
import logging
import numpy as np
import torchvision
import json
import itertools
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset,DataLoader

class RafDB(Dataset):
    def __init__(self,mode="train"):
        self.TRAIN_NUM=12271-2454 # ==9817
        self.VAL_NUM=2454 # 12271//5 == 2454
        self.TEST_NUM=3068
        self.NUM2EMO={
            1: "Surprise",
            2: "Fear",
            3: "Disgust",
            4: "Happiness",
            5: "Sadness",
            6: "Anger",
            7: "Neutral",
        }
        self.CLASSNAMES=['sur','fea','dis','hap','sad','ang','neu']
        self.file_path=f'./npy/file_{mode}.npy'
        self.label_path=f'./npy/label_{mode}.npy'
        self.mode=mode
        self.preproc=transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.575,0.450,0.402],std=[0.151,0.136,0.144])])
    def __getitem__(self,index):
        x_data=self.preproc(Image.open(np.load(self.file_path)[index]))
        y_data=torch.tensor([np.load(self.label_path)[index]-1])
        return x_data,y_data
    def __len__(self):
        if self.mode=='train':
            return self.TRAIN_NUM
        if self.mode=='val':
            return self.VAL_NUM
        if self.mode=='test':
            return self.TEST_NUM
        
class RafDBLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
class Pack(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return False
    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

def prepare_logger(config):
    logger_level = logging.DEBUG if config.debug else logging.INFO
    logger=logging.getLogger()
    logger.setLevel(logger_level)
    
    formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console=logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logger_level)
    logger.addHandler(console)
    
    file_handler=logging.FileHandler(os.path.join(config.output_dir,f"sesson-{config.mode}{'-fwd'if (config.mode=='train' and config.forward_only==True) else ''}.log"))
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
    
    
def init(config):
    torch.backends.cudnn.benchmark = True
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    json.dump(config,open(os.path.join(config.output_dir,'config.conf'),'w'))
    return prepare_logger(config)


def save_checkpoint(ckpt_path:str,model,opti,epoch:int,n_iter:int):
    ckpt={'net':model.state_dict(),'optim':opti.state_dict(),'epoch':epoch,'n_iter':n_iter}
    pickle.dump(ckpt,open(ckpt_path,'wb'))
    
def load_checkpoint(ckpt_path):
    return pickle.load(open(ckpt_path,'rb'))
    
def log_result(result,pred_data,y_data):
    """
    result: np.array sized 7*7
    pred_data: torch.tensor sized [BSZ], int ranged [0,6]
    y_data: torch.tensor sized [BSZ], int ranged [0,6]
    """
    TP=0
    if pred_data.size(0)!=y_data.size(0):
        return
    for k in range(y_data.size(0)):
        result[y_data[k].item()][pred_data[k].item()]+=1
        TP+=(1 if y_data[k].item()==pred_data[k].item() else 0)
    return result,TP

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    @params:
    cm: numpy.ndarray([[],[],..])
    classes: list(str,str,...)
    normalize: bool, if we normalize the matrix
    """
    epsilon=0.0001
    if normalize:
        cm = cm.astype('float') / (epsilon+cm.sum(axis=1)[:, np.newaxis])
    plt.rcParams["figure.dpi"]=180
    plt.rcParams["figure.figsize"]=(5,3.5)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,)
    plt.yticks(tick_marks, classes,)
    plt.ylim(0-0.5,len(classes)-0.5)
    plt.xlim(0-0.5,len(classes)-0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig