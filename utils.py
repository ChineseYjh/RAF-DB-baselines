import os
import sys
import pickle
import logging
import numpy as np
import torchvision
import json
import itertools
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset,DataLoader

"""
===========================================CONSTANT=====================================================
"""

MEAN=[0.575,0.450,0.401]
STD=[0.152,0.136,0.144]
NUM2EMO={
            1: "Surprise",
            2: "Fear",
            3: "Disgust",
            4: "Happiness",
            5: "Sadness",
            6: "Anger",
            7: "Neutral",
        }
CLASSNAMES=['sur','fea','dis','hap','sad','ang','neu']

"""
=============================================CLASS======================================================
"""

class RafDB(Dataset):
    def __init__(self,mode="train"):
        self.TRAIN_NUM=12271
        self.VAL_NUM=3068
        self.TEST_NUM=3068
        self.NUM2EMO=NUM2EMO
        self.CLASSNAMES=CLASSNAMES
        self.file_path=f'./npyv2/file_{mode}.npy'
        self.label_path=f'./npyv2/label_{mode}.npy'
        self.mode=mode
        self.preproc=transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=MEAN,std=STD)])
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
    
    
"""
=============================================FUNCTION======================================================
"""

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
    if config.use_gpu==True and torch.cuda.is_available():
        pass
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    torch.backends.cudnn.benchmark = True
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    json.dump(config,open(os.path.join(config.output_dir,'config.conf'),'w'))
    return prepare_logger(config)

def config_optimizer(params,config):
    if config.optim=='Adam':
        optimizer=torch.optim.Adam(params,lr=config.lr,weight_decay=config.weight_decay)
    elif config.optim=='SGD':
        optimizer=torch.optim.SGD(params,lr=config.lr,momentum=config.momentum,weight_decay=config.weight_decay)
    else:
        raise
    return optimizer

def config_scheduler(optimizer,config,mode='iter'):
    if config[mode+'_scheduler']=="MultiStepLR":
        scheduler=optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(st) for st in config[mode+'_milestones'].split(',')],
                                                 gamma=config[mode+'_gamma'])
    elif config[mode+'_scheduler']=="ExponentialLR":
        scheduler=optim.lr_scheduler.ExponentialLR(optimizer,
                                                   gamma=config[mode+'_gamma'])
    elif config[mode+'_scheduler']=="StepLR":
        scheduler=optim.lr_scheduler.StepLR(optimizer,
                                            step_size=config[mode+'_step_size'],
                                            gamma=config[mode+'_gamma'])
    else:
        raise
    return scheduler

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

def add_embedding(writer,mat,metadata,label_img,global_step,tag):
    REVERSED_STD=[1/std for std in STD]
    REVERSED_MEAN=[-mean/std for std,mean in list(zip(MEAN,STD))]
    reverse_preproc=transforms.Normalize(mean=REVERSED_MEAN,std=REVERSED_STD)
    new_label_img=torch.cat([reverse_preproc(img).unsqueeze(0) for img in label_img])
    metadata=[CLASSNAMES[meta.item()] for meta in metadata]
    writer.add_embedding(mat=mat,metadata=metadata,label_img=new_label_img,global_step=global_step,tag=tag)
    
def check_save(tmdv,mmdv):
    return tmdv>mmdv