from networks.layers import *
from networks.baseDCNN import *

class DLPCNN(baseDCNN):
    """
    return list[unnormalized tensor [bsz,7], deep feature tensor [bsz,2000]]
    """
    def __init__(self):
        super(DLPCNN,self).__init__()
        