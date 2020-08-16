import torch
import torch.nn as nn
import heapq

class DLPCNNLoss(nn.Module):
    def __init__(self,weight=None):
        super(DLPCNNLoss,self).__init__()
        self.loss=nn.CrossEntropyLoss(weight=weight)
        self.lamda=0.003
        self.k=20
    def topk_dis(self,xi,row,xi_row,ys,topk):
        class Comp(object):
            def __init__(self,x,dis):
                self.x=x
                self.dis=dis
            def __lt__(self,other):
                return self.dis<other.dis
        cmp_l=[]
        for idx,xj in enumerate(xi_row[0]):
            if xi_row[1][idx]!=row and ys[row].item()==ys[xi_row[1][idx]].item():
                cmp_l.append(Comp(xj,torch.sum((xj-xi)**2)))
        cmp_l=heapq.nsmallest(topk,cmp_l)
        center_k=sum([cmp.x for cmp in cmp_l])/topk
        del cmp_l
        return torch.sum((xi-center_k)**2)
    def forward(self,x,y):
        """
        x: list of tensors [torch.tensor[bsz,#class],torch.tensor[bsz,2000]]
        y: torch.tensor sized [bsz]
        """
        preds,x=x
        loss_lp=0
        class2xi_row=dict()
        for row,xi in enumerate(x):
            nowy=int(y[row].item())
            if nowy not in class2xi_row.keys():
                class2xi_row[nowy]=[[xi],[row]]
            else:
                class2xi_row[nowy][0]+=[xi]
                class2xi_row[nowy][1]+=[row]
        for row,xi in enumerate(x):
            nowy=int(y[row].item())
            loss_lp+=self.topk_dis(xi,row,class2xi_row[nowy],y,self.k)
        loss_lp/=x.size(0)
        del class2xi_row
        return self.lamda*loss_lp/2+self.loss(preds,y)