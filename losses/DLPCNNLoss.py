import torch
import torch.nn as nn
import heapq

class DLPCNNLoss(nn.Module):
    def __init__(self):
        super(DLPCNNLoss,self).__init__()
        self.loss=nn.CrossEntropyLoss()
        self.lamda=0.003
        self.k=20
    def topk_dis(self,xi,row,xi_row,ys,topk):
        dis=[]
        for idx,xj in enumerate(xi_row[0]):
            if xi_row[1][idx]!=row and ys[row].item()==ys[xi_row[1][idx]].item():
                dis.append(torch.sum((xj-xi)**2))
        loss_lp_xi=sum(heapq.nsmallest(topk,dis))
        
        del dis
        return loss_lp_xi
        
    def forward(self,x,y):
        """
        x: list of tensors [torch.tensor[bsz,#class],torch.tensor[bsz,2000]]
        y: torch.tensor sized [bsz]
        """
        x_soft,x_feat=x
        del x
        loss_lp=0
        class2xi_row=dict()
        for row,xi in enumerate(x_feat):
            nowy=int(y[row].item())
            if nowy not in class2xi_row.keys():
                class2xi_row[nowy]=[[xi],[row]]
            else:
                class2xi_row[nowy][0]+=[xi]
                class2xi_row[nowy][1]+=[row]
        for row,xi in enumerate(x_feat):
            nowy=int(y[row].item())
            loss_lp+=self.topk_dis(xi,row,class2xi_row[nowy],y,self.k)
            
        del class2xi_row,x_feat
        return self.lamda*loss_lp/2+self.loss(x_soft,y)