import os
import sys
import argparse
import logging
import time
import json
import torch
import numpy as np

BASE_DIR=os.path.dirname(os.path.abspath(__file__))+'/../'

import networks
import losses
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import *

def parse_config():
    """
    one config may change a lot to produce different output dir, and there is the specific *.conf in each output dir
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--config_name",type=str,default="baseDCNN")
    args=parser.parse_args()
    config=Pack(json.load(open(f"{BASE_DIR}/code/configs/{args.config_name}.conf")))
    return config

def train(config,logger):
    
    #init logger and seed
    start_time = time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))
    logger.info('[START TRAINING]\n{}\n{}'.format(start_time, '=' * 90))
    logger.info(config)
    logger.info('load data...')
    
    #load data
    train_dataset=RafDB(mode="train")
    val_dataset=RafDB(mode="val")
    train_loader=RafDBLoader(dataset=train_dataset,batch_size=config.bsz,shuffle=True)
    val_loader=RafDBLoader(dataset=val_dataset,batch_size=config.bsz,shuffle=True)
    logger.info('create net and loss instance...')
    
    #define net,loss
    net_class=getattr(networks,config.net)
    net=net_class()
    net_criterion=getattr(losses,config.loss)
    criterion=net_criterion()
    logger.info('check and set GPU...')
    
    #check gpu
    if config.use_gpu==True and torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
        device=torch.device('cuda')
        net.to(device)
        if torch.cuda.device_count()>1:
            device_ids=[int(gpu) for gpu in config.gpu_ids.split(',')]
            torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://localhost:{config.localhost}', rank=0, world_size=1)
            net=torch.nn.parallel.DistributedDataParallel(net, device_ids=device_ids, find_unused_parameters=True)
    logger.info('create optimizer...')
    
    #define optimizer
    optimizer_class=getattr(torch.optim,config.optim)
    optimizer=optimizer_class(net.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    
    #load checkpoint if needed
    start_n_iter=0
    start_epoch=0
    if config.resume==True:
        logger.info(f'load checkpoint from {config.ckpt_path}...')
        ckpt=load_checkpoint(config.ckpt_path)
        start_n_iter=ckpt['n_iter']
        start_epoch=ckpt['epoch']
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optim'])
        
    #tensorboardX
    writer_dir=os.path.join(config.output_dir,'boardX')
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    writer=SummaryWriter(writer_dir)
    
    #ckpt dir
    ckpt_dir=os.path.join(config.output_dir,'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    #start
    logger.info(f'start with epoch range of [{start_epoch}, {config.epoch})...')
    n_iter=start_n_iter
    if n_iter==0:
        optimizer.zero_grad()
    for epoch in range(start_epoch,config.epoch):
        if config.forward_only==False:
            net.train()
            pbar=tqdm(enumerate(train_loader),total=len(train_loader))
            start_time=time.time()
            tot_loss=0
            for i,data in pbar:
                #prepare
                x_data,y_data=data
                if len(x_data.size())!=4:
                    logger.error(f"x_data size wrong! Now the size is {x_data.size()}")
                    raise
                if len(y_data.size())>1:
                    y_data=y_data.squeeze(-1)
                if config.use_gpu==True:
                    x_data=x_data.to(device)
                    y_data=y_data.to(device)
                prepare_time=time.time()-start_time
                #forward and backward
                pred_data=net(x_data)
                loss=criterion(pred_data,y_data)
                loss=torch.mean(loss,dim=0)
                loss.backward()
                if n_iter%config.gd_acc==config.gd_acc-1:
                    optimizer.step()
                    optimizer.zero_grad()
                #log
                tot_loss+=loss.item()
                writer.add_scalars('loss',{'Train':loss.item()},n_iter)
                process_time=time.time()-start_time-prepare_time
                pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                    process_time/(process_time+prepare_time), epoch, config.epoch))
                n_iter+=1
            logger.info(f"[Epoch: {epoch}]TrainLoss:{tot_loss/len(train_loader)}")
            
            #val and save
            if epoch%config.save_per_epoch==config.save_per_epoch-1:
                logger.info(f"suspend to save ckpt and  validate on val set when epoch[{epoch}] is complete...")
                save_checkpoint(os.path.join(ckpt_dir,f'ckpt-{n_iter}.pickle'),net,optimizer,epoch,n_iter)
                result=np.zeros((7,7))
                net.eval()
                with torch.no_grad():
                    pbar=tqdm(enumerate(val_loader),total=len(val_loader))
                    start_time=time.time()
                    tot_loss=0
                    TOT,TP=0,0
                    for i,data in pbar:
                        #prepare
                        x_data,y_data=data
                        if len(x_data.size())!=4:
                            logger.error(f"x_data size wrong! Now the size is {x_data.size()}")
                            raise
                        if len(y_data.size())>1:
                            y_data=y_data.squeeze(-1)
                        if config.use_gpu==True:
                            x_data=x_data.to(device)
                            y_data=y_data.to(device)
                        prepare_time=time.time()-start_time
                        #forward and predict
                        pred_data=net(x_data)
                        loss=criterion(pred_data,y_data)
                        loss=torch.mean(loss,dim=0)
                        if isinstance(pred_data,list):
                            pred_data=pred_data[0]
                        pred_data=torch.argmax(pred_data,dim=1)
                        result,tp=log_result(result,pred_data,y_data)
                        #log
                        tot_loss+=loss.item()
                        TP+=tp
                        TOT+=y_data.size(0)
                        process_time=time.time()-start_time-prepare_time
                        pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                            process_time/(process_time+prepare_time), epoch, config.epoch))
                    writer.add_scalars('loss',{'Val':tot_loss/len(val_loader)},n_iter)
                    normalized_result = result.astype('float') / (0.0001+result.sum(axis=1)[:, np.newaxis])
                    writer.add_scalar('mean_diagonal_value',sum([normalized_result[i][i] for i in range(7)])/7,n_iter)
                    writer.add_figure('confusion_matrix_on_val_set',figure=plot_confusion_matrix(result, classes=val_dataset.CLASSNAMES, normalize=True,title='confusion matrix on val set'),global_step=n_iter)
                    writer.add_scalar('accuracy',TP/TOT,n_iter)
                    logger.info(f"[Epoch: {epoch}]ValLoss:{tot_loss/len(val_loader)}")
        else:
            logger.info(f"forward only! So validate on train set when epoch[{epoch}] is complete...")
            result=np.zeros((7,7))
            net.eval()
            with torch.no_grad():
                pbar=tqdm(enumerate(train_loader),total=len(train_loader))
                start_time=time.time()
                tot_loss=0
                TOT,TP=0,0
                for i,data in pbar:
                    #prepare
                    x_data,y_data=data
                    if len(x_data.size())!=4:
                        logger.error(f"x_data size wrong! Now the size is {x_data.size()}")
                        raise
                    if len(y_data.size())>1:
                        y_data=y_data.squeeze(-1)
                    if config.use_gpu==True:
                        x_data=x_data.to(device)
                        y_data=y_data.to(device)
                    prepare_time=time.time()-start_time
                    #forward and predict
                    pred_data=net(x_data)
                    loss=criterion(pred_data,y_data)
                    loss=torch.mean(loss,dim=0)
                    if isinstance(pred_data,list):
                        pred_data=pred_data[0]
                    pred_data=torch.argmax(pred_data,dim=1)
                    result,tp=log_result(result,pred_data,y_data)
                    #log
                    tot_loss+=loss.item()
                    TP+=tp
                    TOT+=y_data.size(0)
                    process_time=time.time()-start_time-prepare_time
                    pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                        process_time/(process_time+prepare_time), epoch, config.epoch))
                    writer.add_scalars('loss',{'Train':loss.item()},i)
                writer.add_figure('confusion_matrix_on_train_set',figure=plot_confusion_matrix(result, classes=train_dataset.CLASSNAMES, normalize=True,title='confusion matrix on train set'),global_step=n_iter)
                normalized_result = result.astype('float') / (0.0001+result.sum(axis=1)[:, np.newaxis])
                writer.add_scalar('mean_diagonal_value',sum([normalized_result[i][i] for i in range(7)])/7,n_iter)
                writer.add_scalar('accuracy',TP/TOT,n_iter)
                logger.info(f"[Epoch: {epoch}]TrainLoss:{tot_loss/len(train_loader)}")
            break
    logger.info("exit 0.")
    
def val(config,logger):
    pass

def test(config,logger):
    """
    Note: when 'config.mode' is 'test', you don't neet to set 'config.forward_only' and 'config.resume' to true
    
    test model on test set
    """
    #init logger and seed
    start_time = time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))
    logger.info('[START TESTING]\n{}\n{}'.format(start_time, '=' * 90))
    logger.info(config)
    logger.info('load data...')
    
    #load data
    test_dataset=RafDB(mode="test")
    test_loader=RafDBLoader(dataset=test_dataset,batch_size=config.bsz,shuffle=True)
    logger.info('create net and loss instance...')
    
    #define net,loss
    net_class=getattr(networks,config.net)
    net=net_class()
    net_criterion=getattr(losses,config.loss)
    criterion=net_criterion()
    logger.info('check and set GPU...')
    
    #check gpu
    if config.use_gpu==True and torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
        device=torch.device('cuda')
        net.to(device)
        if torch.cuda.device_count()>1:
            device_ids=[int(gpu) for gpu in config.gpu_ids.split(',')]
            torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://localhost:{config.localhost}', rank=0, world_size=1)
            net=torch.nn.parallel.DistributedDataParallel(net, device_ids=device_ids, find_unused_parameters=True)
    
    #load checkpoint if needed
    start_n_iter=0
    start_epoch=0
    if len(config.ckpt_path)>0:
        logger.info(f'load checkpoint from {config.ckpt_path}...')
        ckpt=load_checkpoint(config.ckpt_path)
        start_n_iter=ckpt['n_iter']
        start_epoch=ckpt['epoch']
        net.load_state_dict(ckpt['net'])
        logger.info(f'checkpoint is on epoch[{start_epoch}], iter[{start_n_iter}]')
    
    #tensorboardX
    writer_dir=os.path.join(config.output_dir,'boardX')
    if os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    writer=SummaryWriter(writer_dir)
    logger.info(f"test on test set...")
    result=np.zeros((7,7))
    net.eval()
    with torch.no_grad():
        pbar=tqdm(enumerate(test_loader),total=len(test_loader))
        start_time=time.time()
        tot_loss=0
        TOT,TP=0,0
        for i,data in pbar:
            #prepare
            x_data,y_data=data
            if len(x_data.size())!=4:
                logger.error(f"x_data size wrong! Now the size is {x_data.size()}")
                raise
            if len(y_data.size())>1:
                y_data=y_data.squeeze(-1)
            if config.use_gpu==True:
                x_data=x_data.to(device)
                y_data=y_data.to(device)
            prepare_time=time.time()-start_time
            #forward and predict
            pred_data=net(x_data)
            loss=criterion(pred_data,y_data)
            loss=torch.mean(loss,dim=0)
            if isinstance(pred_data,list):
                pred_data=pred_data[0]
            pred_data=torch.argmax(pred_data,dim=1)
            result,tp=log_result(result,pred_data,y_data)
            #log
            tot_loss+=loss.item()
            TOT+=y_data.size(0)
            TP+=tp
            process_time=time.time()-start_time-prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, iter: {}/{}:".format(
                process_time/(process_time+prepare_time), i, len(test_loader)))
            writer.add_scalars('loss',{'Test':loss.item()},i)
        
        #write and log
        writer.add_figure('confusion_matrix_on_test_set',figure=plot_confusion_matrix(result, classes=test_dataset.CLASSNAMES, normalize=True,title='confusion matrix on test set'),global_step=start_n_iter)
        normalized_result = result.astype('float') / (0.0001+result.sum(axis=1)[:, np.newaxis])
        mdv=sum([normalized_result[i][i] for i in range(7)])/7
        writer.add_scalar('mean_diagonal_value',mdv,start_n_iter)
        writer.add_scalar('accuracy',TP/TOT,start_n_iter)
        logger.info(f"loss of checkpoint in {config.ckpt_path}: {tot_loss/len(test_loader)}")
        logger.info(f"mean diagonal value of checkpoint in {config.ckpt_path}: {mdv}")
        logger.info(f"accuracy of checkpoint in {config.ckpt_path}: {TP/TOT}")
        logger.info(f"confusion matrix on test set: {normalized_result}")
        logger.info("exit 0.")
    