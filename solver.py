import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import *
import logging
import wandb

class Solver(object):  # object是所有类的父类
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params  # args
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.is_train = is_train
        self.model = model = MPA(hp)
        model.print_trainable_parameters()
        # Training hyperarams

        self.update_batch = hp.update_batch  # 默认是1

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        else:
            self.device = torch.device("cpu")

        self.criterion = nn.L1Loss(reduction="mean")  # loss 函数
        # self.criterion = nn.HuberLoss(reduction='mean')
        
        # optimizer
        self.optimizer={}

        if self.is_train:
            # mmilb_param = []
            main_param = []
            bert_param = []
            lora_param = []
            adapter_param = []

            for name, p in model.named_parameters():  # 把值p加入到bert_param, mmilb_param, main_param其中之一
                if p.requires_grad:
                    if 'lora_' in name: 
                        lora_param.append(p)
                    elif 'adapter' in name:
                        adapter_param.append(p)
                    else: 
                        main_param.append(p)
                
            for p in main_param:
                if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p)  # xavier的初始化方式

        
        optimizer_main_group = [
            {'params': lora_param, 'weight_decay': hp.weight_decay_lora, 'lr': hp.lr_lora},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main},
            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        # self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5, verbose=True)
        # self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)
        # 用来更新学习率。min表示当指标不再降低(如监测loss) 
        # 当监测指标达到要求时，lr=lr×factor; 是否打印学习率信息，print( 'Epoch {:5d} reducing learning rate of group {} to {:.4e}.'.format(epoch, i, new_lr)
        # patience 忍受该指标多少个epoch不变化
        # 默认的情况是20次不降低的时候学习率变为一半。

        # 学习率衰减.mosi为8
        self.scheduler_main = StepLR(self.optimizer_main, step_size=hp.when,gamma=0.1)
        # self.scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_main,T_max=30, eta_min=0, last_epoch=-1)
        # self.scheduler_main = CosineAnnealingLR(self.optimizer_main,T_max=10)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0

            model.train()
            proc_loss, proc_size = 0, 0
            main_loss = 0.0
            multi_con_loss = 0.0

            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(self.train_loader):  # 取数据
                # text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                vision = batch_data['vision']
                audio = batch_data['audio']
                text = batch_data['text']
                y = batch_data['labels']['M']
                # print(text.shape)
                # print(vision.shape)
                
                model.zero_grad()  # 我们进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了. 置零. 

                with torch.cuda.device(0):
                    vision, audio, text, y = vision.cuda(), audio.cuda(), text.cuda(), y.cuda()
                
                batch_size = y.size(0)               
                preds = model(vision, audio, text)

                y_loss = criterion(preds, y)
               
                loss = y_loss
                # loss = y_loss
                loss.backward()
                
                # -------------------------------------------------------- #
                left_batch -= 1  # left_batch 是做什么的
                if left_batch == 0:  # self.update_batch = hp.update_batch  # 默认是1
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)  # 进行梯度裁剪. 默认1.0
                    optimizer.step()
                # -------------------------------------------------------- #

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                main_loss +=y_loss.item() * batch_size
                
                # con_loss=0
                    
            return epoch_loss
            # return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            main_loss = 0.0        
            results = []
            truths = []

            with torch.no_grad():
                for batch_data in loader:
                    vision = batch_data['vision']
                    audio = batch_data['audio']
                    text = batch_data['text']
                    y = batch_data['labels']['M']

                    with torch.cuda.device(0):
                        vision, audio, text, y = vision.cuda(), audio.cuda(), text.cuda(), y.cuda()
                

                    batch_size = y.size(0)       # bert_sent in size (bs, seq_len, emb_size)
                    preds = model(vision, audio, text)                 
                    criterion = nn.L1Loss()

                    main_loss += criterion(preds, y).item() * batch_size   
                    results.append(preds)
                    truths.append(y)
            
            # avg_main_loss = main_loss / (self.hp.n_test if test else self.hp.n_valid)
            

            results = torch.cat(results)
            truths = torch.cat(truths)
            test_preds = results.view(-1).cpu().detach().numpy()
            test_truth = truths.view(-1).cpu().detach().numpy()
            avg_main_loss =  np.mean(np.absolute(test_preds - test_truth))
            return avg_main_loss, results, truths

        # 两个函数定义到此结束

        best_valid = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            logging.info(f'epoch {epoch}:')

            self.epoch = epoch

            # maximize likelihood
            # if self.hp.contrast:
            #     train_loss = train(model, optimizer_mmilb, criterion, 0)

            # minimize all losses left
            train_main_loss= train(model, optimizer_main, criterion)  # 训练一个epoch

            val_loss, results_val, truths_val = evaluate(model, criterion, test=False)  # 验证集损失
            test_loss, results, truths = evaluate(model, criterion, test=True)  # 测试集损失
            
            # --------------计算二分类准确率------------------
            test_preds = results.view(-1).cpu().detach().numpy()
            test_truth = truths.view(-1).cpu().detach().numpy()
            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  # 非0值的test_truth索引
            binary_truth_non0 = test_truth[non_zeros] > 0
            binary_preds_non0 = test_preds[non_zeros] > 0
            acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

            val_preds = results_val.view(-1).cpu().detach().numpy()
            val_truth = truths_val.view(-1).cpu().detach().numpy()
            non_zeros_val = np.array([i for i, e in enumerate(val_truth) if e != 0])  # 非0值的test_truth索引
            binary_truth_non0_val = val_truth[non_zeros_val] > 0
            binary_preds_non0_val = val_preds[non_zeros_val] > 0
            acc_2_non0_val = accuracy_score(binary_truth_non0_val, binary_preds_non0_val)
            
            end = time.time()
            duration = end-start
            # scheduler_main.step(val_loss)    # Decay learning rate by validation loss
            scheduler_main.step()
            learning_rate = optimizer_main.state_dict()['param_groups'][0]['lr']

            # weight and bias
            # wandb.log({"Train/Loss": train_main_loss, 
            #            'Train/multi_contrast Loss': train_mc_loss,
            #            'Train/ta_con_loss': ta_con_loss,
            #            'Train/av_con_loss': av_con_loss,
            #            'Train/vt_con_loss': vt_con_loss,
            #            'Train/joint_Loss': joint_loss_train,

            #            "Val/main Loss": val_loss,
            #            'Val/multi_contrast Loss': val_mc_loss,
            #            'Val/av_Loss': vtl,
            #            'Val/tv_Loss': val,
            #            'Val/ta_Loss': vvl,
            #            'Val/joint_Loss': joint_loss_val,

            #            'Test/Loss': test_loss,
            #            'Test/BA': acc_2_non0,
            #            'Val/BA':acc_2_non0_val,
            #            'learning_rate':learning_rate
            #            })


            # validation F1
            logging.info("-"*50)
            logging.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            logging.info("-"*50)
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            # --------------日志结束------------------
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # if test_loss < best_mae:
                best_epoch = epoch
                best_mae = test_loss
                if self.hp.dataset in ["mosei_senti", "mosei"]:
                    eval_mosei_senti(results, truths, True)
                elif self.hp.dataset == 'mosi':
                    eval_mosi(results, truths, True)
                best_results = results
                best_truths = truths
                # save_model(model)

            else:
                patience -= 1
                if patience == 0:
                    break

        wandb.finish()

        # print(f'Best epoch: {best_epoch}')
        logging.info(f'Best epoch: {best_epoch}')

        if self.hp.dataset in ["mosei_senti", "mosei"]:
            best_dict = eval_mosei_senti(best_results, best_truths, True)  # 保存最好的模型
        elif self.hp.dataset == 'mosi':
            best_dict = eval_mosi(best_results, best_truths, True)
        sys.stdout.flush()  # sys.stdout.flush()的作用就是显示地让缓冲区的内容输出。防止只能一次性的内容输出
        return best_dict