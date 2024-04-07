import torch
import argparse
import numpy as np
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定GPU
# from dataset import *
from dataset import *  # 数据集
from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
# from data_loader import get_loader
from utils.logs import set_arg_log



logging.getLogger ().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, filename='MMEPA.log')

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')  # 设置默认的CPU tensor类型
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果。
    # 设置随机种子后，是每次运行文件的输出结果都一样，但不是每次随机函数生成的结果一样。
    if torch.cuda.is_available():  # GPU是否可用
        torch.cuda.manual_seed_all(seed)  # # 设置CPU生成随机数的种子
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # 设置默认的GPU tensor类型
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False  

        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    
    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    dataLoader = MMDataLoader(args)

    train_loader = dataLoader['train']
    valid_loader = dataLoader['valid']
    test_loader = dataLoader['test']

    torch.autograd.set_detect_anomaly(True)
    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader)

    logging.info(f'Runing code on the {args.dataset} dataset.')
    set_arg_log(args)
    best_dict = solver.train_and_eval()

    logging.info(f'Training complete')
    logging.info('--'*50)
    logging.info('\n')