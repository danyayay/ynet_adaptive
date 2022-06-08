import re
import argparse
import pathlib
import numpy as np
import pandas as pd
from utils.util_fusion import get_position
import matplotlib.pyplot as plt


def moving_average(x, window, mode='same'):
    data = np.convolve(x, np.ones(window), mode) / window
    n = x.shape[0]
    adjust = window // 2
    for i in range(adjust):
        # first several points
        data[i] = np.mean(data[:(i+adjust+1)])
        # last several points 
        data[n-i-1] = np.mean(x[(n-i-adjust-1):])
    return data 


def extract_training_score(text):
    df = pd.DataFrame()
    for row in re.findall('Epoch ([\d]+): 	Train \(Top-1\) ADE: ([\d\.]+) FDE: ([\d\.]+) 		Val \(Top-k\) ADE: ([\d\.]+) FDE: ([\d\.]+)', text):
        d = {'epoch': row[0], 'train_ade': row[1], 'train_fde': row[2], 'val_ade': row[3], 'val_fde': row[4]}
        df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)
    df.epoch = df.epoch.astype(int)
    df.train_ade = df.train_ade.astype(float)
    df.train_fde = df.train_fde.astype(float)
    df.val_ade = df.val_ade.astype(float)
    df.val_fde = df.val_fde.astype(float)
    return df


def extract_curve(train_msgs, test_msgs=None, window=9, out_path='figures/training_curve/curve.png'):
    if test_msgs is not None:
        extract_test_score(test_msgs)
    train_msgs_list = re.split('save_every_n', train_msgs)[1:]
    df = pd.DataFrame(columns=['seed', 'pretrained_ckpt', 'experiment', 'n_param', 'n_epoch', 'ade', 'fde'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for msg in train_msgs_list: 
        df_curve = extract_training_score(msg)

        experiment = re.search("Experiment (.*?) has started", msg).group(1)
        train_seed = re.search("'seed': ([\d+]),", msg).group(1)
        n_epoch = re.search("Early stop at epoch ([\d]+)", msg)
        n_epoch = n_epoch.group(1) if n_epoch is not None else 99
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        ade = round(float(metric.group(1)), 2)
        fde = round(float(metric.group(2)), 2)
        train_net = get_train_net(experiment)
        n_train = get_n_train(experiment)
        position = get_position(experiment, return_list=False)
        
        if position is not None:    
            label_name = f'TrS{train_seed}_{train_net}[{position}]({n_train})_{ade}/{fde}'
        
        if window is not None:
            val_ade = moving_average(df_curve.val_ade, window)
            val_fde = moving_average(df_curve.val_fde, window)
        else:
            val_ade = df_curve.val_ade
            val_fde = df_curve.val_fde

        start = 5
        # ade 
        p = axes[0].plot(df_curve.epoch[start:], df_curve.val_ade[start:], linewidth=0.5, alpha=0.5)
        axes[0].scatter(df_curve.epoch[start:], df_curve.val_ade[start:], c=p[-1].get_color(), s=1)
        axes[0].plot(df_curve.epoch[start:], val_ade[start:], c=p[-1].get_color())
        # fde 
        p = axes[1].plot(df_curve.epoch[start:], df_curve.val_fde[start:], linewidth=0.5, alpha=0.5)
        axes[1].scatter(df_curve.epoch[start:], df_curve.val_fde[start:], c=p[-1].get_color(), s=1)
        axes[1].plot(df_curve.epoch[start:], val_fde[start:], c=p[-1].get_color(), label=label_name)

    axes[0].set_ylabel('ADE')
    axes[1].set_ylabel('FDE')
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.savefig(out_path)   
    print('Saved', out_path)
    plt.close(fig)
    
    return df


def extract_test_score(test_msg):
    msg_split = re.split('save_every_n', test_msg)[1:]
    df = pd.DataFrame(columns=['seed', 'pretrained_ckpt', 'tuned_ckpt', 'ade', 'fde'])
    for msg in msg_split: 
        # metric = re.search("Round 0: \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)", msg)
        metric = re.search('Average performance \(by [\d]+\): \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)', msg)
        seed = re.search("'seed': ([\d+]),", msg)
        pretrained_ckpt = re.search("'pretrained_ckpt': '(.*?)',", msg)
        tuned_ckpt = re.search("'tuned_ckpt': '(.*?)',", msg)

        df = pd.concat([df, pd.DataFrame({
            'seed': seed.group(1) if seed is not None else None,
            'pretrained_ckpt': pretrained_ckpt.group(1).split('/')[-1] if pretrained_ckpt is not None else None,
            'tuned_ckpt': tuned_ckpt.group(1).split('/')[-1] if tuned_ckpt is not None else None,
            'ade': metric.group(1) if metric is not None else None, 
            'fde': metric.group(2) if metric is not None else None}, index=[0])], )
    df.seed = df.seed.astype(int)
    df.ade = df.ade.astype(float)
    df.fde = df.fde.astype(float)
    df['train_net'] = df['tuned_ckpt'].apply(lambda x: get_train_net(x))
    df['n_train'] = df['tuned_ckpt'].apply(lambda x: get_n_train(x))#.astype(int)
    df['position'] = df['tuned_ckpt'].apply(lambda x: get_position(x, return_list=False))
    df['lr'] = df['tuned_ckpt'].apply(lambda x: get_lr(x))
    df['is_ynet_bias'] = df['tuned_ckpt'].apply(lambda x: get_bool_bias(x))
    df['is_augment'] = df['tuned_ckpt'].apply(lambda x: get_bool_aug(x))
    # reorder columns 
    reordered_cols = ['seed', 'train_net', 'n_train', 'position', 'lr', 'is_ynet_bias', 'is_augment', 'ade', 'fde', 'tuned_ckpt', 'pretrained_ckpt']
    df = df.reindex(columns=reordered_cols)
    return df


def get_seed_number(ckpt_path):
    if ckpt_path is not None:
        return int(ckpt_path.split('__')[0].split('_')[-1])
    else:
        return None 


def get_train_net(ckpt_path):
    if ckpt_path is not None:
        return ckpt_path.split('__')[5]
    else:
        return None 


def get_n_train(ckpt_path):
    if ckpt_path is not None:
        if 'Pos' in ckpt_path: 
            n_train = int(ckpt_path.split('__')[7].split('_')[1])
        else:
            n_train = int(ckpt_path.split('__')[6].split('_')[1])
        return n_train
    else:
        return None 


def get_lr(ckpt_path):
    if ckpt_path is not None:
        if 'lr' in ckpt_path: 
            return ckpt_path.split('lr_')[1].split('_')[0].split('.pt')[0]
        else:
            return 0.00005
    else:
        return None 


def get_bool_bias(ckpt_path):
    if ckpt_path is not None:
        if 'bias' in ckpt_path.split('TrN')[-1]:
            return True
        else:
            return False 
    else:
        return None 


def get_bool_aug(ckpt_path):
    if ckpt_path is not None:
        if 'AUG' in ckpt_path:
            return True 
        else:
            return False 
    else:
        return None 


def extract_file(file_path, test_file_path, out_dir, window):
    with open(file_path, 'r') as f:
        msgs = f.read()
    if test_file_path is not None:
        with open(test_file_path, 'r') as f:
            test_msgs = f.read()
    else:
        test_msgs = None 

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    if '/' in file_path:
        file_name = re.search('/([^/]+).out', file_path).group(1)
    else:
        file_name = file_path.replace('.out', '')
    
    extract_curve(msgs, test_msgs, window, f'{out_dir}/{file_name}_{window}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default=None, type=str)
    parser.add_argument('--test_file_path', default=None, type=str)
    parser.add_argument('--window', default=9, type=int)
    parser.add_argument('--out_dir', default='csv/log', type=str)
    args = parser.parse_args()
    
    extract_file(args.file_path, args.test_file_path, args.out_dir, args.window)
