from multiprocessing.sharedctypes import Value
import re
import argparse
import pathlib
import pandas as pd


def extract_train_msg(test_msg):
    msg_split = re.split('save_every_n', test_msg)[1:]
    df = pd.DataFrame(columns=['seed', 'pretrained_ckpt', 'experiment', 'n_param', 'n_epoch', 'ade', 'fde'])
    for msg in msg_split: 
        metric = re.search("Round 0: \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)", msg)
        n_epoch = re.search("Early stop at epoch ([\d]+)", msg)
        df = pd.concat([df, pd.DataFrame({
            'seed': re.search("'seed': ([\d+]),", msg).group(1),
            'pretrained_ckpt': re.search("'pretrained_ckpt': '(.*?)',", msg).group(1).split('/')[1],
            'experiment': re.search("Experiment (.*?) has started", msg).group(1),
            'n_param': re.search("The number of trainable parameters: ([\d]+)", msg).group(1),
            'n_epoch': n_epoch.group(1) if n_epoch is not None else 99,
            'ade': metric.group(1), 
            'fde': metric.group(2)}, index=[0])], ignore_index=True)
    df.seed = df.seed.astype(int)
    df.n_param = df.n_param.astype(int)
    df.n_epoch = df.n_epoch.astype(int)
    df.ade = df.ade.astype(float)
    df.fde = df.fde.astype(float)
    df['train_net'] = df['experiment'].apply(lambda x: get_train_net(x))
    df['n_train'] = df['experiment'].apply(lambda x: get_n_train(x)).astype(int)
    df['adapter_position'] = df['experiment'].apply(lambda x: get_adapter_position(x))
    # reorder columns 
    reordered_cols = ['seed', 'train_net', 'n_train', 'adapter_position', 'n_param', 'n_epoch', 'ade', 'fde', 'experiment', 'pretrained_ckpt']
    df = df.reindex(columns=reordered_cols)
    return df


def extract_test_msg(test_msg):
    msg_split = re.split('save_every_n', test_msg)[1:]
    df = pd.DataFrame(columns=['seed', 'pretrained_ckpt', 'tuned_ckpt', 'ade', 'fde'])
    for msg in msg_split: 
        metric = re.search("Round 0: \nTest ADE: ([\d\.]+) \nTest FDE: ([\d\.]+)", msg)
        df = pd.concat([df, pd.DataFrame({
            'seed': re.search("'seed': ([\d+]),", msg).group(1),
            'pretrained_ckpt': re.search("'pretrained_ckpt': '(.*?)',", msg).group(1).split('/')[1],
            'tuned_ckpt': re.search("'tuned_ckpt': '(.*?)',", msg).group(1).split('/')[1],
            'ade': metric.group(1) if metric is not None else None, 
            'fde': metric.group(2) if metric is not None else None}, index=[0])], ignore_index=True)
    df.seed = df.seed.astype(int)
    df.ade = df.ade.astype(float)
    df.fde = df.fde.astype(float)
    df['train_net'] = df['tuned_ckpt'].apply(lambda x: get_train_net(x))
    df['n_train'] = df['tuned_ckpt'].apply(lambda x: get_n_train(x)).astype(int)
    df['adapter_position'] = df['tuned_ckpt'].apply(lambda x: get_adapter_position(x))
    # reorder columns 
    reordered_cols = ['seed', 'train_net', 'n_train', 'adapter_position', 'ade', 'fde', 'tuned_ckpt', 'pretrained_ckpt']
    df = df.reindex(columns=reordered_cols)
    return df


def get_train_net(ckpt_path):
    return ckpt_path.split('__')[5]


def get_n_train(ckpt_path):
    if 'adapter' in ckpt_path:
        n_train = ckpt_path.split('__')[7].split('_')[1].split('.')[0]
    elif 'weight' in ckpt_path:
        n_train = ckpt_path.split('__')[6].split('_')[1]
    else:
        n_train = ckpt_path.split('__')[6].split('_')[1].split('.')[0]
    return n_train


def get_adapter_position(ckpt_path):
    if 'adapter' in ckpt_path:
        return ckpt_path.split('__')[6]
    else:
        return None 


def extract_file(file_path, out_dir):
    with open(file_path, 'r') as f:
        msgs = f.read()
    if 'eval' in file_path:
        df = extract_test_msg(msgs)
    elif 'train' in file_path:
        df = extract_train_msg(msgs)
    else:
        raise ValueError('Unclear eval/train function to use')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    file_name = re.search('/([^/]+).out', file_path).group(1)
    out_name = f'{out_dir}/{file_name}.csv'
    print(out_name)
    df.to_csv(out_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default=None, type=str)
    parser.add_argument('--out_dir', default='csv/log', type=str)
    args = parser.parse_args()
    
    extract_file(args.file_path, args.out_dir)
