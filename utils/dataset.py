import os
import cv2
import torch
import random
import pathlib
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'figure.max_open_warning': 0})


def load_sdd_raw(path):
    data_path = os.path.join(path, "annotations")
    scenes_main = os.listdir(data_path)
    SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax',
                'frame', 'lost', 'occluded', 'generated', 'label']
    data = []
    for scene_main in sorted(scenes_main):
        scene_main_path = os.path.join(data_path, scene_main)
        for scene_sub in sorted(os.listdir(scene_main_path)):
            scene_path = os.path.join(scene_main_path, scene_sub)
            annot_path = os.path.join(scene_path, 'annotations.txt')
            scene_df = pd.read_csv(annot_path, header=0,
                                   names=SDD_cols, delimiter=' ')
            # Calculate center point of bounding box
            scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
            scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
            scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
            scene_df = scene_df.drop(
                columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost'])
            scene_df['sceneId'] = f"{scene_main}_{scene_sub.split('video')[1]}"
            # new unique id by combining scene_id and track_id
            scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
                                       zip(scene_df.sceneId, scene_df.trackId)]
            data.append(scene_df)
    data = pd.concat(data, ignore_index=True)
    rec_trackId2metaId = {}
    for i, j in enumerate(data['rec&trackId'].unique()):
        rec_trackId2metaId[j] = i
    data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
    data = data.drop(columns=['rec&trackId'])
    return data


def mask_step(x, step):
    """
    Create a mask to only contain the step-th element starting from the first element. Used to downsample
    """
    mask = np.zeros_like(x)
    mask[::step] = 1
    return mask.astype(bool)


def downsample(df, step):
    """
    Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
    df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
    pedestrian (metaId)
    :param df: pandas DataFrame - necessary to have column 'metaId'
    :param step: int - step size, similar to slicing-step param as in array[start:end:step]
    :return: pd.df - downsampled
    """
    mask = df.groupby(['metaId'])['metaId'].transform(mask_step, step=step)
    return df[mask]


def filter_short_trajectories(df, threshold):
    """
    Filter trajectories that are shorter in timesteps than the threshold
    :param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
    :param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
    :return: pd.df with trajectory length over threshold
    """
    len_per_id = df.groupby(by='metaId', as_index=False).count(
    )  # sequence-length for each unique pedestrian
    idx_over_thres = len_per_id[len_per_id['frame']
                                >= threshold]  # rows which are above threshold
    # only get metaIdx with sequence-length longer than threshold
    idx_over_thres = idx_over_thres['metaId'].unique()
    # filter df to only contain long trajectories
    df = df[df['metaId'].isin(idx_over_thres)]
    return df


def groupby_sliding_window(x, window_size, stride):
    x_len = len(x)
    n_chunk = (x_len - window_size) // stride + 1
    idx = []
    metaId = []
    for i in range(n_chunk):
        idx += list(range(i * stride, i * stride + window_size))
        metaId += ['{}_{}'.format(x.metaId.unique()[0], i)] * window_size
    df = x.iloc()[idx]
    df['newMetaId'] = metaId
    return df


def sliding_window(df, window_size, stride):
    """
    Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
    chunked trajectories are overlapping
    :param df: df
    :param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
    :param stride: timesteps to move from one trajectory to the next one
    :return: df with chunked trajectories
    """
    gb = df.groupby(['metaId'], as_index=False)
    df = gb.apply(groupby_sliding_window,
                  window_size=window_size, stride=stride)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    df = df.reset_index(drop=True)
    return df


def split_at_fragment_lambda(x, frag_idx, gb_frag):
    """ Used only for split_fragmented() """
    metaId = x.metaId.iloc()[0]
    counter = 0
    if metaId in frag_idx:
        split_idx = gb_frag.groups[metaId]
        for split_id in split_idx:
            x.loc[split_id:, 'newMetaId'] = '{}_{}'.format(metaId, counter)
            counter += 1
    return x


def split_fragmented(df):
    """
    Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
    Formally, this is done by changing the metaId at the fragmented frame and below
    :param df: DataFrame containing trajectories
    :return: df: DataFrame containing trajectories without fragments
    """

    gb = df.groupby('metaId', as_index=False)
    # calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
    df['frame_diff'] = gb['frame'].diff().fillna(value=1.0).to_numpy()
    # df containing all the first frames of fragmentation
    fragmented = df[df['frame_diff'] != 1.0]
    gb_frag = fragmented.groupby('metaId')  # helper for gb.apply
    frag_idx = fragmented.metaId.unique()  # helper for gb.apply
    df['newMetaId'] = df['metaId']  # temporary new metaId

    df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    return df


def load_raw_dataset(path, step, window_size, stride):
    df = load_sdd_raw(path=path)
    df = split_fragmented(df)  # split track if frame is not continuous
    df = downsample(df, step=step)
    df = filter_short_trajectories(df, threshold=window_size)
    df = sliding_window(df, window_size=window_size, stride=stride)
    return df


def rot(df, image, k=1):
    '''
    Rotates image and coordinates counter-clockwise by k * 90° within image origin
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :param k: Number of times to rotate by 90°
    :return: Rotated Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    for i in range(k):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def fliplr(df, image):
    '''
    Flip image and coordinates horizontally
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :return: Flipped Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    R = np.array([[-1, 0], [0, 1]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    image = cv2.flip(image, 1)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def augment_data(data, image_path='data/SDD/train', images={}, image_file='reference.jpg', seg_mask=False, use_raw_data=False):
    '''
    Perform data augmentation
    :param data: Pandas df, needs x,y,metaId,sceneId columns
    :param image_path: example - 'data/SDD/val'
    :param images: dict with key being sceneId, value being PIL image
    :param image_file: str, image file name
    :param seg_mask: whether it's a segmentation mask or an image file
    :return:
    '''
    ks = [1, 2, 3]
    for scene in data.sceneId.unique():
        scene_name, scene_idx = scene.split("_")
        if use_raw_data:
            im_path = os.path.join(
                image_path, scene_name, f"video{scene_idx}", image_file)
        else:
            im_path = os.path.join(image_path, scene, image_file)
        if seg_mask:
            im = cv2.imread(im_path, 0)
        else:
            im = cv2.imread(im_path)
        images[scene] = im
    # data without rotation, used so rotated data can be appended to original df
    data_ = data.copy()
    k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        metaId_max = data['metaId'].max()
        for scene in data_.sceneId.unique():
            if use_raw_data:
                im_path = os.path.join(
                    image_path, scene_name, f"video{scene_idx}", image_file)
            else:
                im_path = os.path.join(image_path, scene, image_file)
            if seg_mask:
                im = cv2.imread(im_path, 0)
            else:
                im = cv2.imread(im_path)

            data_rot, im = rot(data_[data_.sceneId == scene], im, k)
            # image
            rot_angle = k2rot[k]
            images[scene + rot_angle] = im

            data_rot['sceneId'] = scene + rot_angle
            data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
            data = pd.concat([data, data_rot], axis=0)

    metaId_max = data['metaId'].max()
    for scene in data.sceneId.unique():
        im = images[scene]
        data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
        data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
        data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
        data = pd.concat([data, data_flip], axis=0)
        images[scene + '_fliplr'] = im_flip

    return data, images


def resize_and_pad_image(images, size, pad=2019):
    """ Resize image to desired size and pad image to make it square shaped and ratio of images still is the same, as
    images all have different sizes.
    """
    for key, im in images.items():
        H, W, C = im.shape
        im = cv2.copyMakeBorder(
            im, 0, pad - H, 0, pad - W, cv2.BORDER_CONSTANT)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        images[key] = im


def create_images_dict(unique_scene, image_path, image_file='reference.jpg', use_raw_data=False):
    images = {}
    for scene in unique_scene:
        if image_file == 'oracle.png':
            im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
        else:
            if use_raw_data:
                scene_name, scene_idx = scene.split("_")
                im_path = os.path.join(
                    image_path, scene_name, f"video{scene_idx}", image_file)
            else:
                im_path = os.path.join(image_path, scene, image_file)
            im = cv2.imread(im_path)
        images[scene] = im
    # images channel: blue, green, red 
    return images


def load_images(scenes, image_path, image_file='reference.jpg'):
    images = {}
    if type(scenes) is list:
        scenes = set(scenes)
    for scene in scenes:
        if image_file == 'oracle.png':
            im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
        else:
            im = cv2.imread(os.path.join(image_path, scene, image_file))
        images[scene] = im
    return images


def get_varf_table(df, varf_list, obs_len):
    if obs_len:
        print(f'Computing variation fatcor by obs_len')
    else:
        print(f'Computing variation fatcor by obs_len + pred_len')

    df_varfs = df.groupby(['metaId', 'label', 'sceneId']).size().reset_index()[['metaId', 'label', 'sceneId']]
    df_varfs['scene'] = df_varfs.sceneId.apply(lambda x: x.split('_')[0])
    for varf in varf_list:
        df_stats = aggregate_per_varf_value(df, varf, obs_len)
        df_varfs = df_varfs.merge(df_stats[['metaId', varf]], on='metaId')
    return df_varfs
    

def aggregate_per_varf_value(df, varf, obs_len):
    out = df.groupby('metaId').apply(aggregate_per_varf_value_per_metaId, varf, obs_len)
    df_stats = pd.DataFrame(
        [[idx, item[0], item[1]] for idx, item in out.items()], 
        columns=['metaId', varf, 'label'])
    return df_stats


def aggregate_per_varf_value_per_metaId(df_meta, varf, obs_len):
    x = df_meta["x"].values
    y = df_meta["y"].values

    # sanity check
    unique_labels = np.unique(df_meta["label"].values)
    assert len(unique_labels) == 1
    label = unique_labels[0]

    unique_frame_step = (
        df_meta['frame'].shift(periods=-1) - df_meta['frame']).iloc[:-1].unique()
    assert len(unique_frame_step) == 1
    frame_step = unique_frame_step[0]

    op, attr = varf.split('_')

    # take the observed trajectory, or obs + pred
    if not obs_len:
        obs_len = len(x)
    
    # compute stats
    if attr == 'vel':
        stats_seqs = np.sqrt((x[: obs_len-1] - x[1: obs_len]) ** 2 + \
                             (y[: obs_len-1] - y[1: obs_len]) ** 2) / frame_step
    elif attr == 'acc':
        vel = np.sqrt((x[:obs_len-1] - x[1: obs_len]) ** 2 + \
                      (y[:obs_len-1] - y[1: obs_len]) ** 2) / frame_step
        stats_seqs = (vel[:obs_len-2] - vel[1: obs_len-1]) / frame_step
    elif attr == 'dist':
        stats_seqs = df_meta[:obs_len].dist.apply(
            lambda x: x.min() if not isinstance(x, float) else np.inf)
    elif 'den' in attr:
        stats_seqs = df_meta[:obs_len].dist.apply(
            lambda x: x[x < int(attr[3:])].shape[0] if not isinstance(x, float) else 0)
    else:
        raise ValueError(f'Cannot compute {attr} statistic')

    # take statistic for one sequence
    if op == 'max':
        stats = np.max(stats_seqs)
    elif op == 'avg':
        stats = np.mean(stats_seqs)
    elif op == 'min':
        stats = np.min(stats_seqs)
    elif op == 'abs+max':
        stats = np.max(np.abs(stats_seqs))
    elif op == 'abs+avg':
        stats = np.mean(np.abs(stats_seqs))
    elif op == 'abs+min':
        stats = np.mean(np.abs(stats_seqs))
    elif op == 'tot':
        stats = np.sum(stats_seqs)
    else:
        raise ValueError(f'Cannot compute {op} operation')

    return stats, label


def add_range_column(df, varf, varf_ranges, obs_len, inclusive='both'):
    df_stats = aggregate_per_varf_value(df, varf, obs_len)
    for r in varf_ranges:
        df_stats.loc[df_stats[varf].between(r[0], r[1], inclusive=inclusive), f'{varf}_range'] = f'{r[0]}_{r[1]}'
    df = df.merge(df_stats[['metaId', f'{varf}_range']], on='metaId')
    return df


def convert_df_to_dict(df_gb):
    varf_group_dict = dict()
    for g_range in df_gb.groups.keys():
        df_g = df_gb.get_group(g_range)[['metaId', 'sceneId', 'label']].drop_duplicates()
        assert df_g.metaId.nunique() == df_g.shape[0]
        varf_group_dict[g_range] = df_g.to_dict('list')
    return varf_group_dict


def create_dataset_by_agent_type(df, labels, out_dir, statistic_only, same_group_size=False):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_label = df[df.label.isin(labels)]
    df_gb = df_label.groupby(by='label', dropna=True)
    print('Statistics:\n', df_gb.count()['metaId'] / 20)
    print('# total:', (df_gb.count()['metaId'] / 20).sum())
    if not statistic_only:
        varf_group_dict = convert_df_to_dict(df_gb)
        for varf_range, varf_group in varf_group_dict.items():
            if same_group_size:
                min_n = min([len(g["metaId"]) for g in varf_group_dict.values()])
                meta_id_mask = reduce_group_size(varf_group, varf_range, min_n)
                df_varf = df_label[df_label.metaId.isin(varf_group['metaId'][meta_id_mask])]
            else:
                df_varf = df_label[df_label.metaId.isin(varf_group['metaId'])] 
            out_path = os.path.join(out_dir, f"{varf_range}.pkl")
            df_varf.to_pickle(out_path)
    

def create_customized_dataset(df, varf, varf_ranges, labels, out_dir, 
        obs_len, statistic_only, inclusive='both', same_group_size=False):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        varf (str | list of str): _description_
        varf_ranges (list of tuple | list of list of tuple): _description_
        labels (_type_): _description_
        out_dir (_type_): _description_
        obs_len (_type_): _description_
        statistic_only (bool): 
            Whether store the generated dataset or give statistic of categorized dataset only.
        inclusive (str):
            Choices = [both, right, left, neither]
        same_group_size (bool, optional): 
            _description_. Defaults to False.
    """
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_label = df[df.label.isin(labels)]

    # categorize by factor of variation
    if isinstance(varf, str):
        df_label = add_range_column(df_label, varf, varf_ranges, obs_len, inclusive=inclusive)
        varf_col_name = f'{varf}_range'
    elif isinstance(varf, list):
        for f, r in zip(varf, varf_ranges):
            df_label = add_range_column(df_label, f, r, obs_len, inclusive=inclusive)
        varf_col_name = '__'.join(varf)+'_range'
        nonan_mask = df_label.isna().any(axis=1)
        df_label.loc[~nonan_mask, varf_col_name] = \
            df_label.loc[~nonan_mask, [f + '_range' for f in varf]].agg('__'.join, axis=1)
    else:
        raise ValueError(f'Cannot process {varf}.')
    df_gb = df_label.groupby(by=varf_col_name, dropna=True)
    print('Statistics:\n', df_gb.count()['metaId'] / 20)
    print('# total:', (df_gb.count()['metaId'] / 20).sum())

    # save categorized dataset
    if not statistic_only:
        varf_group_dict = convert_df_to_dict(df_gb)
        for varf_range, varf_group in varf_group_dict.items():
            if same_group_size:
                min_n = min([len(g["metaId"]) for g in varf_group_dict.values()])
                meta_id_mask = reduce_group_size(varf_group, varf_range, min_n)
                df_varf = df_label[df_label.metaId.isin(varf_group['metaId'][meta_id_mask])]
            else:
                df_varf = df_label[df_label.metaId.isin(varf_group['metaId'])] 
            out_path = os.path.join(out_dir, f"{varf_range}.pkl")
            df_varf.to_pickle(out_path)


def reduce_group_size(varf_group, varf_range, min_n):
    print(f"Group {varf_range}")
    scene_ids, scene_counts = np.unique(
        varf_group["sceneId"], return_counts=True)
    sorted_unique_scene_counts = np.unique(np.sort(scene_counts))
    total_count = 0
    prev_count = 0
    mask = np.zeros_like(scene_counts).astype(bool)
    for scene_count in sorted_unique_scene_counts:
        total_count += (scene_counts >= scene_count).sum() * \
            (scene_count - prev_count)
        if total_count >= min_n:
            break
        mask[scene_counts == scene_count] = True
        prev_count = scene_count
    total_counts = np.zeros_like(scene_counts)
    total_counts[mask] = scene_counts[mask]
    total_counts[mask == False] = prev_count
    less = True
    while less:
        for i in np.where(mask == False)[0]:
            total_counts[i] += min(1, min_n - total_counts.sum())
            if min_n == total_counts.sum():
                less = False
                break
    varf_group["sceneId"] = np.array(varf_group["sceneId"])
    varf_group["metaId"] = np.array(varf_group["metaId"])
    varf_group["label"] = np.array(varf_group["label"])
    meta_id_mask = np.zeros_like(varf_group["metaId"]).astype(bool)
    for scene_idx, scene_id in enumerate(scene_ids):
        scene_count = total_counts[scene_idx]
        scene_mask = varf_group["sceneId"] == scene_id
        scene_labels = varf_group["label"][scene_mask]
        unique_scene_labels, scene_labels_count = np.unique(
            scene_labels, return_counts=True)
        scene_labels_chosen = []
        while len(scene_labels_chosen) < scene_count:
            for label_idx, (unique_scene_label, scene_label_count) in enumerate(zip(unique_scene_labels, scene_labels_count)):
                if scene_label_count != 0:
                    scene_labels_chosen.append(unique_scene_label)
                    scene_labels_count[label_idx] -= 1
                    if len(scene_labels_chosen) == scene_count:
                        break
        labels_chosen, labels_chosen_count = np.unique(
            scene_labels_chosen, return_counts=True)
        for label, label_count in zip(labels_chosen, labels_chosen_count):
            meta_id_idx = np.where(np.logical_and(
                varf_group["label"] == label, varf_group["sceneId"] == scene_id))[0][:label_count]
            meta_id_mask[meta_id_idx] = True
    return meta_id_mask


def compute_distance_with_neighbors(df_scene):
    return df_scene.apply(lambda_distance_with_neighbors, axis=1, df_scene=df_scene)


def lambda_distance_with_neighbors(row, df_scene, step=12):
    # start = datetime.datetime.now()
    frame_diff = df_scene.frame - row.frame
    df_sim = df_scene[(frame_diff < step/2) & \
        (frame_diff >= -step/2) & (df_scene.metaId != row.metaId)]
    dist = np.inf if df_sim.shape[0] == 0 else compute_distance_xy(df_sim, row.x, row.y)
    # duration = datetime.datetime.now() - start
    # print(f'### meta_id = {row.metaId}, time = {duration}')
    return dist


def compute_distance_xy(df_sim, x, y):
    x_diff = df_sim['x'] - x
    y_diff = df_sim['y'] - y
    dist = np.sqrt((x_diff ** 2 + y_diff ** 2))
    return np.array(dist)


def plot_varf_histograms(df_varf, out_dir):
    stats_all = np.array([])
    varf = df_varf.columns[-1]
    # Visualize data
    for label, indices in df_varf.groupby('label').groups.items():
        if label not in ["Pedestrian", "Biker"]:
            continue
        stats = df_varf.iloc[indices].loc[:, varf].values
        plot_histogram(stats, f'{label}_{varf}', out_dir)
        stats_all = np.append(stats_all, stats)
    plot_histogram(stats_all, f"Mixed_{varf}", out_dir)


def plot_varf_hist_obs_and_complete(df_varf, out_dir):
    varf_obs, varf_com = df_varf.columns[-2], df_varf.columns[-1]
    data_all_diff, data_all_obs, data_all_com = np.array([]), np.array([]), np.array([])
    # Visualize data
    for label, indices in df_varf.groupby('label').groups.items():
        if label not in ["Pedestrian", "Biker"]:
            continue
        data_obs = df_varf.iloc[indices].loc[:, varf_obs].values
        data_com = df_varf.iloc[indices].loc[:, varf_com].values
        data_diff = data_obs - data_com
        plot_histogram(data_diff, f'{label}_{varf_obs}_element_diff', out_dir)
        plot_histogram_overlay(data_obs, data_com, f'{label}_{varf_obs}_distr_diff', out_dir)
        data_all_diff = np.append(data_all_diff, data_diff)
        data_all_obs = np.append(data_all_obs, data_obs)
        data_all_com = np.append(data_all_com, data_com)
    plot_histogram(data_all_diff, f"Mixed_{varf_obs}_element_diff", out_dir)
    plot_histogram_overlay(data_all_obs, data_all_com, f'Mixed_{varf_obs}_distr_diff', out_dir)


def plot_histogram(data, title, out_dir, format='png'):
    fig = plt.figure()
    data, stats = filter_long_tail_arr(data)
    mean, std, min, max, p_zero, p_filter = stats
    sns.histplot(data, kde=True)
    plt.title(
        f"{title}, \nMean: {mean}, Std: {std}, Min: {min}, Max: {max}, Zero: {p_zero}, Filter: {p_filter}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title+'.'+format))
    plt.close(fig)


def plot_histogram_overlay(data_obs, data_com, title, out_dir, format='png'):
    fig = plt.figure()
    data_obs, _ = filter_long_tail_arr(data_obs)
    data_com, _ = filter_long_tail_arr(data_com)
    data_obs = data_obs[data_obs != 0]
    data_com = data_com[data_com != 0]
    df_obs = pd.DataFrame(data_obs, columns=['value'])
    df_obs['type'] = 'observation'
    df_com = pd.DataFrame(data_com, columns=['value'])
    df_com['type'] = 'complete'
    df_cat = pd.concat([df_obs, df_com], axis=0)
    df_cat = df_cat.reset_index(drop=True)
    sns.histplot(data=df_cat, x='value', hue="type")
    plt.title(title)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title+'.'+format))
    plt.close(fig)


def plot_pairplot(df_varfs, varf_list, label, title, out_dir, kind='kde', format='png'):
    if label == 'Mixed':
        df_label = df_varfs[
            (df_varfs.label == 'Pedestrian') | (df_varfs.label == 'Biker')]
    elif label == 'All':
        df_label = df_varfs
    else:
        df_label = df_varfs[df_varfs.label == label]

    fig = plt.figure()
    df_label_filter, p_filter = filter_long_tail_df(
        df_label[['metaId', 'scene', 'label']+varf_list], varf_list)
    plot_kws = dict(s=1) if kind == 'scatter' else None
    sns.pairplot(
        data=df_label_filter, 
        hue="scene", 
        vars=varf_list, 
        plot_kws=plot_kws,
        diag_kind="hist",
        kind=kind
    )
    title = f'{title}_{label}_{kind}_{str(p_filter)}'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title+'.'+format))
    plt.close(fig)


def plot_jointplot(df_varfs, varf_list, label, title, out_dir, hue, kind='kde', format='png'):
    if label == 'Mixed':
        df_label = df_varfs[
            (df_varfs.label == 'Pedestrian') | (df_varfs.label == 'Biker')]
    elif label == 'All':
        df_label = df_varfs
    else:
        df_label = df_varfs[df_varfs.label == label]

    for i, varf1 in enumerate(varf_list):
        for j, varf2 in enumerate(varf_list):
            if i < j:
                fig = plt.figure()
                df_label_filter, p_filter = filter_long_tail_df(
                    df_label[['metaId', 'scene', 'label', varf1, varf2]], [varf1, varf2])
                try:
                    sns.jointplot(data=df_label_filter, x=varf1, y=varf2, 
                        kind=kind, hue=hue)
                except np.linalg.LinAlgError:
                    kind = 'scatter'
                    sns.jointplot(data=df_label_filter, x=varf1, y=varf2, 
                        kind=kind, hue=hue)
                except:
                    print('Error!')
                title_save = f'{title}_{hue}_{label}_{varf1}_{varf2}_{kind}_{str(p_filter)}.{format}'
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(out_dir, title_save))
                plt.close(fig)


def plot_scene_w_numeric(df_varfs, varf, title, out_dir, format='png'):
    df_filter, p_filter = filter_long_tail_df(
        df_varfs[['metaId', 'scene', 'label', varf]], [varf])
    # filter out scene "quad"
    df_filter = df_filter[df_filter.scene != 'quad']

    unique_scenes = df_filter.scene.unique()
    n_scene = unique_scenes.shape[0]
    fig, axs = plt.subplots(4, n_scene+1, 
        figsize=(4*(n_scene+1), 16), sharex=True, sharey=True)
    binwidth = df_filter[varf].max() / 30
    for c, scene in enumerate(unique_scenes):
        data = df_filter[(df_filter.scene == scene)]
        axs[0, c].set_title(unique_scenes[c])
        # pedestrain
        sns.histplot(data=data[data.label == 'Pedestrian'], x=varf, 
            ax=axs[0, c], stat='probability', binwidth=binwidth)
        # biker 
        sns.histplot(data=data[data.label == 'Biker'], x=varf, 
            ax=axs[1, c], stat='probability', binwidth=binwidth)
        # mixed 
        sns.histplot(data=data[(data.label == 'Pedestrian') | (data.label == 'Biker')], x=varf, 
            ax=axs[2, c], hue='label', stat='probability', 
            hue_order=['Biker', 'Pedestrian'], binwidth=binwidth)
        # all 
        sns.histplot(data=data, x=varf, 
            ax=axs[3, c], stat='probability', binwidth=binwidth)
    axs[0, -1].set_title('All scenes')
    # pedestrain
    sns.histplot(data=df_filter[df_filter.label == 'Pedestrian'], x=varf, 
        ax=axs[0, -1], stat='probability', binwidth=binwidth)
    # biker 
    sns.histplot(data=df_filter[df_filter.label == 'Biker'], x=varf, 
        ax=axs[1, -1], stat='probability', binwidth=binwidth)
    # mixed 
    sns.histplot(data=df_filter[(df_filter.label == 'Pedestrian') | (df_filter.label == 'Biker')], x=varf, 
        ax=axs[2, -1], hue='label', stat='probability', 
        hue_order=['Biker', 'Pedestrian'], binwidth=binwidth)
    # all 
    sns.histplot(data=df_filter, x=varf, 
        ax=axs[3, -1], stat='probability', binwidth=binwidth)
    axs[0, 0].set_ylabel('Pedestrian')
    axs[1, 0].set_ylabel('Biker')
    axs[2, 0].set_ylabel('Pedestrian + Biker')
    axs[3, 0].set_ylabel('All agent types')
    plt.tight_layout()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{title}_scene_w_{varf}_{p_filter}_noquad.{format}'))
    plt.close(fig)


def filter_long_tail_arr(arr, n=3):
    # for statistics computing
    n_data = arr.shape[0]
    arr = arr[~np.isnan(arr) & (arr != np.inf)]
    if arr.shape[0]:
        mean = np.round(np.mean(arr), 2)
        std = np.round(np.std(arr), 2)
        min = np.round(np.min(arr), 2)
        max = np.round(np.max(arr), 2)
    else:
        raise ValueError('stats array is empty')
    p_zero = np.round((arr == 0).sum() / n_data, 2)
    arr = arr[
        (arr < mean + n * std) & (arr > mean - n * std) & (arr != 0)]
    p_filter = np.round((n_data - arr.shape[0]) / n_data, 2)
    return arr, (mean, std, min, max, p_zero, p_filter)


def filter_long_tail_series(series, n=3):
    full_index = series.index
    series = series[~series.isnull() & (series != np.inf)]
    if series.shape[0]:
        mean = np.round(series.mean(), 2)
        std = np.round(series.std(), 2)
    else:
        raise ValueError('Series is empty')
    series = series[
        (series < mean + n * std) & (series > mean - n * std) & (series != 0)]
    return full_index.drop(series.index)


def filter_long_tail_df(df_varfs, varf_list, n=3):
    idx_out = pd.Index([])
    for varf in varf_list:
        idx_out = idx_out.append(filter_long_tail_series(df_varfs[varf]))
    idx_out_unique = idx_out.unique()
    df_varfs_filter = df_varfs.drop(idx_out_unique)
    p_filter = round(len(idx_out_unique) / df_varfs.shape[0], 2)
    return df_varfs_filter, p_filter


def dataset_split(
    data_path, train_files, val_split=0.1, n_leftouts=None, 
    shuffle_data=False, share_val_test=False, given_scenes=None, given_test_meta_ids=None):
    if n_leftouts:
        df_train, df_val, df_test = dataset_split_given_ratio_and_leftout(
            data_path, train_files, val_split, n_leftouts, shuffle_data, share_val_test, given_test_meta_ids)
    elif given_scenes:
        df_test = dataset_split_given_scenes(
            data_path, train_files, given_scenes)
        df_train, df_val = None, None
    else:
        # TODO: cases where train files and val files do not overlap 
        raise ValueError('No implementation in this case')
    return df_train, df_val, df_test 


def dataset_split_given_ratio_and_leftout(
    data_path, train_files, val_split, n_leftouts=None, 
    shuffle_data=False, share_val_test=False, given_test_meta_ids=None):
    print(f"Split {train_files} given val_split={val_split}, n_leftout={n_leftouts}")
    df_train, df_val, df_test = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    for train_file, n_leftout in zip(train_files, n_leftouts):
        df_train_ = pd.read_pickle(os.path.join(data_path, train_file))
        df_train_, df_val_, df_test_ = split_train_val_test(
            df_train_, val_split, n_leftout, share_val_test, shuffle_data, given_test_meta_ids)
        df_train = pd.concat([df_train, df_train_])
        df_val = pd.concat([df_val, df_val_])
        if df_test_ is not None:
            df_test = pd.concat([df_test, df_test_])
    return df_train, df_val, df_test


def split_train_val_test(
    df, val_split, n_test=None, share_val_test=False, shuffle_data=False, given_test_meta_ids=None):
    # idx
    unique_meta_ids = np.unique(df["metaId"])
    if shuffle_data:
        print('Shuffling data')
        np.random.shuffle(unique_meta_ids)
    n_metaId = unique_meta_ids.shape[0]
    n_val = int(val_split) if val_split > 1 else int(val_split * n_metaId)
    # split
    if n_test:
        if share_val_test:
            print('Share validation and test set')
            n_train = n_metaId - n_test 
            train_meta_ids, test_meta_ids = np.split(unique_meta_ids, [n_train])
            if n_val != 0:
                interval = n_test // n_val if n_test // n_val > 1 else 3
                val_meta_ids = test_meta_ids[::interval]
                df_val = reduce_df_meta_ids(df, val_meta_ids)
            else:
                df_val = None
            df_test = reduce_df_meta_ids(df, test_meta_ids)
        else:
            print('Validation and test sets are independent')
            n_train = n_metaId - n_val - n_test 
            train_meta_ids, val_meta_ids, test_meta_ids = np.split(unique_meta_ids, [n_train, n_train + n_val])
            if given_test_meta_ids is not None:
                test_meta_ids = given_test_meta_ids
                print('Replaced test set by given test meta ids')
            df_test = reduce_df_meta_ids(df, test_meta_ids)
            df_val = reduce_df_meta_ids(df, val_meta_ids)
    else:
        n_train = n_metaId - n_val
        val_meta_ids, train_meta_ids = np.split(
            unique_meta_ids, [n_train])
        df_test = None
        df_val = reduce_df_meta_ids(df, val_meta_ids)
    df_train = reduce_df_meta_ids(df, train_meta_ids)
    return df_train, df_val, df_test


def reduce_df_meta_ids(df, meta_ids):
    return df[(df["metaId"].values == meta_ids[:, None]).sum(axis=0).astype(bool)]


def dataset_split_given_scenes(data_path, files, scenes):
    print(f"Split {files} given scenes={scenes}")
    df = pd.concat([pd.read_pickle(os.path.join(data_path, file)) for file in files])
    df_selected = df[df.sceneId.isin(scenes)]
    return df_selected


def get_meta_ids_focus(df=None, given_meta_ids=None, given_csv=None, random_n=None):
    if given_meta_ids is not None:
        if isinstance(given_meta_ids, int):
            meta_ids_focus = [given_meta_ids]
        elif isinstance(given_meta_ids, list):
            meta_ids_focus = given_meta_ids
        else:
            raise ValueError(f'Invalid given_meta_ids={given_meta_ids}')
    elif given_csv['path'] is not None:
        path = given_csv['path']
        col1, col2, op = given_csv['name'].split('__')
        n_limited = given_csv['n_limited']
        df_result = pd.read_csv(path)
        if op == 'diff':
            df_result.loc[:, 'diff'] = df_result[col1].values - df_result[col2].values
        elif op == 'abs_diff':
            df_result.loc[:, 'diff'] = np.abs(df_result[col1].values - df_result[col2].values)
        else:
            raise ValueError(f'Invalid op={op}')
        meta_ids_focus = df_result.sort_values(
            by='diff', ascending=False).head(n_limited).metaId.values
    elif random_n is not None:
        unique_meta_ids = df.metaId.unique() 
        np.random.shuffle(unique_meta_ids)
        meta_ids_focus = unique_meta_ids[:random_n]
    else:
        meta_ids_focus = df.metaId.unique() 
    print('Focusing on meta_ids=', meta_ids_focus)
    return meta_ids_focus


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    cv2.setRNGSeed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def limit_samples(df, num, batch_size, random_ids=True):
    if num is None:
        return df
    num_total = num * batch_size
    meta_ids = np.unique(df["metaId"])
    if random_ids:
        np.random.shuffle(meta_ids)
    meta_ids = meta_ids[:num_total]
    df = reduce_df_meta_ids(df, meta_ids)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default='data/sdd/raw', type=str)
    parser.add_argument("--raw_data_name", default='data.pkl', type=Str)
    parser.add_argument("--filter_data_dir", default='data/sdd/filter', type=str)
    parser.add_argument("--reload", action='store_true')
    parser.add_argument("--statistic_only", action="store_true", 
                        help='False to save also the categorized dataset')
    
    parser.add_argument("--step", default=12, type=int)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--stride", default=20, type=int)
    parser.add_argument("--obs_len", default=8, type=int)

    parser.add_argument("--varf", default='avg_vel', 
                        help="variation factors from: 'avg_vel', 'max_vel', "+\
                            "'avg_acc', 'max_acc', 'abs+max_acc', 'abs+avg_acc', "+\
                            "'min_dist', 'avg_den50', 'avg_den100', 'agent_type'")
    parser.add_argument("--varf_ranges", help='range of varation factor to take',
                        default=[(0.25, 0.7), (1, 3)])
                        # default=[(0.1, 0.3), (0.5, 1.5)])  # small gap 
                        # default=[(0.1, 0.2), (0.6, 1.4)])  # big gap 
                        # default=[(0, 1.3), (1.7, 4.3)])  # small gap, two groups
                        # default=[(0, 0.3), (0.7, 1.3), (1.7, 2.3), (2.7, 3.3), (3.7, 4.3)])  # small gap 
                        # default=[(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3.2), (3.8, 4.2)])  # big gap 
                        # default=[[(0.1, 0.3), (0.5, 1.5)], [(0, 0.3), (0.7, 1.3), (1.7, 2.3), (2.7, 3.3), (3.7, 4.3)]])
                        # default=[[(0.1, 0.2), (0.6, 1.4)], [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3.2), (3.8, 4.2)]])

    parser.add_argument("--labels", default=['Pedestrian', 'Biker'], nargs='+', type=str,
                        choices=['Biker', 'Bus', 'Car', 'Cart', 'Pedestrian', 'Skater'])

    parser.add_argument("--vis", action='store_true') 

    args = parser.parse_args()
    args.labels.sort()
    print(args)

    # ============== load raw dataset ===============
    if not args.reload:
        # ## load raw dataset
        df = load_raw_dataset(args.raw_data_dir, args.step, args.window_size, args.stride)
        print('Loaded raw dataset')
        # possibly add a column of distance with neighbors 
        if 'dist' in args.varf or 'den' in args.varf or np.array(['dist' in f or 'den' in f for f in args.varf]).any():
            out = df.groupby('sceneId').apply(compute_distance_with_neighbors)
            for idx_1st in out.index.get_level_values('sceneId').unique():
                df.loc[out[idx_1st].index, 'dist'] = out[idx_1st].values
            print(f'Added a column of distance with neighbors to df')
        # save
        out_path = os.path.join(args.raw_data_dir, f"data_.pkl")
        df.to_pickle(out_path)
        print(f'Saved data to {out_path}')
    else:  # reload = True
        # ## or load from stored pickle
        df = pd.read_pickle(os.path.join(args.raw_data_dir, args.raw_data_name))
        print('Reloaded raw dataset')


    # ================= plot =================
    if args.vis:
        varf_list = ['avg_vel', 'max_vel', 'avg_acc', 'max_acc', 
                    'abs+max_acc', 'abs+avg_acc', 'min_dist', 'avg_den100', 'avg_den50']

        if not args.reload:
            # ## get variation factor table 
            df_varfs = get_varf_table(df, varf_list, args.obs_len)
            df_varfs_com = get_varf_table(df, varf_list, None)
            df_varfs = df_varfs.merge(
                df_varfs_com.drop(['label', 'sceneId', 'scene'], axis=1), 
                on='metaId', suffixes=('', '_com'))
            out_path = os.path.join('data/sdd/raw', "df_varfs.pkl")
            df_varfs.to_pickle(out_path)
            print(f'Saved df_varfs to {out_path}')
        else:
            # ## or load from stored one
            df_varfs = pd.read_pickle(os.path.join(args.raw_data_dir, "raw/df_varfs.pkl"))
            print('Loaded df_varfs')

        for varf in varf_list:
            # plot_varf_hist_obs_and_complete(df_varfs[['label', varf, varf+'_com']], 'figures/filtered_distr/hist/diff')
            plot_varf_histograms(df_varfs[['label', varf]], 'figures/filtered_distr/hist/obs')
            plot_scene_w_numeric(df_varfs, varf, 'Bivar', 'figures/filtered_distr/bivar')

        for label in ['Pedestrian', 'Biker', 'Mixed', 'All']:
            plot_jointplot(df_varfs, varf_list,  label, 'Joint', 'figures/bivar_distr/filter', 'scene', kind='kde')
        plot_jointplot(df_varfs, varf_list, 'All', 'Joint', 'figures/bivar_distr/filter', 'label', kind='kde')


    # ============== create designed dataset ================
    if isinstance(args.varf, str):
        out_dir = os.path.join(args.filter_data_dir, args.varf, '_'.join(args.labels))
    elif isinstance(args.varf, list):
        out_dir = os.path.join(args.filter_data_dir, '__'.join(args.varf), '_'.join(args.labels))
    else:
        raise ValueError(f'Cannot process {varf}')
    if args.varf == 'agent_type':
        out_dir = os.path.join(args.filter_data_dir, args.varf)
        create_dataset_by_agent_type(df, args.labels, out_dir, statistic_only=args.statistic_only)
    else:
        create_customized_dataset(df, args.varf, args.varf_ranges, args.labels, 
            out_dir, obs_len=args.obs_len, statistic_only=args.statistic_only)
    print(f'Created dataset by {args.varf} using {args.labels}')
