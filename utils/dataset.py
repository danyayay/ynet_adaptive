import numpy as np
import torch
import random
import pandas as pd
import os
import cv2
import argparse
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib


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
            data = data.append(data_rot)

    metaId_max = data['metaId'].max()
    for scene in data.sceneId.unique():
        im = images[scene]
        data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
        data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
        data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
        data = data.append(data_flip)
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


def create_images_dict(data, image_path, image_file='reference.jpg', use_raw_data=False):
    images = {}
    for scene in data.sceneId.unique():
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


def gather_all_stats(df, varf, obs_len):
    stats_dict = {}
    meta_ids = np.unique(df["metaId"].values)
    for meta_id in meta_ids:
        stats, label = compute_stats_per_metaId(df, meta_id, varf, obs_len)
        if label not in stats_dict:
            stats_dict[label] = []
        stats_dict[label] += [stats]
    return stats_dict


def compute_stats_per_metaId(df, meta_id, varf, obs_len):
    df_meta = df[df["metaId"] == meta_id]
    x = df_meta["x"].values
    y = df_meta["y"].values

    unique_labels = np.unique(df_meta["label"].values)
    assert len(unique_labels) == 1
    label = unique_labels[0]
    
    frame_steps = []
    for frame_idx, frame in enumerate(df_meta["frame"].values):
        if frame_idx != len(df_meta["frame"].values) - 1:
            frame_steps.append(df_meta["frame"].values[frame_idx + 1] - frame)
    unique_frame_step = np.unique(frame_steps)
    assert len(unique_frame_step) == 1
    frame_step = unique_frame_step[0]
    stats_seqs = []
    op, attr = varf.split('_')

    # take only the observed trajectory, instead of obs + pred
    for i in range(obs_len):
        if attr == 'vel':
            if i != obs_len - 1:
                stats_seq = math.sqrt((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2) / frame_step
                stats_seqs.append(stats_seq)
        elif attr == 'acc':
            if i != obs_len - 2:
                # todo: check divide by frame_step
                stats_seq = (math.sqrt((x[i+2] - x[i+1]) ** 2 + (y[i+2] - y[i+1]) ** 2) - 
                             math.sqrt((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2)) / frame_step ** 2
                stats_seqs.append(stats_seq)
        else:  # attr == 'dist'
            pass

    # take statistic for one sequence
    if op == 'max':
        stats = np.max(stats_seqs)
    elif op == 'avg':
        stats = np.mean(stats_seqs)
    else:  # op == 'min'
        stats = np.min(stats_seqs)

    return stats, label


def create_dataset_by_varf(df, varf, varf_ranges, labels, out_dir, obs_len):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    varf_group_dict = {varf_range: {"metaId": [], "sceneId": [], "label": []}
                       for varf_range in varf_ranges}

    # categorize by factor of variation
    meta_ids = np.unique(df["metaId"].values)
    for meta_id in meta_ids:
        stats, label = compute_stats_per_metaId(df, meta_id, varf, obs_len)
        if label not in labels:
            continue
        for varf_range in varf_group_dict.keys():
            varf_min, varf_max = varf_range
            if stats >= varf_min and stats <= varf_max:
                varf_group_dict[varf_range]["metaId"].append(meta_id)
                unique_scene_ids = np.unique(
                    df[df["metaId"] == meta_id]["sceneId"].values)
                assert len(unique_scene_ids) == 1
                scene_id = unique_scene_ids[0]
                varf_group_dict[varf_range]["sceneId"].append(scene_id)
                varf_group_dict[varf_range]["label"].append(label)

    # keep each group with the same number of data
    min_n_metas = min([len(varf_group["metaId"]) for varf_group in varf_group_dict.values()])
    for varf_range, varf_group in varf_group_dict.items():
        scene_ids, scene_counts = np.unique(
            varf_group["sceneId"], return_counts=True)
        sorted_unique_scene_counts = np.unique(np.sort(scene_counts))
        total_count = 0
        prev_count = 0
        mask = np.zeros_like(scene_counts).astype(bool)
        for scene_count in sorted_unique_scene_counts:
            total_count += (scene_counts >= scene_count).sum() * \
                (scene_count - prev_count)
            if total_count >= min_n_metas:
                break
            mask[scene_counts == scene_count] = True
            prev_count = scene_count
        total_counts = np.zeros_like(scene_counts)
        total_counts[mask] = scene_counts[mask]
        total_counts[mask == False] = prev_count
        less = True
        while less:
            for i in np.where(mask == False)[0]:
                total_counts[i] += min(1, min_n_metas - total_counts.sum())
                if min_n_metas == total_counts.sum():
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

        df_varf = df[np.array(
            [df["metaId"] == meta_id for meta_id in varf_group["metaId"][meta_id_mask]]).any(axis=0)]
        varf_range_name = f"{varf_range[0]}_{varf_range[1]}"
        df_varf["varf_range"] = varf_range_name
        out_path = os.path.join(out_dir, f"{varf_range_name}.pkl")
        df_varf.to_pickle(out_path)


def plot_varf_histograms(df, out_dir, varf, obs_len, bin=4):
    stats_dict = gather_all_stats(df, varf, obs_len)
    stats_all = []
    # Visualize data
    for label, stats in stats_dict.items():
        if label not in ["Pedestrian", "Biker"]:
            continue
        plot_histogram(stats, f'{label}_{varf}', out_dir, bin)
        stats_all += stats
    plot_histogram(stats_all, f"All_{varf}", out_dir, bin)


def plot_histogram(stats, title, out_dir, bin=4):
    fig = plt.figure()
    mean = np.round(np.mean(stats), 2)
    std = np.round(np.std(stats), 2)
    min = np.round(np.min(stats), 2)
    max = np.round(np.max(stats), 2)
    n_zero = np.round((np.array(stats) == 0).sum() / len(stats), 2)
    stats = np.sort(stats)[int(len(stats) * 0.00): int(len(stats) * 0.99)]
    stats = stats[stats != 0]
    plt.hist(stats, bins=bin)
    plt.title(
        f"{title}, Mean: {mean}, Std: {std}, Min: {min}, Max: {max}, Zero: {n_zero}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title))
    plt.close(fig)


def plot_varf_histograms_sns(df, out_dir, varf, obs_len):
    stats_dict = gather_all_stats(df, varf, obs_len)
    stats_all = []
    # Visualize data
    for label, stats in stats_dict.items():
        if label not in ["Pedestrian", "Biker"]:
            continue
        plot_histogram_sns(stats, f'sns_{label}_{varf}', out_dir)
        stats_all += stats
    plot_histogram_sns(stats_all, f"sns_All_{varf}", out_dir)


def plot_histogram_sns(stats, title, out_dir):
    fig = plt.figure()
    mean = np.round(np.mean(stats), 2)
    std = np.round(np.std(stats), 2)
    min = np.round(np.min(stats), 2)
    max = np.round(np.max(stats), 2)
    n_zero = np.round((np.array(stats) == 0).sum() / len(stats), 2)
    stats = np.sort(stats)[int(len(stats) * 0.00): int(len(stats) * 0.99)]
    stats = stats[stats != 0]
    sns.histplot(stats, kde=True)
    plt.title(
        f"{title}, Mean: {mean}, Std: {std}, Min: {min}, Max: {max}, Zero: {n_zero}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title))
    plt.close(fig)
    

def split_df_ratio(df, ratio):
    meta_ids = np.unique(df["metaId"])
    test_meta_ids, train_meta_ids = np.split(
        meta_ids, [int(ratio * len(meta_ids))])
    df_train = reduce_df_meta_ids(df, train_meta_ids)
    df_test = reduce_df_meta_ids(df, test_meta_ids)
    return df_train, df_test


def reduce_df_meta_ids(df, meta_ids):
    return df[(df["metaId"].values == meta_ids[:, None]).sum(axis=0).astype(bool)]


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
    parser.add_argument("--data_raw", default='sdd_ynet/dataset_raw', type=str)
    parser.add_argument("--data_filter", default='sdd_ynet/dataset_filter', type=str)

    parser.add_argument("--step", default=12, type=int)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--stride", default=20, type=int)
    parser.add_argument("--obs_len", default=8, type=int)

    parser.add_argument("--varf", default='avg_vel', type=str, help='variation factor',
                        choices=['avg_vel', 'max_vel', 'avg_acc', 'max_acc', 'min_dist'])
    parser.add_argument("--varf_ranges", help='range of varation factor to take',
                        default=[(0.25, 0.75), (1.25, 1.75), (2.25, 2.75), (3.25, 3.75)])

    parser.add_argument("--labels", default=['Pedestrian', 'Biker'], nargs='+', type=str,
                        choices=['Biker', 'Bus', 'Car', 'Cart', 'Pedestrian', 'Skater'])
    args = parser.parse_args()

    # Create dataset
    print('Loading raw dataset')
    df = load_raw_dataset(args.data_raw, args.step,
                          args.window_size, args.stride)

    print(f'Creating dataset by {args.varf}')
    create_dataset_by_varf(df, args.varf, args.varf_ranges,
                           args.labels, args.data_filter, args.obs_len)
    print('Done')
