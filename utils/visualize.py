import os
import cv2
import glob
import pathlib
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')
from utils.dataset import create_images_dict


labels = {"all": "vanilla fine-tuning",
          "encoder": "encoder adaptation", 
          "modulator": "modulator adaptation"}

labels_ckpt = {
    'GT': 'groundtruth',
    'INDG': 'in-domain generalization',
    'OODG': 'out-of-domain generalization',
    'FT': 'full finetuning',
    'ET': 'encoder finetuning'
}


colors = {
    'OODG': 'tab:blue', 'FT': 'tab:orange', 'ET': 'tab:red', 
    'diff_OODG_FT': 'tab:orange', 'diff_OODG_ET': 'tab:red'}


def create_few_shot_plot(results_dir, out_dir, fontsize=16):
    update_modes = sorted(os.listdir(results_dir))
    ades = {}
    for update_mode in update_modes:
        update_mode_dir = os.path.join(results_dir, update_mode)
        seeds = os.listdir(update_mode_dir)
        ades[update_mode] = {}
        for seed in seeds:
            seed_dir = os.path.join(update_mode_dir, seed)
            num_files = os.listdir(seed_dir)
            for num_file in num_files:
                num = int(num_file.split('.csv')[0])
                num_path = os.path.join(seed_dir, num_file)
                # float(pd.read_csv(num_path).columns[0])
                ade = float(pd.read_csv(num_path).values[0][0])
                if num not in ades[update_mode]:
                    ades[update_mode][num] = []
                ades[update_mode][num].append(ade)
            zero_shot_path = results_dir.split("/")
            zero_shot_path[-2] = "None"
            zero_shot_path += ['eval', seed, '0.csv']
            zero_shot_path = '/'.join(zero_shot_path)
            if os.path.isfile(zero_shot_path):
                # float(pd.read_csv(num_path).columns[0])
                ade = float(pd.read_csv(zero_shot_path).values[0][0])
                num = 0
                if num not in ades[update_mode]:
                    ades[update_mode][num] = []
                ades[update_mode][num].append(ade)

    f, ax = plt.subplots(figsize=(6, 4))
    for train_name, train_vals in ades.items():
        v = [i for j in list(train_vals.values()) for i in j]
        k = [j for j in list(train_vals.keys())
             for _ in range(len(list(train_vals.values())[0]))]
        df = pd.DataFrame({'x': k, 'y': v})
        sns.lineplot(data=df, x='x', y='y',
                     label=labels[train_name], ax=ax, marker="o")
        sns.despine()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.ylabel('ADE', fontsize=fontsize)
    plt.xlabel('# Batches', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.savefig(f'{out_dir}/result.png', bbox_inches='tight', pad_inches=0)


def plot_input_space(semantic_images, observed_map, meta_ids, scene_id, out_dir='figures', format='png'):
    # semantic_images: (batch_size, n_class, height, width)
    # observed_map: (batch_size, obs_len, height, width)
    fig, axes = plt.subplots(2, observed_map.shape[1], figsize=(observed_map.shape[1]*4, 2*4))
    for i, meta_id in enumerate(meta_ids):
        observed_map_i = observed_map[i]
        semantic_image_i = semantic_images[i]
        # plot semantic map
        for c in range(semantic_image_i.shape[0]):
            im = axes[0, c].imshow(semantic_image_i[c], vmin=0, vmax=1, interpolation='nearest')
        divider = make_axes_locatable(axes[0, c])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        axes[0, c // 2].set_title('Semantic map')
        # hide empty plots
        for c in range(semantic_image_i.shape[0], observed_map_i.shape[0]):
            axes[0, c].axis('off')
        # plot observed trajectory map
        for t in range(observed_map_i.shape[0]):
            axes[1, t].imshow(observed_map_i[t], vmin=0, vmax=1)
        axes[1, t // 2].set_title('Observed trajectory map')
        # save
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_name = f'{meta_id}__{scene_id}'
        out_path = os.path.join(out_dir, out_name + '.' + format)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')


def plot_feature_space(dict_features, out_dir='figures/feature_space', show_diff=True, format='png'):
    """Plot feature space 

    Args:
        dict_features (dict): 
            {ckpt_name: 
                {scene_id: 
                    {feature_name:
                        {np.array()},
                     ...,
                     meta_ids:
                        list(),
                    }
                }
            }
        out_dir (str, optional): _description_. Defaults to 'figures/feature_space'.
        plot_diff (bool, optional): _description_. Defaults to False.
        format (str, optional): _description_. Defaults to 'png'.
    """
    # TODO: show colorbar 
    # TODO: add plot only first k figures
    first_dict = dict_features[list(dict_features)[0]]
    for scene_id, dict_scene in first_dict.items():
        for i, meta_id in enumerate(dict_scene['metaId']):
            features_name = list(dict_scene)
            features_name.remove('metaId')
            # for each sample, visualize feature space
            if show_diff:
                # show the difference between OODG and FT / ET
                for _, feature_name in enumerate(features_name):
                    n_channel = dict_scene[feature_name].shape[1]
                    diff_i = {}
                    for ckpt_name in ['FT', 'ET']:
                        if ('OODG' in dict_features.keys()) & (ckpt_name in dict_features.keys()):
                            diff_i[ckpt_name] = dict_features['OODG'][scene_id][feature_name][i] - \
                                dict_features[ckpt_name][scene_id][feature_name][i]
                    height, width = diff_i[ckpt_name][0].shape
                    while height >= 6:
                        height /= 2
                        width /= 2
                    fig, axes = plt.subplots(
                        len(diff_i), n_channel, figsize=(n_channel*width, len(diff_i)*height))
                    for k, ckpt_name in enumerate(diff_i):
                        for c in range(n_channel):
                            if len(axes.shape) == 1:
                                axes[c].imshow(diff_i[ckpt_name][c])
                                axes[c].set_xlabel(f'channel_{c+1}')
                                if c == 0: axes[c].set_ylabel(labels_ckpt[ckpt_name])
                            else:
                                axes[k, c].imshow(diff_i[ckpt_name][c])
                                axes[k, c].set_xlabel(f'channel_{c+1}')
                                if c == 0: axes[k, c].set_ylabel(labels_ckpt[ckpt_name])
                    title = f'meta_id={meta_id}, scene_id={scene_id}, feature_name={feature_name}'
                    if len(axes.shape) == 1:
                        axes[n_channel//2].set_title(title)
                    else:
                        axes[0, n_channel//2].set_title(title)
                    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                    out_name = f'{meta_id}__{scene_id}__{feature_name}_diff'
                    out_path = os.path.join(out_dir, out_name + '.' + format)
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved {out_path}')
            else:
                # show the original feature space 
                for _, feature_name in enumerate(features_name):
                    n_channel = dict_scene[feature_name].shape[1]
                    n_ckpt = len(dict_features)
                    height, width = dict_ckpt[scene_id][feature_name][i].shape
                    while height >= 6:
                        height /= 2
                        width /= 2
                    fig, axes = plt.subplots(n_ckpt, n_channel, 
                        figsize=(n_channel*width, n_ckpt*height))
                    for k, (ckpt_name, dict_ckpt) in enumerate(dict_features.items()):
                        feature_i = dict_ckpt[scene_id][feature_name][i]  # (n_channel, height, width)
                        for c in range(n_channel):
                            if len(axes.shape) == 1:
                                axes[c].imshow(feature_i[c])
                                axes[c].set_xlabel(f'channel_{c+1}')
                                if c == 0: axes[c].set_ylabel(labels_ckpt[ckpt_name])
                            else:
                                axes[k, c].imshow(feature_i[c])
                                axes[k, c].set_xlabel(f'channel_{c+1}')
                                if c == 0: axes[k, c].set_ylabel(labels_ckpt[ckpt_name])
                    title = f'meta_id={meta_id}, scene_id={scene_id}, feature_name={feature_name}'
                    if len(axes.shape) == 1:
                        axes[n_channel//2].set_title(title)
                    else:
                        axes[0, n_channel//2].set_title(title)
                    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                    out_name = f'{meta_id}__{scene_id}__{feature_name}'
                    out_path = os.path.join(out_dir, out_name + '.' + format)
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved {out_path}')


def plot_feature_space_diff_evolution(
    dict_features, out_dir='figures/feature_space_diff', 
    encoder_only=False, diff_type='absolute', 
    by_scene=True, format='png'):
    """Plot the difference of OODG and FT/ET along with layers. 

    Args:
        dict_features (dict): 
            Dict storing all features. 
        out_dir (str, optional): 
            Path for figures. Defaults to 'figures/feature_space_diff'.
        encoder_only (bool, optional): 
            Visualize only encoder or the whole network. Defaults to False.
        diff_type (str, optional): 
            [absolute, relative]. Defaults to 'absolute'.
        by_scene (str, optional): 
            Defaults to True.
        format (str, optional): 
            format. Defaults to 'png'.

    Raises:
        ValueError: _description_
    """
    diff_dict, df_dict, original_dict, ckpt_scene_dict = {}, {}, {}, {}
    df_original = pd.DataFrame()
    for ckpt_name in ['FT', 'ET']:
        if ('OODG' in dict_features.keys()) & (ckpt_name in dict_features.keys()):
            name = f'diff_OODG_{ckpt_name}'
            diff_dict[name] = {}
            original_dict['OODG'] = {}
            original_dict[ckpt_name] = {}
            ckpt_scene_dict[ckpt_name] = {}
            for s, (scene_id, dict_scene) in enumerate(dict_features[list(dict_features)[0]].items()):
                features_name = list(dict_scene)
                features_name.remove('metaId')
                if encoder_only:
                    features_name = [f for f in features_name if 'Encoder' in f]
                if s == 0:
                    for feature_name in features_name:
                        diff_dict[name][feature_name] = []
                        original_dict['OODG'][feature_name] = []
                        original_dict[ckpt_name][feature_name] = []
                index = dict_scene['metaId']
                ckpt_scene_df = pd.DataFrame(index=index)
                for feature_name in features_name:
                    # diff using all pixels and channels
                    n_tot = dict_features['OODG'][scene_id][feature_name][0].reshape(-1).shape[0]
                    original_oodg = dict_features['OODG'][scene_id][feature_name]
                    original_ckpt = dict_features[ckpt_name][scene_id][feature_name]
                    diff = original_oodg - original_ckpt

                    if diff_type == 'overall_relative':
                        add = diff.mean(axis=(1,2,3)) / original_oodg.mean(axis=(1,2,3))
                    elif diff_type == 'pixel_relative':
                        add = np.empty(diff.shape)
                        add.fill(np.nan)
                        np.divide(diff, original_oodg, out=add, where=original_oodg!=0)
                        add = np.nanmean(add, axis=(1,2,3))
                    else:
                        raise ValueError(f'No support for diff_type={diff_type}')
                    
                    diff_dict[name][feature_name].extend(add)
                    ckpt_scene_df.loc[index, feature_name] = add

                    original_dict['OODG'][feature_name].extend(original_oodg.sum(axis=(1,2,3))/n_tot)
                    original_dict[ckpt_name][feature_name].extend(original_ckpt.sum(axis=(1,2,3))/n_tot)
                
                ckpt_scene_dict[ckpt_name][scene_id] = ckpt_scene_df
            
            # average over samples
            df = pd.DataFrame()
            n_data = len(diff_dict[name][features_name[0]])
            for feature_name in features_name:
                df.loc[feature_name, name] = np.mean(diff_dict[name][feature_name])
                df.loc[feature_name, name+'_std'] = np.std(diff_dict[name][feature_name])
                df_original.loc[feature_name, 'OODG'] = np.mean(original_dict['OODG'][feature_name])
                df_original.loc[feature_name, 'OODG_std'] = np.std(original_dict['OODG'][feature_name])
                df_original.loc[feature_name, ckpt_name] = np.mean(original_dict[ckpt_name][feature_name])
                df_original.loc[feature_name, ckpt_name+'_std'] = np.std(original_dict[ckpt_name][feature_name])
            df_dict[name] = df

            # plot configuration
            if df.shape[0] == 3:
                fig_size, depth = 4, 0
            elif df.shape[0] == 20: # 6+7+7
                fig_size, depth = 10, 1
            elif df.shape[0] == 83: # 23+30+30
                fig_size, depth = 20, 2
            else: 
                # when encoder_only=True, depth will be unknown
                fig_size, depth = df.shape[0]*0.25 + 4, df.shape[0]
    
    # ## feature space difference plot along with layers 
    fig, ax = plt.subplots(figsize=(fig_size, 4))
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    for name in df_dict.keys():
        df = df_dict[name]
        for b, block in enumerate(['Encoder', 'GoalDecoder', 'TrajDecoder']):
            df_block = df[df.index.str.contains(block)]
            if df_block.shape[0]:
                if b == 0:
                    plt.plot(df_block.index, df_block[name].values, '.-', c=colors[name], label=name)
                else:
                    plt.plot(df_block.index, df_block[name].values, '.-', c=colors[name])
                plt.fill_between(df_block.index, df_block[name].values - df_block[name+'_std'].values, 
                    df_block[name].values + df_block[name+'_std'].values, color=colors[name], alpha=0.2)
    plt.title('Feature space difference')
    plt.ylabel(f'{diff_type} difference')
    plt.xlabel('Layers')
    plt.legend(loc='best')
    if (depth == 0) | (depth == 1) | (encoder_only):
        plt.xticks(rotation=45, ha='right')
    else: # (depth == 2) | (depth == unknown)
        plt.xticks(rotation=90, ha='right')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = f'{"_".join(df_dict.keys())}__D{depth}__N{n_data}__{diff_type}'
    if encoder_only: out_name += '__encoder'
    out_path = os.path.join(out_dir, f'{out_name}.{format}')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    # ## plot the original values in feature space 
    fig, ax = plt.subplots(figsize=(fig_size, 4))
    for ckpt_name in original_dict.keys():
        for b, block in enumerate(['Encoder', 'GoalDecoder', 'TrajDecoder']):
            df_block = df_original[df_original.index.str.contains(block)]
            if df_block.shape[0]:
                if b == 0:
                    plt.plot(df_block.index, df_block[ckpt_name].values, '.-', c=colors[ckpt_name], label=ckpt_name)
                else:
                    plt.plot(df_block.index, df_block[ckpt_name].values, '.-', c=colors[ckpt_name])
                plt.fill_between(df_block.index, df_block[ckpt_name].values - df_block[ckpt_name+'_std'].values, 
                    df_block[ckpt_name].values + df_block[ckpt_name+'_std'].values, color=colors[ckpt_name], alpha=0.2)
    plt.title('Feature space')
    plt.ylabel('Value')
    plt.xlabel('Layers')
    plt.legend(loc='best')
    if (depth == 0) | (depth == 1) | (encoder_only):
        plt.xticks(rotation=45, ha='right')
    else: # (depth == 2) | (depth == unknown)
        plt.xticks(rotation=90, ha='right')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = f'{"_".join(original_dict.keys())}__D{depth}__N{n_data}'
    if encoder_only: out_name = f'{out_name}__encoder'
    out_path = os.path.join(out_dir, f'{out_name}.{format}')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    # ## plot by_scene
    if by_scene:
        for ckpt_name, scene_dict in ckpt_scene_dict.items():
            for scene_id in scene_dict.keys():
                fig, ax = plt.subplots(figsize=(fig_size, 4))
                plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                df = scene_dict[scene_id]
                for i, meta_id in enumerate(df.index):
                    for b, block in enumerate(['Encoder', 'GoalDecoder', 'TrajDecoder']):
                        # plot each example
                        cols = df.columns[df.columns.str.contains(block)]
                        example = df.loc[meta_id, cols].to_numpy()
                        if example.shape[0]:
                            plt.plot(cols, example, c=colors[ckpt_name], linewidth=0.5, alpha=0.3)
                        # plot average
                        mean = df.loc[:, cols].mean(axis=0).to_numpy()
                        if (i == 0) & (b == 0):
                            plt.plot(cols, mean, '.-', c=colors[ckpt_name], label=f'diff_OODG_{ckpt_name}')
                        else:
                            plt.plot(cols, mean, '.-', c=colors[ckpt_name])
                plt.title(f'Feature space difference ({scene_id})')
                plt.ylabel(f'{diff_type} difference')
                plt.xlabel('Layers')
                plt.legend(loc='best')
                if (depth == 0) | (depth == 1) | (encoder_only):
                    plt.xticks(rotation=45, ha='right')
                else: # (depth == 2) | (depth == unknown)
                    plt.xticks(rotation=90, ha='right')
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_name = f'diff_OODG_{ckpt_name}__{scene_id}__D{depth}__N{n_data}__{diff_type}'
                if encoder_only: out_name = f'{out_name}__encoder'
                out_path = os.path.join(out_dir, f'{out_name}.{format}')
                plt.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {out_path}') 


def plot_trajectories_scenes_overlay(image_path, df_biker, df_ped, out_dir='figures/scene_with_trajs', format='png'):
    unique_scene = list(set(df_biker.sceneId.unique()).intersection(set(df_ped.sceneId.unique())))
    scene_images = create_images_dict(unique_scene, image_path, 'reference.jpg', True)
    for scene_id in unique_scene:
        print(f'Plotting {scene_id}')
        scene_biker = df_biker[df_biker.sceneId == scene_id]
        scene_ped = df_ped[df_ped.sceneId == scene_id]
        height, width = scene_images[scene_id].shape[0], scene_images[scene_id].shape[1]
        fig = plt.figure(figsize=(height/100, width/100))
        plt.imshow(scene_images[scene_id])
        ms = 2
        for _, traj in scene_biker.groupby('metaId'):
            plt.scatter(traj.x, traj.y, s=ms, c='r', alpha=0.4)
            plt.plot(traj.x, traj.y, 'r-', ms=ms, alpha=0.2)
        plt.plot(0,0,'r-', alpha=0.5, label='Biker')

        for _, traj in scene_ped.groupby('metaId'):
            plt.scatter(traj.x, traj.y, s=ms, c='b', alpha=0.4)
            plt.plot(traj.x, traj.y, 'b-', alpha=0.2)
        plt.plot(0,0,'b-', alpha=0.5, label='Pedestrian')

        plt.plot(0,0,'w')
        plt.title(f'scene: {scene_id}')
        plt.legend(loc='best')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, scene_id + '.' + format)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')


def plot_obs_pred_trajs(image_path, dict_trajs, out_dir='figures/prediction', format='png', obs_len=8):
    first_dict = dict_trajs[list(dict_trajs)[0]]
    scene_images = create_images_dict(first_dict['sceneId'], image_path, 'reference.jpg', True)
    colors = {'OB': 'black', 'GT': 'green', 'INDG': 'cyan', 'OODG': 'blue', 'FT': 'orange', 'ET': 'red'}
    for i, meta_id in enumerate(first_dict['metaId']):
        scene_id = first_dict['sceneId'][i]
        scene_image = scene_images[scene_id]
        fig = plt.figure(figsize=(scene_image.shape[0]/100, scene_image.shape[1]/100))
        plt.imshow(scene_image)
        ms = 3
        for j, (ckpt_name, value) in enumerate(dict_trajs.items()):
            gt_traj = value['groundtruth'][i]
            pred_traj = value['prediction'][i]
            if j == 0:
                plt.plot(gt_traj[:obs_len,0], gt_traj[:obs_len,1], 
                    '.-', ms=ms, c=colors['OB'], label='observed')
                plt.plot(gt_traj[(obs_len-1):,0], gt_traj[(obs_len-1):,1], 
                    '.-', ms=ms, c=colors['GT'], label=labels_ckpt['GT'])
            plt.plot([gt_traj[obs_len-1,0], pred_traj[0,0]], [gt_traj[obs_len-1,1], pred_traj[0,1]],
                '.-', ms=ms, c=colors[ckpt_name])
            plt.plot(pred_traj[:,0], pred_traj[:,1], 
                '.-', ms=ms, c=colors[ckpt_name], label=labels_ckpt[ckpt_name])
        title = f'meta_id={meta_id}, scene_id={scene_id}'
        plt.title(title)
        plt.legend(loc='best')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_name = f'{meta_id}__{scene_id}'
        out_path = os.path.join(out_dir, out_name + '.'+ format)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')


def plot_decoder_overlay(image_path, dict_features, out_dir='figures/decoder', format='png', resize_factor=0.25):
    # take decoder name 
    first_ckpt_dict = dict_features[list(dict_features)[0]]
    if 'GoalDecoder' in first_ckpt_dict[list(first_ckpt_dict)[0]]:
        goal_dec_name, traj_dec_name = 'GoalDecoder', 'TrajDecoder'
    elif 'GoalDecoder_B7' in first_ckpt_dict[list(first_ckpt_dict)[0]]:
        goal_dec_name, traj_dec_name = 'GoalDecoder_B7', 'TrajDecoder_B7'
    else: # 'GoalDecoder_B7_L1' in first_ckpt_dict[list(first_ckpt_dict)[0]]
        goal_dec_name, traj_dec_name = 'GoalDecoder_B7_L1', 'TrajDecoder_B7_L1'
    # take unique scene images
    scene_images = create_images_dict(
        first_ckpt_dict.keys(), image_path, 'reference.jpg', True)
    for scene_id, dict_scene in first_ckpt_dict.items():
        for i, meta_id in enumerate(dict_scene['metaId']):
            scene_image = scene_images[scene_id]
            scene_image = cv2.resize(
                scene_image, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            height, width = scene_image.shape[0], scene_image.shape[1]
            # for each sample, visualize overlayed feature space
            for _, feature_name in enumerate([goal_dec_name, traj_dec_name]):
                n_channel = dict_scene[feature_name].shape[1]
                n_ckpt = len(dict_features)
                fig, axes = plt.subplots(
                    n_ckpt, n_channel, figsize=(n_channel*width/100, n_ckpt*height/100))
                for k, (ckpt_name, dict_ckpt) in enumerate(dict_features.items()):
                    feature_i = dict_ckpt[scene_id][feature_name][i]  # (n_channel, height, width)
                    for c in range(n_channel):
                        axes[k, c].imshow(scene_image)
                        axes[k, c].imshow(feature_i[c], alpha=0.7, cmap='coolwarm')                     
                        if c == 0: 
                            axes[k, c].set_ylabel(labels_ckpt[ckpt_name])
                        else:
                            axes[k, c].set_yticklabels([])
                        if k != (n_ckpt - 1): 
                            axes[k, c].set_xticklabels([])
                        else:
                            axes[k, c].set_xlabel(f'channel_{c+1}')  
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
                title = f'meta_id={meta_id}, scene_id={scene_id}, feature_name={feature_name}'
                axes[0, n_channel//2].set_title(title)
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_name = f'{meta_id}__{scene_id}__{feature_name}__overlay'
                out_path = os.path.join(out_dir, out_name + '.' + format)
                plt.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {out_path}')
    

def plot_filters(model_dict, out_dir='figures/filters', format='png'):
    for model_name, model in model_dict.items():
        for param_name, param in model.model.named_parameters():
            if param_name.startswith(('encoder', 'goal_decoder', 'traj_decoder')):
                if param_name.endswith('weight'):
                    c_out, c_in, height, width = param.shape
                    vmin, vmax = param.min().item(), param.max().item()
                    fig, axes = plt.subplots(c_in, c_out, figsize=(c_out*width, c_in*height))
                    for o in range(c_out):
                        for i in range(c_in):
                            im = axes[i, o].imshow(param[o, i].cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                            axes[i, o].set_xticklabels([])
                            axes[i, o].set_yticklabels([])
                    divider = make_axes_locatable(axes[0, c_out-1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.tight_layout()
                    axes[0, c_out//2-1].set_title('Out channels')#
                    axes[c_in//2-1, 0].set_ylabel('In channels')
                    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                    out_name = f'{model_name}__{param_name}'
                    out_path = os.path.join(out_dir, out_name + '.' + format)
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved {out_path}')                        


def plot_filters_diff_evolution(
    model_dict, out_dir='figures/filters_diff', format='png'):
    if 'OODG' in model_dict.keys():
        df_filters = pd.DataFrame()
        for model_name, model in model_dict.items():
            if model_name == 'OODG':
                for param_name, param in model.model.named_parameters():
                    if not param_name.startswith('semantic_segmentation'):
                        df_filters.loc[param_name, model_name+'__sum'] = param.sum().item()
                        df_filters.loc[param_name, model_name+'__avg'] = param.mean().item()
            else:
                name = f'diff_OODG_{model_name}'
                for (_, param_oodg), (param_name, param) in zip(
                    model_dict['OODG'].model.named_parameters(), model.model.named_parameters()):
                    if not param_name.startswith('semantic_segmentation'):
                        # df_filters.loc[param_name, model_name] = param.sum().item()
                        # d1 = (param_oodg - param).sum().item() / param_oodg.sum().item()
                        # d2 = ((param_oodg - param) / param_oodg).mean().item()
                        # d3 = (param_oodg - param).sum().item()
                        # print(model_name, param_name, d1, d2, d3)
                        df_filters.loc[param_name, model_name+'__sum'] = param.sum().item()
                        df_filters.loc[param_name, model_name+'__avg'] = param.mean().item()
                        # overall relative: can be distorted by outliers 
                        df_filters.loc[param_name, name+'__overall_relative'] = \
                            (param_oodg - param).sum().item() / param_oodg.sum().item()
                        # pixel_relative: can be smoothed by the mostly low values in filters; std is possible..(big)
                        df_filters.loc[param_name, name+'__pixel_relative'] = \
                            ((param_oodg - param) / param_oodg).mean().item()
                        # absolute
                        df_filters.loc[param_name, name+'__absolute'] = \
                            (param_oodg - param).sum().item()

        colors.update({
            'OODG_weight': 'tab:blue', 'FT_weight': 'tab:orange', 'ET_weight': 'tab:red',
            'OODG_bias': 'lightsteelblue', 'FT_bias': 'navajowhite', 'ET_bias': 'pink',
            'diff_OODG_FT_weight': 'tab:orange', 'diff_OODG_FT_bias': 'navajowhite',
            'diff_OODG_ET_weight': 'tab:red', 'diff_OODG_ET_bias': 'pink'})
        fig_width = df_filters.shape[0]*0.25+3
        mask_w = df_filters.index.str.contains('weight')
        mask_b = df_filters.index.str.contains('bias')
        index = [n.rstrip('.weight') for n in df_filters.index[mask_w]]
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        # absolute values: plot
        for op in ['sum', 'avg']:
            fig, ax = plt.subplots(figsize=(fig_width, 4))
            for model_name in model_dict.keys():
                plt.plot(index, df_filters.loc[mask_w, f'{model_name}__{op}'], 
                    '-', c=colors[model_name], label=model_name+'_weight')
                plt.plot(index, df_filters.loc[mask_b, f'{model_name}__{op}'], 
                    '--', c=colors[model_name], label=model_name+'_bias')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.title('Filters')
            plt.ylabel('Value')
            plt.xlabel('Layers')
            plt.legend(loc='best')
            plt.xticks(rotation=45, ha='right')
            out_name = f'filters_{"_".join(model_dict.keys())}__{op}__plot.{format}'
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out_path}') 

        # absolute values: bar plot
        df_dict = {}
        for op in ['sum', 'avg']:
            df = pd.DataFrame(index=index)
            for model_name in model_dict.keys():
                df.loc[index, model_name+'_weight'] = df_filters.loc[mask_w, f'{model_name}__{op}'].values
                df.loc[index, model_name+'_bias'] = df_filters.loc[mask_b, f'{model_name}__{op}'].values
            df_dict[op] = df
            df.plot(kind='bar', color=[colors.get(x) for x in df.columns],
                figsize=(fig_width, 4), title='Filters',
                xlabel='Layers', ylabel='Value', rot=45, legend=True)
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            out_name = f'filters_{"_".join(model_dict.keys())}__{op}__bar.{format}'
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f'Saved {out_path}') 

        # barplot separately 
        for op in ['sum', 'avg']:
            df = df_dict[op]
            for name in ['weight', 'bias']:
                cols = [c for c in df.columns if c.endswith(name)]
                df[cols].plot(kind='bar', color=[colors.get(x) for x in cols],
                    figsize=(fig_width/1.7, 4), title='Filters',
                    xlabel='Layers', ylabel='Value', rot=45, legend=True)
                plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                out_name = f'filters_{"_".join(model_dict.keys())}__{op}__bar__{name}.{format}'
                out_path = os.path.join(out_dir, out_name)
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                print(f'Saved {out_path}') 

        # plot filters' difference 
        for diff_type in ['overall_relative', 'pixel_relative', 'absolute']:
            # plot 
            fig, ax = plt.subplots(figsize=(fig_width, 4))
            columns = df_filters.columns[
                df_filters.columns.str.startswith('diff') & df_filters.columns.str.endswith(diff_type)]
            for column in columns:
                name, _ = column.split('__')
                plt.plot(index, df_filters.loc[mask_w, column], '-', c=colors[name], label=name+'_weight')
                plt.plot(index, df_filters.loc[mask_b, column], '--', c=colors[name], label=name+'_bias')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.title('Filters')
            plt.xlabel('Layers')
            plt.legend(loc='best')
            plt.xticks(rotation=45, ha='right')
            out_name = f'filters_diff_{"_".join(model_dict.keys())}__{diff_type}__plot.{format}'
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out_path}') 
            
            # barplot 
            df = pd.DataFrame(index=index)
            for column in columns:
                name, _ = column.split('__')
                df.loc[index, name+'_weight'] = df_filters.loc[mask_w, column].values
                df.loc[index, name+'_bias'] = df_filters.loc[mask_b, column].values
            df.plot(kind='bar', color=[colors.get(x) for x in df.columns],
                figsize=(fig_width, 4), title='Filters',
                xlabel='Layers', ylabel='Value', legend=True)
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            out_name = f'filters_diff_{"_".join(model_dict.keys())}__{diff_type}__bar.{format}'
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f'Saved {out_path}') 

            # barplot separately 
            for name in ['weight', 'bias']:
                cols = [c for c in df.columns if c.endswith(name)]
                df[cols].plot(kind='bar', color=[colors.get(x) for x in cols],
                    figsize=(fig_width/1.7, 4), title='Filters',
                    xlabel='Layers', ylabel='Value', rot=45, legend=True)
                plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                out_name = f'filters_diff_{"_".join(model_dict.keys())}__{diff_type}__bar__{name}.{format}'
                out_path = os.path.join(out_dir, out_name)
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                print(f'Saved {out_path}')             
    else:
        raise ValueError('No generalization model found')
    

def plot_per_importance_analysis(
    tuned_name, df, n_test, scene_id, depth,
    ade_oodg_mean, fde_oodg_mean, ade_oodg_std, fde_oodg_std, 
    ade_tuned_mean, fde_tuned_mean, ade_tuned_std, fde_tuned_std, 
    out_dir='figures/importance_analysis', format='png', plot_err_bar=False
):
    print('OODG:')
    print(f'ADE mean={round(ade_oodg_mean, 2)}, FDE mean={round(fde_oodg_mean, 2)}')
    print(f'ADE std={round(ade_oodg_std, 2)}, FDE std={round(fde_oodg_std, 2)}')

    print(tuned_name, ':')
    print(f'ADE mean={round(ade_tuned_mean, 2)}, FDE mean={round(fde_tuned_mean, 2)}')
    print(f'ADE std={round(ade_tuned_std, 2)}, FDE std={round(fde_tuned_std, 2)}')
    
    tuned_diff = {
        'ade_diff': ade_oodg_mean - ade_tuned_mean, 
        'fde_diff': fde_oodg_mean - fde_tuned_mean
    }

    # bar plot 
    for metric in ['ade_diff', 'fde_diff']:
        fig_width = df.shape[0] * 0.25 + 3
        if depth == -1:
            # plot bias and weight with two colors
            colors = {'weight': 'tab:blue', 'bias': 'lightsteelblue'}
            mask_w = df.index.str.contains('weight')
            mask_b = df.index.str.contains('bias')
            index = [n.rstrip('.weight') for n in df.index[mask_w]]
            df_data = pd.DataFrame(index=index)
            df_data.loc[index, 'weight'] = df.loc[mask_w, metric].values
            df_data.loc[index, 'bias'] = df.loc[mask_b, metric].values
            if plot_err_bar:
                df_err = pd.DataFrame(index=index)
                df_err.loc[index, 'weight'] = df.loc[mask_w, metric+'_std'].values
                df_err.loc[index, 'bias'] = df.loc[mask_b, metric+'_std'].values 
                df_data.plot(kind='bar', color=[colors.get(c) for c in df_data.columns], 
                    figsize=(fig_width/1.7, 4), yerr=df_err, xlabel='Layers', ylabel=metric, 
                    title='Importance analysis' if not scene_id else f'Importance analysis ({scene_id})')
            else:
                df_data.plot(kind='bar', color=[colors.get(c) for c in df_data.columns], 
                    figsize=(fig_width/1.7, 4), xlabel='Layers', ylabel=metric, 
                    title='Importance analysis' if not scene_id else f'Importance analysis ({scene_id})')
        elif (depth == 1) or (depth == 2):
            # organize index
            df.index = df.reset_index()['layer'].apply(
                lambda x: x.split(',')[-1] if len(x.split(','))==1 else x.split(',')[-1].lstrip(" '").rstrip("']"))
            df = df.sort_values(by='layer')
            # plot
            if plot_err_bar:
                df[[metric]].plot(kind='bar', 
                    yerr=df[[metric+'_std']].rename(columns={metric+'_std': metric}), 
                    figsize=(fig_width/1.3, 4), xlabel='Layers', ylabel=metric, 
                    title='Importance analysis' if not scene_id else f'Importance analysis ({scene_id})')
            else:
                df[[metric]].plot(kind='bar',
                    figsize=(fig_width/1.3, 4), xlabel='Layers', ylabel=metric, 
                    title='Importance analysis' if not scene_id else f'Importance analysis ({scene_id})')
        else:
            raise ValueError('No support for depth={depth}')
        plt.axhline(y=tuned_diff[metric], color='tab:red', 
            linestyle='--', linewidth=1, alpha=0.5, label=f'diff_OODG_{tuned_name}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc="upper right")
        if not scene_id:
            out_name = f'{tuned_name}_{metric}__N{n_test}' 
        else:
            out_name = f'{tuned_name}_{metric}__N{n_test}__{scene_id}'
        if plot_err_bar: out_name += f'__err.{format}' 
        out_path = os.path.join(out_dir, out_name)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}') 


def plot_importance_analysis(
    in_dir, out_dir='figures/importance_analysis', format='png', 
    n_test=500, depth=-1, plot_err_bar=False):

    # pretrained models
    df_oodg = pd.read_csv(f'{in_dir}/OODG__N{n_test}.csv')
    ade_oodg_mean, fde_oodg_mean = df_oodg.ade.mean(), df_oodg.fde.mean()
    ade_oodg_std, fde_oodg_std = df_oodg.ade.std(), df_oodg.fde.std()

    # tuned models
    for tuned_name in ['FT', 'ET']:
        # results 
        if not os.path.exists(f'{in_dir}/{tuned_name}__N{n_test}.csv'):
            continue

        df_tuned = pd.read_csv(f'{in_dir}/{tuned_name}__N{n_test}.csv')
    
        ade_tuned_mean, fde_tuned_mean = df_tuned.ade.mean(), df_tuned.fde.mean()
        ade_tuned_std, fde_tuned_std = df_tuned.ade.std(), df_tuned.fde.std()

        # results after replacing one layer
        df_avg, df_sample = pd.DataFrame(), pd.DataFrame()
        
        # collect files 
        pattern = f'{in_dir}/{tuned_name}__N{n_test}__*.csv'
        file_names = glob.glob(pattern)
        
        if file_names: 
            for file_name in file_names: 
                layer_name = file_name.split('__')[-1].replace('.csv', '')
                df_file = pd.read_csv(file_name) 
                df_file['layer'] = layer_name
                df_file.loc[:, 'ade_diff'] = df_oodg.ade - df_file.ade
                df_file.loc[:, 'fde_diff'] = df_oodg.fde - df_file.fde
                df_avg = pd.concat([df_avg, pd.DataFrame({
                    'layer': layer_name, 
                    'ade_diff': df_file.ade_diff.mean(),
                    'fde_diff': df_file.fde_diff.mean(),
                    'ade_diff_std': df_file.ade_diff.std(),
                    'fde_diff_std': df_file.fde_diff.std()}, 
                    index=[0])], ignore_index=True, axis=0)
                df_sample = pd.concat([df_sample, df_file], ignore_index=True, axis=0)
            df_avg = df_avg.sort_values(by='layer', ascending=True)
            df_avg.set_index('layer', drop=True, inplace=True)

            # plot averaged case      
            plot_per_importance_analysis(
                tuned_name, df_avg, n_test, None, depth, 
                ade_oodg_mean, fde_oodg_mean, ade_oodg_std, fde_oodg_std, 
                ade_tuned_mean, fde_tuned_mean, ade_tuned_std, fde_tuned_std, 
                out_dir, plot_err_bar=False
            )
            plot_per_importance_analysis(
                tuned_name, df_avg, n_test, None, depth, 
                ade_oodg_mean, fde_oodg_mean, ade_oodg_std, fde_oodg_std, 
                ade_tuned_mean, fde_tuned_mean, ade_tuned_std, fde_tuned_std, 
                out_dir, plot_err_bar=True
            )

            # plot by scene 
            df_gb = df_sample.groupby(by=['sceneId', 'layer']).agg(['mean', 'std']).reset_index()
            df_selected = df_gb[['sceneId', 'layer']].copy()
            df_selected.loc[:, 'ade_diff'] = df_gb['ade_diff']['mean']
            df_selected.loc[:, 'fde_diff'] = df_gb['fde_diff']['mean']
            df_selected.loc[:, 'ade_diff_std'] = df_gb['ade_diff']['std']
            df_selected.loc[:, 'fde_diff_std'] = df_gb['fde_diff']['std']
            for scene_id in df_selected.sceneId.unique():
                df_scene = df_selected[df_selected.sceneId == scene_id]
                df_scene = df_scene.sort_values(by='layer', ascending=True)
                df_scene.set_index('layer', drop=True, inplace=True)
                # TODO: legend format is incorrect 
                plot_per_importance_analysis(
                    tuned_name, df_scene[['ade_diff', 'fde_diff']], n_test, scene_id, depth, 
                    ade_oodg_mean, fde_oodg_mean, ade_oodg_std, fde_oodg_std, 
                    ade_tuned_mean, fde_tuned_mean, ade_tuned_std, fde_tuned_std, 
                    out_dir+'/scenes', plot_err_bar=False
                )
                plot_per_importance_analysis(
                    tuned_name, 
                    df_scene[['ade_diff', 'fde_diff', 'ade_diff_std', 'fde_diff_std']], 
                    n_test, scene_id, depth, 
                    ade_oodg_mean, fde_oodg_mean, ade_oodg_std, fde_oodg_std, 
                    ade_tuned_mean, fde_tuned_mean, ade_tuned_std, fde_tuned_std, 
                    out_dir+'/scenes', plot_err_bar=True
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", default='csv/dataset_filter/dataset_ped_biker/gap/3.25_3.75/3.25_3.75', type=str)
    parser.add_argument("--out_dir", default='figures', type=str)
    args = parser.parse_args()
    create_few_shot_plot(args.results_dir, args.out_dir)
