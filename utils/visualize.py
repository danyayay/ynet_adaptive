import os
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
                    fig, axes = plt.subplots(len(diff_i), n_channel, 
                        figsize=(n_channel*3, len(diff_i)*2))
                    for k, ckpt_name in enumerate(diff_i):
                        for c in range(n_channel):
                            axes[k, c].imshow(diff_i[ckpt_name][c])
                            axes[k, c].set_xlabel(f'channel_{c+1}')
                            if c == 0: axes[k, c].set_ylabel(labels_ckpt[ckpt_name])
                    title = f'meta_id={meta_id}, scene_id={scene_id}, feature_name={feature_name}'
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
                    fig, axes = plt.subplots(n_ckpt, n_channel, 
                        figsize=(n_channel*4+2, n_ckpt*3+2))
                    for k, (ckpt_name, dict_ckpt) in enumerate(dict_features.items()):
                        feature_i = dict_ckpt[scene_id][feature_name][i]  # (n_channel, height, width)
                        for c in range(n_channel):
                            axes[k, c].imshow(feature_i[c])
                            axes[k, c].set_xlabel(f'channel_{c+1}')
                            if c == 0: axes[k, c].set_ylabel(labels_ckpt[ckpt_name])
                    title = f'meta_id={meta_id}, scene_id={scene_id}, feature_name={feature_name}'
                    axes[0, n_channel//2].set_title(title)
                    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                    out_name = f'{meta_id}__{scene_id}__{feature_name}'
                    out_path = os.path.join(out_dir, out_name + '.' + format)
                    plt.savefig(out_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved {out_path}')


def plot_feature_space_diff_evolution(
    dict_features, out_dir='figures/feature_space_diff', is_avg=True, format='png'):
    diff = {}
    df_dict = {}
    for ckpt_name in ['FT', 'ET']:
        if ('OODG' in dict_features.keys()) & (ckpt_name in dict_features.keys()):
            name = f'diff_OODG_{ckpt_name}'
            diff[name] = {}
            for s, (scene_id, dict_scene) in enumerate(dict_features[list(dict_features)[0]].items()):
                features_name = list(dict_scene)
                features_name.remove('metaId')
                if s == 0:
                    for feature_name in features_name:
                        diff[name][feature_name] = []
                for feature_name in features_name:
                    if is_avg:
                        diff[name][feature_name].extend((
                            dict_features['OODG'][scene_id][feature_name] - 
                            dict_features[ckpt_name][scene_id][feature_name]).sum(axis=(1,2,3)) / (
                                dict_features['OODG'][scene_id][feature_name][0].reshape(-1).shape[0]))
                    else:
                        diff[name][feature_name].extend((
                            dict_features['OODG'][scene_id][feature_name] - 
                            dict_features['FT'][scene_id][feature_name]).sum(axis=(1,2,3)))
            # average over samples
            df = pd.DataFrame()
            n_data = len(diff[name][features_name[0]])
            for feature_name in features_name:
                df.loc[feature_name, name] = np.mean(diff[name][feature_name])
                df.loc[feature_name, name+'_std'] = np.std(diff[name][feature_name])
            df_dict[name] = df

            # plot evolution
            if df.shape[0] == 3:
                fig_size, depth = 4, 0
            elif df.shape[0] == 20: # 6+7+7
                fig_size, depth = 10, 1
            elif df.shape[0] == 83: # 23+30+30
                fig_size, depth = 20, 2
            else: 
                fig_size, depth = df.shape[0]*0.25, 'unknown'
            fig, ax = plt.subplots(figsize=(fig_size, 4))
            plt.plot(df.index, df[name].values)
            plt.fill_between(df.index, df[name].values-df[name+'_std'].values, 
                df[name].values+df[name+'_std'].values, alpha=0.2)
            plt.title(f'Feature space difference between ODDG and {ckpt_name}')
            plt.ylabel('Difference')
            plt.xlabel('Layers')
            if (depth == 0) | (depth == 1):
                plt.xticks(rotation=45, ha='right')
            else: # (depth == 2) | (depth == unknown)
                plt.xticks(rotation=90, ha='right')
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_name = f'{name}_D{depth}_N{n_data}_avg' if is_avg else f'{name}_D{depth}_N{n_data}'
            out_path = os.path.join(out_dir, out_name + '.' + format)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out_path}')
    
    fig, ax = plt.subplots(figsize=(fig_size, 4))
    for key in df_dict.keys():
        df = df_dict[key]
        plt.plot(df.index, df[key].values, label=key)
        plt.fill_between(df.index, df[key].values-df[key+'_std'].values, 
            df[key].values+df[key+'_std'].values, alpha=0.2)
    plt.title(f'Feature space difference')
    plt.ylabel('Difference')
    plt.xlabel('Layers')
    plt.legend(loc='best')
    if (depth == 0) | (depth == 1):
        plt.xticks(rotation=45, ha='right')
    else: # (depth == 2) | (depth == unknown)
        plt.xticks(rotation=90, ha='right')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = f'diff_OODG_FT_ET_D{depth}_N{n_data}_avg' if is_avg else f'diff_OODG_FT_ET_D{depth}_N{n_data}'
    out_path = os.path.join(out_dir, out_name + '.' + format)
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

        fig = plt.figure(figsize=(scene_images[scene_id].shape[0]/100, scene_images[scene_id].shape[1]/100))
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
        fig = plt.figure(figsize=(10, 10))
        scene_id = first_dict['sceneId'][i]
        scene_image = scene_images[scene_id]
        plt.imshow(scene_image)
        j = 0
        ms = 3
        for ckpt_name, value in dict_trajs.items():
            gt_traj = value['groundtruth'][i]
            pred_traj = value['prediction'][i]
            if j == 0:
                plt.scatter(gt_traj[:obs_len][:,0], gt_traj[:obs_len][:,1], s=ms, c=colors['OB'])
                plt.scatter(gt_traj[obs_len:][:,0], gt_traj[obs_len:][:,1], s=ms, c=colors['GT'])
                plt.plot(gt_traj[:obs_len][:,0], gt_traj[:obs_len][:,1], ms=ms, c=colors['OB'], label='observed')
                plt.plot(gt_traj[obs_len:][:,0], gt_traj[obs_len:][:,1], ms=ms, c=colors['GT'], label=labels_ckpt['GT'])
                j += 1
            plt.scatter(pred_traj[:,0], pred_traj[:,1], s=ms, c=colors[ckpt_name])
            plt.plot(pred_traj[:,0], pred_traj[:,1], ms=ms, c=colors[ckpt_name], label=labels_ckpt[ckpt_name])
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
            # for each sample, visualize overlayed feature space
            for _, feature_name in enumerate([goal_dec_name, traj_dec_name]):
                n_channel = dict_scene[feature_name].shape[1]
                n_ckpt = len(dict_features)
                fig, axes = plt.subplots(n_ckpt, n_channel, figsize=(n_channel*2, n_ckpt*3))
                for k, (ckpt_name, dict_ckpt) in enumerate(dict_features.items()):
                    # TODO: traj size and scene size  
                    feature_i = dict_ckpt[scene_id][feature_name][i] / resize_factor  # (n_channel, height, width)
                    for c in range(n_channel):
                        axes[k, c].imshow(scene_images[scene_id])
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", default='csv/dataset_filter/dataset_ped_biker/gap/3.25_3.75/3.25_3.75', type=str)
    parser.add_argument("--out_dir", default='figures', type=str)
    args = parser.parse_args()
    create_few_shot_plot(args.results_dir, args.out_dir)
