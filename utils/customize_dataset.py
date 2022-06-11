from utils.dataset import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--additional_data_dir', default='data/sdd/raw', type=str, 
        help='Path to the scene images and variation factor file')
    parser.add_argument('--raw_data_dir', default=None, type=str, 
        help='Path to the raw data, can be a subset of the entire dataset')
    parser.add_argument('--raw_data_filename', default=None, type=str)
    parser.add_argument('--filter_data_dir', default=None, type=str)

    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--statistic_only', action='store_true', 
        help='By default, show the statistics and save the customized dataset. ' + \
            'Set False to show only the statistics of data split.')
    
    parser.add_argument("--step", default=12, type=int)
    parser.add_argument("--window_size", default=20, type=int)
    parser.add_argument("--stride", default=20, type=int)
    parser.add_argument("--obs_len", default=8, type=int)

    parser.add_argument("--varf", default=['avg_vel'], nargs='+',
                        help="Variation factors from: 'avg_vel', 'max_vel', "+\
                            "'avg_acc', 'max_acc', 'abs+max_acc', 'abs+avg_acc', "+\
                            "'min_dist', 'avg_den50', 'avg_den100', 'agent_type'")
    parser.add_argument("--varf_ranges", help='range of varation factor to take', 
                        default=[(0.5, 3.5), (4, 8)])

    parser.add_argument("--labels", default=['Pedestrian', 'Biker'], nargs='+', type=str,
                        choices=['Biker', 'Bus', 'Car', 'Cart', 'Pedestrian', 'Skater'])
    parser.add_argument('--selected_scenes', default=None, type=str, nargs='+')

    parser.add_argument("--viz", action='store_true') 

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
        df = pd.read_pickle(os.path.join(args.raw_data_dir, args.raw_data_filename))
        print('Reloaded raw dataset')


    # ================= plot =================
    if args.viz:
        varf_list = ['avg_vel', 'max_vel', 'avg_acc', 'max_acc', 
                    'abs+max_acc', 'abs+avg_acc', 'min_dist', 'avg_den100', 'avg_den50']

        if not args.reload:
            # ## get variation factor table 
            df_varfs = get_varf_table(df, varf_list, args.obs_len)
            df_varfs_com = get_varf_table(df, varf_list, None)
            df_varfs = df_varfs.merge(
                df_varfs_com.drop(['label', 'sceneId', 'scene'], axis=1), 
                on='metaId', suffixes=('', '_com'))
            out_path = os.path.join(args.additional_data_dir, "df_varfs.pkl")
            df_varfs.to_pickle(out_path)
            print(f'Saved variation factor data to {out_path}')
        else:
            # ## or load from stored one
            df_varfs = pd.read_pickle(os.path.join(args.additional_data_dir, "df_varfs.pkl"))
            print('Loaded variation factor data')

        for varf in varf_list:
            # plot_varf_hist_obs_and_complete(df_varfs[['label', varf, varf+'_com']], 'figures/filtered_distr/hist/diff')
            plot_varf_histograms(df_varfs[['label', varf]], 'figures/filtered_distr/hist/obs')
            plot_scene_w_numeric(df_varfs, varf, 'Bivar', 'figures/filtered_distr/bivar')

        for label in ['Pedestrian', 'Biker', 'Mixed', 'All']:
            plot_jointplot(df_varfs, varf_list,  label, 'Joint', 'figures/bivar_distr/filter', 'scene', kind='kde')
        plot_jointplot(df_varfs, varf_list, 'All', 'Joint', 'figures/bivar_distr/filter', 'label', kind='kde')


    # ============== create customized dataset ================
    if args.varf == ['agent_type']:
        out_dir = os.path.join(args.filter_data_dir, args.varf[0])
        create_dataset_by_agent_type(df, args.labels, out_dir, 
            statistic_only=args.statistic_only, selected_scenes=args.selected_scenes)
    else:
        out_dir = os.path.join(args.filter_data_dir, '__'.join(args.varf), '_'.join(args.labels))
        create_dataset_given_range(df, args.varf, args.varf_ranges, args.labels, 
            out_dir, obs_len=args.obs_len, statistic_only=args.statistic_only)
    print(f'Created dataset: \nVariation factor = {args.varf} \nAgents = {args.labels}')