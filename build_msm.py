import os
import numpy as np
import mdtraj as md
import pyemma
from pyemma import config as pyemma_config
import argparse
import matplotlib.pyplot as plt

pyemma_config.show_progress_bars = False

def get_args():
    parser = argparse.ArgumentParser(description='Build an MSM from multiple simulations.')
    parser.add_argument('--traj_dir', type=str, help='Directory containing trajectory files')
    parser.add_argument('--sim_name', type=str, help='Name of the simulation')
    parser.add_argument('--nsims', type=int, help='Number of sims to analyze')
    
    args = parser.parse_args()

    return args

def get_features(traj_file, top_file):
    back_torsions_feat = pyemma.coordinates.featurizer(top_file)
    back_torsions_feat.add_backbone_torsions(cossin=True, periodic=True)
    back_torsions_data = pyemma.coordinates.load(traj_file, features=back_torsions_feat)
    labels = ['backbone\ntorsions']

    distances_feat = pyemma.coordinates.featurizer(top_file)
    distances_feat.add_distances(
        distances_feat.pairs(distances_feat.select_Backbone()))
    distances_data = pyemma.coordinates.load(traj_file, features=distances_feat)
    labels += ['backbone\ndistances']

    side_torsions_feat = pyemma.coordinates.featurizer(top_file)
    side_torsions_feat.add_sidechain_torsions(cossin=True, periodic=True)
    side_torsions_data = pyemma.coordinates.load(traj_file, features=side_torsions_feat)
    labels += ['side\ntorsions']

    back_torsions_feat.add_sidechain_torsions(cossin=True, periodic=True)
    all_torsion_data = pyemma.coordinates.load(traj_file, features=back_torsions_feat).tolist()
    labels += ['all\ntorsions']

    return back_torsions_data, distances_data, side_torsions_data, all_torsion_data, labels

def plot_detailed_vamp2(data, args):
    lags = [1, 2, 5, 10, 20]
    dims = [i + 1 for i in range(10)]

    fig, ax = plt.subplots()
    for i, lag in enumerate(lags):
        scores_ = np.array([score_cv(data, dim, lag)
                            for dim in dims])
        scores = np.mean(scores_, axis=1)
        errors = np.std(scores_, axis=1, ddof=1)
        color = 'C{}'.format(i)
        ax.fill_between(dims, scores - errors, scores + errors, alpha=0.3, facecolor=color)
        ax.plot(dims, scores, '--o', color=color, label='lag={:.1f}ns'.format(lag * 0.1))
        
    ax.legend()
    ax.set_xlabel('number of dimensions')
    ax.set_ylabel('VAMP2 score')
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_VAMP2_scores_for_dimensions')

def score_cv(data, dim, lag=1, number_of_splits=10, validation_fraction=0.5):
    nval = int(len(data) * validation_fraction)
    scores = np.zeros(number_of_splits)
    for n in range(number_of_splits):
        ival = np.random.choice(len(data), size=nval, replace=False)
        vamp = pyemma.coordinates.vamp(
            np.array([d for i, d in enumerate(data) if i not in ival]), lag=lag, dim=dim)
        scores[n] = vamp.score(np.array([d for i, d in enumerate(data) if i in ival]))

    return scores

def plot_vamp2_scores(data, dim, args, lags=[5, 10, 20]):
    plt.figure(figsize=(18, 6))
    feature_keys = list(data.keys())

    for i, lag in enumerate(lags):
        avg_scores = []
        for key in feature_keys:
            scores = score_cv(data[key], dim, lag=lag)
            avg_scores.append(np.mean(scores))
        
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(feature_keys)), avg_scores, tick_label=feature_keys)
        
        plt.title(f'Average VAMP2 Scores for lag={lag}')
        plt.xlabel('Feature')
        plt.ylabel('Average VAMP2 Score')
    
    plt.tight_layout()
    plt.savefig(f'{args.sim_name}_VAMP2_Scores_for_Features')

    max_value = max(avg_scores)
    highest_vamp_index = [index for index, value in enumerate(avg_scores) if value == max_value][0]
    
    data_to_MSM = data[feature_keys[highest_vamp_index]]
    plot_detailed_vamp2(data_to_MSM, args)

    return data_to_MSM

def TICA(data, args):
    tica = pyemma.coordinates.tica(data, lag=20)
    tica_output = tica.get_output()
    tica_concatenated = np.concatenate(tica_output)

    dims=(np.shape(tica_concatenated))[1]
    print(f'TICA reduced data to {dims} dimensions')

    try:
        fig, axes = plt.subplots(1, 2)
        pyemma.plots.plot_feature_histograms(
            tica_concatenated,
            ax=axes[0],
            feature_labels=[f'IC{i+1}' for i in range(dims)],
            ylog=True)
        pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], logscale=True)
        axes[1].set_xlabel('IC 1')
        axes[1].set_ylabel('IC 2')
        fig.tight_layout()
        plt.savefig(f'{args.sim_name}_IC_coverage')
    except:
        print('Too many dims')

    fig, axes = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
    x = 0.1 * np.arange(tica_output[0].shape[0])
    for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
        ax.plot(x, tic)
        ax.set_ylabel('IC {}'.format(i + 1))
    axes[-1].set_xlabel('time / ns')
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_IC_time_series')

    return tica_output

def determine_cluster_number(data, args):
    n_clustercenters = [5, 10, 30, 50, 75]

    scores = np.zeros((len(n_clustercenters), 5))
    for n, k in enumerate(n_clustercenters):
        for m in range(5):
            _cl = pyemma.coordinates.cluster_kmeans(
                data, k=k, max_iter=50, stride=50)
            _msm = pyemma.msm.estimate_markov_model(_cl.dtrajs, 5)
            scores[n, m] = _msm.score_cv(
                _cl.dtrajs, n=1, score_method='VAMP2', score_k=min(10, k))

    fig, ax = plt.subplots()
    lower, upper = pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
    ax.fill_between(n_clustercenters, lower, upper, alpha=0.3)
    ax.plot(n_clustercenters, np.mean(scores, axis=1), '-o')
    # ax.semilogx()
    ax.set_xlabel('number of cluster centers')
    ax.set_ylabel('VAMP-2 score')
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_clusters')

    differences = np.abs(np.diff(np.mean(scores, axis=1)))

    # Find the index where the change falls below the threshold
    cluster_index = np.argmax(differences < 0.5) + 1
    print(n_clustercenters[cluster_index])
    # return n_clustercenters[cluster_index]

    ###remember to uncomment out the above after finishing testing###
    return 25

def cluster_it(data, n_centers, args):
    print(n_centers)
    data=data[0]
    cluster = pyemma.coordinates.cluster_kmeans(
    data, k=n_centers, max_iter=50, stride=10)
    dtrajs_concatenated = np.concatenate(cluster.dtrajs)

    fig, ax = plt.subplots(figsize=(4, 4))
    pyemma.plots.plot_density(
        *data[:, :2].T, ax=ax, cbar=False, alpha=0.3)
    ax.scatter(*cluster.clustercenters[:, :2].T, s=5, c='C1')
    ax.set_xlabel('IC 1')
    ax.set_ylabel('IC 2')
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_projected_clusters')

    return cluster, dtrajs_concatenated

def plot_implied_timescales(cluster, args):
    its = pyemma.msm.its(cluster.dtrajs, lags=50, nits=10, errors='bayes')
    pyemma.plots.plot_implied_timescales(its, units='ns', dt=0.3)
    plt.savefig(f'{args.sim_name}_impled_timescales')

def make_msm(cluster):
    msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=10, dt_traj='0.1 ns')
    print('fraction of states used = {:.2f}'.format(msm.active_state_fraction))
    print('fraction of counts used = {:.2f}'.format(msm.active_count_fraction))

    return msm

def ck_test(msm, args):
    nstates = 10
    cktest = msm.cktest(nstates, mlags=6)
    print(cktest)
    pyemma.plots.plot_cktest(cktest, dt=0.3, units='ns')
    plt.savefig(f'{args.sim_name}_ck_test.png')

def its_separation_err(ts, ts_err):
    """
    Error propagation from ITS standard deviation to timescale separation.
    """
    return ts[:-1] / ts[1:] * np.sqrt(
        (ts_err[:-1] / ts[:-1])**2 + (ts_err[1:] / ts[1:])**2)

def plot_its_separation(msm, args):
    nits = 10

    timescales_mean = msm.sample_mean('timescales', k=nits)
    timescales_std = msm.sample_std('timescales', k=nits)
    nits = len(timescales_mean)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].errorbar(
        range(1, nits + 1),
        timescales_mean,
        yerr=timescales_std,
        fmt='.', markersize=10)
    axes[1].errorbar(
        range(1, nits),
        timescales_mean[:-1] / timescales_mean[1:],
        yerr=its_separation_err(
            timescales_mean,
            timescales_std),
        fmt='.',
        markersize=10,
        color='C0')

    for i, ax in enumerate(axes):
        ax.set_xticks(range(1, nits + 1))
        ax.grid(True, axis='x', linestyle=':')

    axes[0].axhline(msm.lag * 0.1, lw=1.5, color='k')
    axes[0].axhspan(0, msm.lag * 0.1, alpha=0.3, color='k')
    axes[0].set_xlabel('implied timescale index')
    axes[0].set_ylabel('implied timescales / ns')
    axes[1].set_xticks(range(1, nits))
    axes[1].set_xlabel('implied timescale indices')
    axes[1].set_ylabel('timescale separation')
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_timescale_separation.png')

def plot_stationary_distribution(msm, tica, dtrajs_concatenated, args):
    tica_concatenated = np.concatenate(tica)
    
    # Filter valid indices
    valid_indices = np.where(dtrajs_concatenated < len(msm.pi))[0]
    valid_dtrajs = dtrajs_concatenated[valid_indices]
    valid_tica_concatenated = tica_concatenated[valid_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    
    # Ensure shapes are compatible for contour plot
    if len(valid_tica_concatenated[:, 0]) == len(msm.pi[valid_dtrajs]):
        pyemma.plots.plot_contour(
            *valid_tica_concatenated[:, :2].T,
            msm.pi[valid_dtrajs],
            ax=axes[0],
            mask=True,
            cbar_label='stationary distribution')
    else:
        print("Mismatch in lengths for contour plot. Skipping this plot.")
    
    # Ensure shapes are compatible for free energy plot
    if len(valid_tica_concatenated[:, 0]) == len(np.concatenate(msm.trajectory_weights())[valid_indices]):
        pyemma.plots.plot_free_energy(
            *valid_tica_concatenated[:, :2].T,
            weights=np.concatenate(msm.trajectory_weights())[valid_indices],
            ax=axes[1],
            legacy=False)
    else:
        print("Mismatch in lengths for free energy plot. Skipping this plot.")
    
    for ax in axes.flat:
        ax.set_xlabel('IC 1')
    axes[0].set_ylabel('IC 2')
    axes[0].set_title('Stationary distribution', fontweight='bold')
    axes[1].set_title('Reweighted free energy surface', fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_stationary_distribution.png')

def plot_metastable_states(msm, tica, dtrajs_concatenated, nstates, args):
    import numpy as np
    import matplotlib.pyplot as plt
    import pyemma

    print('PCCA')
    nstates = msm.nstates
    pcca = msm.pcca(nstates)
    pcca_states = pcca.n_metastable

    tica_concatenated = np.concatenate(tica)
    
    valid_indices = np.where(dtrajs_concatenated < len(msm.pi))[0]
    valid_dtrajs = dtrajs_concatenated[valid_indices]

    print('Metastable assignment')
    metastable_traj = msm.metastable_assignments[valid_dtrajs]

    filtered_tica = tica_concatenated[valid_indices, :2]

    fig, ax = plt.subplots(figsize=(5, 4))
    _, _, misc = pyemma.plots.plot_state_map(
        *filtered_tica.T, metastable_traj, ax=ax)
    ax.set_xlabel('IC 1')
    ax.set_ylabel('IC 2')
    misc['cbar'].set_ticklabels([r'$\mathcal{S}_%d$' % (i + 1) for i in range(pcca_states)])
    fig.tight_layout()
    plt.savefig(f'{args.sim_name}_metastable_state_assignment.png')

def main():
    args = get_args()

    data = dict()

    data['backbone_torsions'] = []
    data['distances'] = []
    data['sidechain_torsions'] = []
    data['all_torsions'] = []

    for sim_num in range(args.nsims):
        print(f'Featurizing Simulation {sim_num}')
        traj_file = f'{args.traj_dir}/{args.sim_name}{sim_num}.dcd'
        top_file = f'{args.traj_dir}/{args.sim_name}{sim_num}.pdb'
        
        back_torsions_data, distances_data, side_torsions_data, all_torsion_data, labels = get_features(traj_file, top_file)
        
        data['backbone_torsions'].extend(back_torsions_data)    
        data['distances'].extend(distances_data)
        data['sidechain_torsions'].extend(side_torsions_data)
        data['all_torsions'].extend(all_torsion_data)

    for key in data.keys():
        data[key] = np.vstack(data[key])

    data_to_score = dict()
    data_to_score['backbone_torsions'] = data['backbone_torsions']
    data_to_score['side_chain_torsions'] = data['sidechain_torsions']
    data_to_score['all_torsions'] = data['all_torsions']

    print('VAMP it up (it up!)')
    data_to_MSM = plot_vamp2_scores(data_to_score, 10, args)

    print('TICA it up!')
    reduced_data = TICA(data_to_MSM, args)

    print('Determining cluster number')
    cluster_k = determine_cluster_number(reduced_data, args)

    clustered_data, dtrajs = cluster_it(reduced_data, cluster_k, args)

    plot_implied_timescales(clustered_data, args)

    print('Buliding MSM')
    msm = make_msm(clustered_data)

    print('CK test')
    # ck_test(msm, args)

    print('Getting implied time scales')
    # plot_its_separation(msm, args)

    print('plotting stationary distribution')
    # plot_stationary_distribution(msm, reduced_data, dtrajs, args)

    plot_metastable_states(msm, reduced_data, dtrajs, cluster_k, args)

if __name__ == "__main__":
    main()
