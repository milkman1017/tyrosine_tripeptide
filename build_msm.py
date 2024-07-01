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

def plot_detailed_vamp2(data):
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
    plt.savefig('VAMP2_scores_for_dimensions')

def score_cv(data, dim, lag=1, number_of_splits=10, validation_fraction=0.5):
    nval = int(len(data) * validation_fraction)
    scores = np.zeros(number_of_splits)
    for n in range(number_of_splits):
        ival = np.random.choice(len(data), size=nval, replace=False)
        vamp = pyemma.coordinates.vamp(
            np.array([d for i, d in enumerate(data) if i not in ival]), lag=lag, dim=dim)
        scores[n] = vamp.score(np.array([d for i, d in enumerate(data) if i in ival]))

    return scores

def plot_vamp2_scores(data, dim, lags=[5, 10, 20]):
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
    plt.savefig('VAMP2_Scores_for_Features')

    max_value = max(avg_scores)
    highest_vamp_index = [index for index, value in enumerate(avg_scores) if value == max_value][0]
    
    data_to_MSM = data[feature_keys[highest_vamp_index]]
    plot_detailed_vamp2(data_to_MSM)

    return data_to_MSM

def TICA(data):
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
        plt.savefig('IC_coverage')
    except:
        print('Too many dims')

    fig, axes = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
    x = 0.1 * np.arange(tica_output[0].shape[0])
    for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
        ax.plot(x, tic)
        ax.set_ylabel('IC {}'.format(i + 1))
    axes[-1].set_xlabel('time / ns')
    fig.tight_layout()
    plt.savefig('IC_time_series')

    return tica_output

def determine_cluster_number(data):
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
    plt.savefig('clusters')

    differences = np.abs(np.diff(np.mean(scores, axis=1)))

    # Find the index where the change falls below the threshold
    cluster_index = np.argmax(differences < 1.0)
    print(n_clustercenters[cluster_index])
    return n_clustercenters[cluster_index]

def cluster_it(data, n_centers):
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
    plt.savefig('projected_clusters')

    return cluster

def plot_implied_timescales(cluster):
    its = pyemma.msm.its(cluster.dtrajs, lags=50, nits=10, errors='bayes')
    pyemma.plots.plot_implied_timescales(its, units='ns', dt=0.3)
    plt.savefig('impled_timescales')

def make_msm(cluster):
    msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=10, dt_traj='0.1 ns')
    print('fraction of states used = {:.2f}'.format(msm.active_state_fraction))
    print('fraction of counts used = {:.2f}'.format(msm.active_count_fraction))

    return msm

def ck_test(msm):
    nstates = 10
    cktest = msm.cktest(nstates, mlags=6)
    print(cktest)
    pyemma.plots.plot_cktest(cktest, dt=0.3, units='ns')
    plt.savefig('ck_test.png')

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
    data_to_MSM = plot_vamp2_scores(data_to_score, 10)

    print('TICA it up!')
    reduced_data = TICA(data_to_MSM)

    print('Determining cluster number')
    cluster_k = determine_cluster_number(reduced_data)

    clustered_data = cluster_it(reduced_data, cluster_k)

    plot_implied_timescales(clustered_data)
    msm = make_msm(clustered_data)
    ck_test(msm)

if __name__ == "__main__":
    main()
