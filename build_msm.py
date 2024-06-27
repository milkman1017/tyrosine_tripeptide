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
    back_torsions_data = pyemma.coordinates.load(traj_file, features=back_torsions_feat).tolist()
    labels = ['backbone\ntorsions']

    positions_feat = pyemma.coordinates.featurizer(top_file)
    positions_feat.add_all()
    positions_data = pyemma.coordinates.load(traj_file, features=positions_feat).tolist()
    labels += ['positions']

    distances_feat = pyemma.coordinates.featurizer(top_file)
    distances_feat.add_distances(
        distances_feat.pairs(distances_feat.select_Backbone()))
    distances_data = pyemma.coordinates.load(traj_file, features=distances_feat).tolist()
    labels += ['backbone\ndistances']

    side_torsions_feat = pyemma.coordinates.featurizer(top_file)
    side_torsions_feat.add_sidechain_torsions(cossin=True)
    side_torsions_data = pyemma.coordinates.load(traj_file, features=side_torsions_feat).tolist()
    labels += ['side\ntorsions']

    back_torsions_feat.add_sidechain_torsions(cossin=True)
    all_torsion_data = pyemma.coordinates.load(traj_file, features=back_torsions_feat).tolist()
    labels += ['all\ntorsions']

    return back_torsions_data, positions_data, distances_data, side_torsions_data, all_torsion_data, labels

def score_cv(data, dim, lag=10, number_of_splits=10, validation_fraction=0.5):
    print(data)
    nval = int(len(data) * validation_fraction)
    scores = np.zeros(number_of_splits)
    for n in range(number_of_splits):
        ival = np.random.choice(len(data), size=nval, replace=False)
        vamp = pyemma.coordinates.vamp(
            [d for i, d in enumerate(data) if i not in ival], lag=lag, dim=dim)
        scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])

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
    plt.show()

def main():
    args = get_args()

    data = dict()

    data['backbone_torsions'] = []
    data['positions'] = []
    data['distances'] = []
    data['sidechain_torsions'] = []
    data['all_torsions'] = []

    for sim_num in range(args.nsims):
        print(f'Simulation {sim_num}:')
        traj_file = f'{args.traj_dir}/{args.sim_name}{sim_num}.dcd'
        top_file = f'{args.traj_dir}/{args.sim_name}{sim_num}.pdb'
        
        back_torsions_data, positions_data, distances_data, side_torsions_data, all_torsions_data, labels = get_features(traj_file, top_file)
        
        data['backbone_torsions'].extend(back_torsions_data)    
        data['positions'].extend(positions_data)
        data['distances'].extend(distances_data)
        data['sidechain_torsions'].extend(side_torsions_data)
        data['all_torsions'].extend(all_torsions_data)

    # Convert lists to arrays
    for key in data.keys():
        data[key] = np.array(data[key])

    print(np.shape(data['distances']))

    plot_vamp2_scores(data, 10)

if __name__ == "__main__":
    main()
