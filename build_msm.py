import os
import numpy as np
import mdtraj as md
import pyemma
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Build an MSM from multiple simulations.')
    parser.add_argument('--traj_dir', type=str, help='Directory containing trajectory files')
    parser.add_argument('--sim_name', type=str, help='Name of the simulation')
    parser.add_argument('--nsims', type=int, help='Number of sims to analyze')
    
    args = parser.parse_args()

    return args

def get_features(traj_file, top_file):
    torsions_feat = pyemma.coordinates.featurizer(top_file)
    torsions_feat.add_backbone_torsions(cossin=True, periodic=True)
    torsions_data = pyemma.coordinates.load(traj_file, features=torsions_feat)
    labels = ['backbone\ntorsions']

    positions_feat = pyemma.coordinates.featurizer(top_file)
    positions_feat.add_all()
    positions_data = pyemma.coordinates.load(traj_file, features=positions_feat)
    labels += ['positions']

    distances_feat = pyemma.coordinates.featurizer(top_file)
    distances_feat.add_distances(
        distances_feat.pairs(distances_feat.select_Backbone()))
    distances_data = pyemma.coordinates.load(traj_file, features=distances_feat)
    labels += ['backbone\ndistances']
    
    return torsions_data, positions_data, distances_data

def main():
    args = get_args()

    for sim_num in range(args.nsims):
        traj_file = f'{args.traj_dir}/{args.sim_name}{sim_num}.dcd'
        top_file = f'{args.traj_dir}/{args.sim_name}{sim_num}.pdb'
        
        torsions_data, positions_data, distances_data = get_features(traj_file, top_file)
        
        print(f'Simulation {sim_num}:')
        print('Torsions Data Shape:', torsions_data.shape)
        print('Positions Data Shape:', positions_data.shape)
        print('Distances Data Shape:', distances_data.shape)

if __name__ == "__main__":
    main()
