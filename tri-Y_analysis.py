import mdtraj as md 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def load_sim(filename):
    sim = md.load_dcd(f'{filename}.dcd', top=f'{filename}.pdb')

    return sim

def compute_rmsd(sims):
    rmsds = []
    for sim in sims:
        rmsds.append(md.rmsd(sim, sim[0]))

    for rmsd in rmsds:
        plt.plot(rmsd)

    plt.savefig('rmsd.png')

def main(nsims, file_name):
    sims = []
    for i in range(nsims):
        sims.append(load_sim(f'{file_name}{i}'))

    compute_rmsd(sims)

if __name__ == "__main__":
    nsims = 2
    file_name = 'test_sim'

    main(nsims, file_name)