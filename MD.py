import sys
import random
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from simtk.openmm.app import PDBFile

def load_pdb(file_path):
    return PDBFile(file_path)

def random_rotation_matrix():
    theta = random.uniform(0, 2*np.pi)
    phi = random.uniform(0, 2*np.pi)
    z = random.uniform(0, 1)

    r = np.sqrt(1 - z**2)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = z

    rotation_matrix = np.array([
        [np.cos(theta) + (1 - np.cos(theta)) * x * x, (1 - np.cos(theta)) * x * y - np.sin(theta) * z, (1 - np.cos(theta)) * x * z + np.sin(theta) * y],
        [(1 - np.cos(theta)) * y * x + np.sin(theta) * z, np.cos(theta) + (1 - np.cos(theta)) * y * y, (1 - np.cos(theta)) * y * z - np.sin(theta) * x],
        [(1 - np.cos(theta)) * z * x - np.sin(theta) * y, (1 - np.cos(theta)) * z * y + np.sin(theta) * x, np.cos(theta) + (1 - np.cos(theta)) * z * z]
    ])
    return rotation_matrix

def apply_transformation(positions, translation, rotation_matrix):
    rotated_positions = np.dot(positions, rotation_matrix.T)
    translated_positions = rotated_positions + translation
    return translated_positions

def generate_copies(pdb, model, n_copies):
    original_positions = np.array(pdb.getPositions().value_in_unit(nanometers))
    print(len(original_positions))

    for _ in range(n_copies):
        translation = np.array([random.uniform(-2.5, 2.5) for _ in range(3)])
        rotation_matrix = random_rotation_matrix()
        new_positions = apply_transformation(original_positions, translation, rotation_matrix)*nanometer

        model.add(pdb.topology, new_positions)

def main(pdb_file, n_copies, output_prefix):
    pdb = load_pdb(pdb_file)
    n_copies = int(n_copies)

    # Set up the system
    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    modeller = Modeller(pdb.topology, pdb.positions)

    for i in range(n_copies-1):
        print("making copies")
        generate_copies(pdb, modeller, n_copies)

    # Add solvent
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometers)

    # Create the system
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometers, constraints=HBonds)
    
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}

    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    # Save the initial frame
    PDBFile.writeFile(modeller.topology, modeller.positions, open(f"{output_prefix}.pdb", 'w'))

    # Minimize the energy
    simulation.minimizeEnergy()

    # Run the simulation
    simulation.reporters.append(DCDReporter(f"{output_prefix}.dcd", 1000))
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))

    simulation.step(10000)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <pdb_file> <n_copies> <output_prefix>")
    else:
        pdb_file = sys.argv[1]
        n_copies = int(sys.argv[2])
        output_prefix = sys.argv[3]
        main(pdb_file, n_copies, output_prefix)
