import sys
import random
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from simtk.openmm.app import PDBFile
import threading

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

def is_overlapping(new_positions, existing_positions, threshold=1.5):
    for new_pos in new_positions:
        for existing_pos in existing_positions:
            distance = np.linalg.norm(new_pos - existing_pos)
            if distance < threshold:
                return True
    return False

def generate_copies(pdb, model, n_copies):
    original_positions = np.array(pdb.getPositions().value_in_unit(nanometers))
    existing_positions = [original_positions]

    for i in range(n_copies - 1):  # -1 because original is already there
        attempt = 0
        while True:
            attempt += 1
            translation = np.array([random.uniform(-2.5, 2.5) for _ in range(3)])  # Adjusted range for better separation
            rotation_matrix = random_rotation_matrix()
            new_positions = apply_transformation(original_positions, translation, rotation_matrix)

            if not is_overlapping(new_positions, existing_positions):
                print(f'Copy {i+1} placed successfully after {attempt} attempts')
                break

            if attempt > 50:  # Safety to avoid infinite loops
                print(f'Failed to place copy {i+1} after 100 attempts, skipping')
                break

        existing_positions.append(new_positions)
        model.add(pdb.topology, new_positions * nanometers)

def run_simulation(pdb_file, n_copies, output_prefix, sim_id):
    pdb = load_pdb(pdb_file)
    n_copies = int(n_copies)

    # Set up the system
    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    modeller = Modeller(pdb.topology, pdb.positions)

    generate_copies(pdb, modeller, n_copies)

    # Add solvent
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometers)

    # Create the system
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometers, constraints=HBonds)
    
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.003*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}

    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    # Save the initial frame
    PDBFile.writeFile(modeller.topology, modeller.positions, open(f"{output_prefix}_sim{sim_id}.pdb", 'w'))
    print(f'Initial PDB for simulation {sim_id} saved')

    # Minimize the energy
    simulation.minimizeEnergy()

    # Run the simulation
    simulation.reporters.append(DCDReporter(f"{output_prefix}_sim{sim_id}.dcd", 1000))
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))

    simulation.step(5000000)

def main(pdb_file, n_copies, nsims, output_prefix):
    threads = []
    for sim_id in range(nsims):
        thread = threading.Thread(target=run_simulation, args=(pdb_file, n_copies, output_prefix, sim_id))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <pdb_file> <n_copies> <nsims> <output_prefix>")
    else:
        pdb_file = sys.argv[1]
        n_copies = int(sys.argv[2])
        nsims = int(sys.argv[3])
        output_prefix = sys.argv[4]
        main(pdb_file, n_copies, nsims, output_prefix)
