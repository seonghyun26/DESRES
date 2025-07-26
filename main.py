import os
import wandb
import numpy as np
import mdtraj as md
import hydra
import pyemma
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from itertools import combinations
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm


# Import trajectory data
def load_data(
    cfg : DictConfig,
) -> None:
    molecule_id = cfg.id
    simulation_id = cfg.simulation_id
    
    # Load trajectory data
    print(f"Loading trajectory data...")
    pdb_path = f"./DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}.pdb"
    traj_path = f"./DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}-{simulation_id}-protein/"
    traj_path = Path(traj_path)
    dcd_files = sorted(traj_path.glob("*.dcd"))
    traj_list = []
    for dcd_file in tqdm(
        dcd_files,
        desc="Loading trajectory data",
    ):
        traj = md.load_dcd(dcd_file, top=pdb_path)
        traj_list.append(traj)
    
    traj_all = md.join(traj_list)
    
    return traj_all
        

# Featurization and save
def featurize(
    cfg: DictConfig,
    traj_data: md.Trajectory,
) -> None:
    molecule_id = cfg.id
    simulation_id = cfg.simulation_id
    feature_type = cfg.feature
    lag = cfg.lag
    tica_dim = cfg.tica_dim
    pdb_path = f"./DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}.pdb"
    tica_save_path = f"./DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}_tica.pkl"
    
    
    if os.path.exists(tica_save_path):
        print(f"Loading TICA data from {tica_save_path}...")
        tica_data = np.load(tica_save_path.replace(".pkl", ".npy"))

    else:
        print(f"Featurizing {feature_type}...")
        state_traj = md.load(pdb_path)
        ca_atoms = state_traj.topology.select("name CA")
        atom_pairs = [list(pair) for pair in combinations(ca_atoms, 2)]        
        featurizer = pyemma.coordinates.featurizer(pdb_path)
        if 'alpha carbon distance' in feature_type:
            featurizer.add_distances(atom_pairs)
            print(f"Added {len(atom_pairs)} distance features")
        if "dihedral angle" in feature_type:
            phi_idx, phi_value = md.compute_phi(state_traj)
            psi_idx, psi_value = md.compute_psi(state_traj)
            featurizer.add_dihedrals(phi_idx, cossin=True)
            featurizer.add_dihedrals(psi_idx, cossin=True)
            print(f"Added {len(phi_idx)} phi features")
            print(f"Added {len(psi_idx)} psi features")
        
        features = featurizer.transform(traj_data)
        tica_model = pyemma.coordinates.tica(features, lag=lag, dim=tica_dim)
        tica_data = tica_model.get_output()[0]
        print(f"TICA data shape: {tica_data.shape}")
        np.save(tica_save_path.replace(".pkl", ".npy"), tica_data)
        
        with open(tica_save_path, 'wb') as f:
            pickle.dump(tica_model, f)
    
    return tica_data



# Draw TICA plot
def plot(
    cfg: DictConfig,
    tica_data: np.ndarray,
) -> None:
    molecule_id = cfg.id
    simulation_id = cfg.simulation_id
    save_path = f"./DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}_tica.png"
    
    print(f"Drawing TICA plot...")
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.hist2d(
        tica_data[:, 0],
        tica_data[:, 1],
        bins=200,
        norm=LogNorm(),
        cmap='gist_rainbow'
    )
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    # ax.invert_yaxis()
    plt.show()
    plt.savefig(save_path)
    plt.close()
    
    wandb.log({
        "tica": wandb.Image(save_path),
    })
    
    
    return 



@hydra.main(
  version_base=None,
  config_path="config",
  config_name="cln025",
)
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        project="DESRES",
        entity="eddy26",
        config=OmegaConf.to_container(cfg)
    )

    traj_data = load_data(cfg)
    tica_feat = featurize(cfg, traj_data)
    plot(cfg, tica_feat)
    
    wandb.finish()
    

if __name__ == "__main__":
    main()