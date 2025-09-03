import os
import glob
import argparse
import torch
import pyemma
import pickle
import mdtraj as md
import numpy as np

from tqdm import tqdm
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm



def torch_check_n_save(
    data: torch.Tensor,
    path: str,
):
    if os.path.exists(path):
        print(f"Data already exists at {path}")
    else:
        torch.save(data, path)
        print(f"Saved data to {path}")
    return

def coord2rot(
    pdb,
    coordinates,
):
    ca_indices = pdb.topology.select('name CA')
    n_indices = pdb.topology.select('name N')
    c_indices = pdb.topology.select('name C')
    pdb_xyz = torch.tensor(coordinates)

    a = pdb_xyz[:, ca_indices]
    b = pdb_xyz[:, n_indices]
    c = pdb_xyz[:, c_indices]

    u = b - a  # C_alpha -> N
    v = c - a  # C_alpha -> C

    # Gram-Schmidt Process
    e1 = u / torch.norm(u, dim=-1, keepdim=True)
    u2 = v - torch.sum(e1 * v, dim=-1, keepdim=True) * e1
    e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)

    Q = torch.stack([e1, e2, e3], dim=-1)

    return Q


def load_data(
    molecule: str,
    simulation_idx: int,
    base_dir: str,
    pdb_path: str,
):
    pdb = md.load(pdb_path)
    dcd_pattern = f"{base_dir}/{molecule}-{simulation_idx}-protein-*.dcd"
    dcd_files = glob.glob(dcd_pattern)
    num_files = len(dcd_files)
    print(f"Found {num_files} .dcd files in {base_dir}")
    print(f"Pattern used: {dcd_pattern}")
    file_indices = []
    for file_path in dcd_files:
        filename = os.path.basename(file_path)
        index_part = filename.split('-')[-1].replace('.dcd', '')
        if index_part.isdigit():
            file_indices.append(int(index_part))
    file_indices.sort()
    print(f"File indices range: {min(file_indices)} to {max(file_indices)}")
    print(f"Total files to load: {len(file_indices)}")
    
    # Load trajectories
    traj_list = []
    for i in tqdm(
        file_indices,
        desc="Loading trajectories"
    ):
        file_idx = f"{i:03d}"
        file_path = f"{base_dir}/{molecule}-{simulation_idx}-protein-{file_idx}.dcd"
        if os.path.exists(file_path):
            traj = md.load_dcd(file_path, top=pdb_path)
            traj_list.append(traj)
        else:
            print(f"Warning: File not found: {file_path}")
    print(f"Successfully loaded {len(traj_list)} trajectory files")
    all_traj = md.join(traj_list)
    
    # Re arrange trajectory for some molecules
    if molecule == "GTT":
        atom_mapping = np.array([1, 5, 8, 9, 2, 3, 4, 6, 7, 10, 12, 19, 20, 14, 17, 11, 13, 15, 16, 18, 21, 23, 41, 42, 25, 28, 31, 34, 37, 22, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 40, 43, 45, 60, 61, 47, 50, 52, 56, 44, 46, 48, 49, 51, 53, 54, 55, 57, 58, 59, 62, 66, 74, 75, 63, 68, 71, 64, 65, 67, 69, 70, 72, 73, 76, 80, 88, 89, 77, 82, 85, 78, 79, 81, 83, 84, 86, 87, 90, 92, 95, 96, 91, 93, 94, 97, 99, 119, 120, 101, 104, 105, 107, 109, 110, 111, 113, 115, 117, 98, 100, 102, 103, 106, 108, 112, 114, 116, 118, 121, 123, 134, 135, 125, 128, 131, 132, 133, 122, 124, 126, 127, 129, 130, 136, 138, 156, 157, 140, 143, 146, 149, 152, 137, 139, 141, 142, 144, 145, 147, 148, 150, 151, 153, 154, 155, 158, 160, 180, 181, 162, 165, 168, 171, 173, 174, 177, 159, 161, 163, 164, 166, 167, 169, 170, 172, 175, 176, 178, 179, 182, 184, 197, 198, 186, 189, 192, 193, 183, 185, 187, 188, 190, 191, 194, 195, 196, 199, 201, 208, 209, 203, 206, 200, 202, 204, 205, 207, 210, 212, 232, 233, 214, 217, 220, 223, 225, 226, 229, 211, 213, 215, 216, 218, 219, 221, 222, 224, 227, 228, 230, 231, 234, 236, 244, 245, 238, 241, 242, 243, 235, 237, 239, 240, 246, 248, 251, 252, 247, 249, 250, 253, 255, 275, 276, 257, 260, 263, 266, 268, 269, 272, 254, 256, 258, 259, 261, 262, 264, 265, 267, 270, 271, 273, 274, 277, 279, 291, 292, 281, 283, 287, 278, 280, 282, 284, 285, 286, 288, 289, 290, 293, 295, 312, 313, 297, 300, 301, 303, 305, 306, 308, 310, 294, 296, 298, 299, 302, 304, 307, 309, 311, 314, 316, 333, 334, 318, 321, 322, 324, 326, 327, 329, 331, 315, 317, 319, 320, 323, 325, 328, 330, 332, 335, 337, 353, 354, 339, 342, 343, 345, 347, 349, 351, 336, 338, 340, 341, 344, 346, 348, 350, 352, 355, 357, 367, 368, 359, 362, 363, 364, 356, 358, 360, 361, 365, 366, 369, 371, 373, 376, 378, 379, 381, 382, 384, 385, 370, 372, 374, 375, 377, 380, 383, 386, 388, 403, 404, 390, 392, 396, 399, 387, 389, 391, 393, 394, 395, 397, 398, 400, 401, 402, 405, 407, 417, 418, 409, 411, 413, 406, 408, 410, 412, 414, 415, 416, 419, 421, 423, 424, 420, 422, 546, 425, 427, 429, 430, 547, 548, 549, 426, 428, 550, 551, 552, 553, 554, 431, 433, 435, 436, 555, 556, 557, 432, 434, 558, 559, 560, 561, 562, 437, 439, 452, 453, 441, 444, 447, 448, 449, 438, 440, 442, 443, 445, 446, 450, 451, 454, 456, 472, 473, 458, 461, 462, 464, 466, 468, 470, 455, 457, 459, 460, 463, 465, 467, 469, 471, 474, 476, 487, 488, 478, 481, 484, 485, 486, 475, 477, 479, 480, 482, 483, 489, 491, 511, 512, 493, 496, 499, 502, 504, 505, 508, 490, 492, 494, 495, 497, 498, 500, 501, 503, 506, 507, 509, 510, 513, 517, 525, 526, 514, 519, 522, 515, 516, 518, 520, 521, 523, 524, 527, 529, 536, 537, 531, 534, 528, 530, 532, 533, 535, 541, 543, 538, 539, 540, 542, 544, 545])
        all_traj.xyz = all_traj.xyz[:, atom_mapping - 1]
        all_traj.center_coordinates()
    
    return all_traj, pdb


def compute_descriptors(
    all_traj: md.Trajectory,
    pdb: md.Trajectory,
    molecule: str,
    simulation_idx: int,
    save_dir: str,
):
    # CA indices for plumed dat
    ca_atoms = [atom for atom in all_traj.topology.atoms if atom.name == "CA"]
    ca_indices = [atom.index for atom in ca_atoms]
    with open("CA_index.txt", "w") as f:
        cnt = 1
        for i in range(len(ca_indices)):
            for j in range(i + 1, len(ca_indices)):
                f.write(f"d{cnt}: DISTANCE ATOMS={ca_indices[i]+1},{ca_indices[j]+1}\n")
                cnt += 1
        f.write("ARG=" + ",".join(f"d{i+1}" for i in range(cnt)) + "\n")
    
    # CA pair distance
    ca_resid_pair = np.array([
        (a.index, b.index) for a, b in combinations(list(pdb.topology.residues), 2)
    ])
    ca_pair_contacts, resid_pairs = md.compute_contacts(
        all_traj, scheme="ca", contacts=ca_resid_pair, periodic=False
    )
    ca_path = f"{save_dir}/{molecule}-{simulation_idx}-cad.pt"
    if os.path.exists(ca_path):
        print(f"CA pair distance already exists at {ca_path}")
    else:
        torch.save(
            torch.from_numpy(ca_pair_contacts),
            ca_path
        )
    
    # CA pair distance switch
    exp = 2
    ca_pair_distances_swtich = (1 - (np.power(ca_pair_contacts, exp) / 0.8)) / (1 - (np.power(ca_pair_contacts, exp) / 0.8))
    ca_pair_distances_swtich_path = f"{save_dir}/{molecule}-{simulation_idx}-cad-switch.pt"
    if os.path.exists(ca_pair_distances_swtich_path):
        print(f"CA pair distance switch already exists at {ca_pair_distances_swtich_path}")
    else:
        torch.save(
            torch.from_numpy(ca_pair_distances_swtich),
            ca_pair_distances_swtich_path
        )
    
    # XYZ
    xyz = all_traj.xyz
    xyz_path = f"{save_dir}/{molecule}-{simulation_idx}-pos.pt"
    if os.path.exists(xyz_path):
        print(f"XYZ already exists at {xyz_path}")
    else:
        torch.save(
            torch.from_numpy(xyz),
            xyz_path
        )
    
    # Print stats
    print("=" * 100)
    print(f"CA pair contacts shape: {ca_pair_contacts.shape}")
    print(f"CA pair contacts mean: {ca_pair_contacts.mean()}")
    print(f"CA pair contacts switch shape: {ca_pair_distances_swtich.shape}")
    print(f"CA pair contacts switch mean: {ca_pair_distances_swtich.mean()}")
    print(f"XYZ shape: {xyz.shape}")
    print("=" * 100)
    
    return ca_pair_contacts, ca_pair_distances_swtich, xyz


def compute_tica(
    cad_data: np.ndarray,
    cad_data_switch: np.ndarray,
    molecule: str,
    data_path: str,
    switch: bool = False,
):  

    for lag in (10, 100, 1000):
        try:
            data = cad_data
            tica_model_path = f"{data_path}/{molecule}_tica_model_lag{lag}.pkl"
            if os.path.exists(tica_model_path):
                print(f"TICA model already exists at {tica_model_path}")
                with open(tica_model_path, 'rb') as f:
                    tica_obj = pickle.load(f)
            else:
                tica_obj = pyemma.coordinates.tica(data, lag=lag, dim=2)
                tica_data = tica_obj.get_output()[0]
                x = tica_data[:, 0]
                y = tica_data[:, 1]

                # Plot
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111)
                ax.hist2d(x, y, bins=100, norm=LogNorm())
                # ax.scatter(pdb_tica_x, pdb_tica_y, color="red", s=100)
                ax.set_xlabel("TIC 1")
                ax.set_ylabel("TIC 2")
                plt.title(f"TICA with pair distances, lag={lag}")
                plt.savefig(f'{data_path}/{molecule}_tica_model_lag{lag}.png')
                plt.show()
                plt.close()

                with open(f'{data_path}/{molecule}_tica_model_lag{lag}.pkl', 'wb') as f:
                    pickle.dump(tica_obj, f)
        
        except Exception as e:
            print(f"Error computing TICA with lag={lag}: {e}")
            continue

    if switch:
        for lag in (10, 100, 1000):
            try:
                data = cad_data
                tica_model_path = f"{data_path}/{molecule}_tica_model_switch_lag{lag}.pkl"
                if os.path.exists(tica_model_path):
                    print(f"TICA model already exists at {tica_model_path}")
                    with open(tica_model_path, 'rb') as f:
                        tica_obj = pickle.load(f)
                else:
                    data = cad_data_switch
                    tica_obj = pyemma.coordinates.tica(data, lag=lag, dim=2)
                    tica_data = tica_obj.get_output()[0]
                    x = tica_data[:, 0]
                    y = tica_data[:, 1]
                    
                    # Plot
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.hist2d(x, y, bins=100, norm=LogNorm())
                    ax.set_xlabel("TIC 1")
                    ax.set_ylabel("TIC 2")
                    plt.title(f"TICA with pair distances switch, lag={lag}")
                    plt.savefig(f'{data_path}/{molecule}_tica_model_switch_lag{lag}.png')
                    plt.show()
                    plt.close()

                    with open(f'{data_path}/{molecule}_tica_model_switch_lag{lag}.pkl', 'wb') as f:
                        pickle.dump(tica_obj, f)
            
            except Exception as e:
                print(f"Error computing TICA with lag={lag}, switch: {e}")
                continue
    else:
        print("Switch is not enabled")
        
    return

    
def create_dataset(
    all_traj: md.Trajectory,
    pdb: md.Trajectory,
    pos_data: np.ndarray,
    cad_data: np.ndarray,
    save_dir: str,
    molecule: str,
    dataset_size: int,
    num_data_str: str,
):
    time_lag_list = [0, 10, 100, 1000]
    data_num = all_traj.n_frames
    selected_idx = torch.from_numpy(np.random.choice(data_num - max(time_lag_list) - 1, size = dataset_size, replace=False))
    print(f"Dataset size: {num_data_str}")
    
    print("Saving current data")
    current_pos = pos_data[selected_idx]
    current_cad = cad_data[selected_idx]
    orientation = coord2rot(pdb, pos_data)
    current_orientation = orientation[selected_idx]
    torch_check_n_save(torch.from_numpy(current_pos), f"{save_dir}/{molecule}-{num_data_str}/current-pos.pt")
    torch_check_n_save(torch.from_numpy(current_cad), f"{save_dir}/{molecule}-{num_data_str}/current-cad.pt")
    torch_check_n_save(current_orientation, f"{save_dir}/{molecule}-{num_data_str}/current-orientation.pt")
    
    pbar = tqdm(
        time_lag_list,
        desc="Saving time-lagged data"
    )
    for time_lag in pbar:
        pbar.set_description(f"Saving time-lagged data {time_lag}")
        next_pos = pos_data[selected_idx + time_lag]
        next_cad = cad_data[selected_idx + time_lag]
        next_orientation = orientation[selected_idx + time_lag]
        torch_check_n_save(torch.from_numpy(next_pos), f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{num_data_str}/lag{time_lag}-pos.pt")
        torch_check_n_save(torch.from_numpy(next_cad), f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{num_data_str}/lag{time_lag}-cad.pt")
        torch_check_n_save(next_orientation, f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{num_data_str}/lag{time_lag}-orientation.pt")
    
    return


def main(
    args
):
    # Arguments
    molecule = args.molecule
    simulation_idx = args.simulation_idx
    dataset_size = args.dataset_size
    num_data_str = str(dataset_size // 1000) + "k"
    
    dataset_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset"
    base_dir = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-protein"
    data_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}"
    pdb_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}.pdb"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    
    # Load data    
    all_traj, pdb = load_data(
        molecule=molecule,
        simulation_idx=simulation_idx,
        base_dir=base_dir,
        pdb_path=pdb_path,
    )
    
    
    # Compute desciptors (position, orientation, cad)
    cad_data, cad_data_switch, xyz = compute_descriptors(
        all_traj=all_traj,
        pdb=pdb,
        molecule=molecule,
        simulation_idx=simulation_idx,
        save_dir=f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein",
    )
    
    
    # Compute TICA
    compute_tica(
        cad_data=cad_data,
        cad_data_switch=cad_data_switch,
        molecule=molecule,
        data_path=data_path,
        switch=False,
    )
    
    # Create dataset and save
    pdb = md.load(pdb_path)
    create_dataset(
        all_traj=all_traj,
        pdb=pdb,
        pos_data=xyz,
        cad_data=cad_data,
        save_dir=dataset_path,
        molecule=molecule,
        dataset_size=dataset_size,
        num_data_str=num_data_str,
    )
    
    return    
    


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="1FME")
    parser.add_argument("--dataset_size", type=int, default=50000)
    parser.add_argument("--simulation_idx", type=int, default=0)
    
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    main(args)
    
    print("Done")