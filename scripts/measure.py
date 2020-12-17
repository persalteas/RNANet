#!/usr/bin/python3.8

# usage : pass as an argument a folder containing .cif files of RNA chains, like those produced by RNANet:
# usage : ./measure_bonds_and_angles.py ~/Data/RNA/3D/rna_only
# OR
# usage : ./measure_bonds_and_angles.py ~/Data/RNA/3D/rna_mapped_to_Rfam

from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import Vector, calc_angle
from sys import argv
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import os, signal

def measure_in_chain(f):
    mmcif_parser = MMCIFParser()
    s = mmcif_parser.get_structure('null', os.path.abspath(path_to_3D_data + f))
    chain = next(s[0].get_chains()) # Assume only one chain per .cif file

    c_to_p = []
    p_to_c = []
    c_p_c = []
    p_c_p = []
    last_p = None
    last_c = None
    nres = 0
    for res in chain:
        nres += 1

        # Get the new c1' and p atoms
        atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
        atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
        if len(atom_c1p) + len(atom_p) != 2:
            last_c = None 
            last_p = None
            continue
        atom_c1p = Vector(atom_c1p[0])
        atom_p = Vector(atom_p[0])

        if last_c is not None: # There was a previous residue
            # Get the C1'(i-1) -> P distance
            c_to_p.append((last_c - atom_p).norm()) # the C1'(i-1) -> P bond of the theta angle

            # Get the C1'(i-1)-P(i)-C1'(i) flat angle
            c_p_c.append(calc_angle(last_c, atom_p, atom_c1p))
            
            # Get the P(i-1)-C1'(i-1)-P(i) flat angle
            p_c_p.append(calc_angle(last_p, last_c, atom_p))
        
        p_to_c.append((atom_c1p - atom_p).norm()) # the P -> C1' bond of the eta angle
        last_c = atom_c1p
        last_p = atom_p

    c_to_p = np.array(c_to_p, dtype=np.float16)
    p_to_c = np.array(p_to_c, dtype=np.float16)
    c_p_c = np.array(c_p_c, dtype=np.float16)
    p_c_p = np.array(p_c_p, dtype=np.float16)
    c_to_p = c_to_p[~np.isnan(c_to_p)]
    p_to_c = p_to_c[~np.isnan(p_to_c)]
    c_p_c = c_p_c[~np.isnan(c_p_c)]
    p_c_p = p_c_p[~np.isnan(p_c_p)]
    
    return (c_to_p, p_to_c, c_p_c, p_c_p)

def init_worker(tqdm_lock=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)

def measure_all_dist_angles():
    path_to_3D_data = argv[1]

    if path_to_3D_data[-1] != '/':
        path_to_3D_data += '/'

    r_cp = np.array([], dtype=np.float16)
    r_pc = np.array([], dtype=np.float16)
    flat_angles_cpc = np.array([], dtype=np.float16)
    flat_angles_pcp = np.array([], dtype=np.float16)
    p = mp.Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=os.cpu_count())
    pbar = tqdm(total=len(os.listdir(path_to_3D_data)), desc="Scanning RNA chains", position=0, leave=True)
    try:
        nchains = 0
        for _, r in enumerate(p.imap_unordered(measure_in_chain, os.listdir(path_to_3D_data))):
            pbar.update(1)
            nchains += 1
            r_cp = np.hstack([r_cp, r[0]])
            r_pc = np.hstack([r_pc, r[1]])
            flat_angles_cpc = np.hstack([flat_angles_cpc, r[2]])
            flat_angles_pcp = np.hstack([flat_angles_pcp, r[3]])
        p.close()
        p.join()
        pbar.close()
        np.savez("measures.npz", c_p=r_cp, p_c=r_pc, c_p_c=flat_angles_pcp, p_c_p=flat_angles_pcp)
    except KeyboardInterrupt:
        print("Caught Ctrl-C, quitting")
        p.terminate()
        p.join()
        pbar.close() 
    except Exception as e:
        print(e)
        p.terminate()
        p.join()
        pbar.close()
        np.savez("measures_incomplete.npz", c_p=r_cp, p_c=r_pc, c_p_c=flat_angles_pcp, p_c_p=flat_angles_pcp)


if __name__ == "__main__":
    # Do the computations and save/reload the data
    
    # measure_all_dist_angles()

    d = np.load("measures.npz")
    c_p = d["c_p"]
    p_c = d["p_c"]
    p_c_p = d["p_c_p"]
    c_p_c = d["c_p_c"]

    # Plot stuff
    plt.figure(figsize=(6,4), dpi=300)
    plt.hist(c_p)
    plt.savefig("lengths.png")
    plt.close()

    # print(f"Final values: P->C1' is \033[32m{avg[0]/10:.3f} ± {avg[1]/10:.3f} nm\033[0m, "
    #     f"C1'->P is \033[32m{avg[2]/10:.3f} ± {avg[3]/10:.3f} nm\033[0m, "
    #     f"angles C-P-C \033[32m{avg[4]:.2f} ± {avg[5]:.2f}\033[0m and P-C-P \033[32m{avg[6]:.2f} ± {avg[7]:.2f}\033[0m")
