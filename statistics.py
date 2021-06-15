#!/usr/bin/python3

# This file computes additional statistics over the produced dataset.
# Run this file if you want the base counts, pair-type counts, identity percents, etc
# in the database.

import getopt, os, pickle, sqlite3, shlex, subprocess, sys, warnings
import numpy as np
import pandas as pd
import threading as th
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as sch
import sklearn
import json
import pickle
import Bio
from scipy.spatial.distance import squareform
from mpl_toolkits.mplot3d import axes3d
from Bio import AlignIO, SeqIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.vectors import Vector, calc_angle, calc_dihedral
from functools import partial
from multiprocessing import Pool, Manager
from os import path
from tqdm import tqdm
from collections import Counter
from setproctitle import setproctitle
from RNAnet import Job, read_cpu_number, sql_ask_database, sql_execute, warn, notify, init_worker, trace_unhandled_exceptions
from sklearn.mixture import GaussianMixture
import warnings
from pandas.core.common import SettingWithCopyWarning


np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf, precision=8)
path_to_3D_data = "tobedefinedbyoptions"
path_to_seq_data = "tobedefinedbyoptions"
runDir = os.getcwd()
res_thr = 20.0 # default: all structures

LSU_set = ("RF00002", "RF02540", "RF02541", "RF02543", "RF02546")   # From Rfam CLAN 00112
SSU_set = ("RF00177", "RF02542",  "RF02545", "RF01959", "RF01960")  # From Rfam CLAN 00111

@trace_unhandled_exceptions
def reproduce_wadley_results(carbon=4, show=False, sd_range=(1,4), res=2.0):
    """
    Plot the joint distribution of pseudotorsion angles, in a Ramachandran-style graph.
    See Wadley & Pyle (2007).
    Only unique unmapped chains with resolution < res argument are considered.

    Arguments:
    carbon:     1 or 4, use C4' (eta and theta) or C1' (eta_prime and theta_prime)
    show:       True or False, call plt.show() at this end or not
    sd_range:   tuple, set values below avg + sd_range[0] * stdev to 0,
                    and values above avg + sd_range[1] * stdev to avg + sd_range[1] * stdev.
                    This removes noise and cuts too high peaks, to clearly see the clusters.
    res:        Minimal resolution (maximal resolution value, actually) of the structure to 
                    consider its nucleotides.
    """

    os.makedirs(runDir + "/results/figures/wadley_plots/", exist_ok=True)

    if carbon == 4:
        angle = "eta"
        xlabel = "$\\eta=C_4'^{i-1}-P^i-C_4'^i-P^{i+1}$"
        ylabel = "$\\theta=P^i-C_4'^i-P^{i+1}-C_4'^{i+1}$"
    elif carbon == 1:
        angle = "eta_prime"
        xlabel = "$\\eta'=C_1'^{i-1}-P^i-C_1'^i-P^{i+1}$"
        ylabel = "$\\theta'=P^i-C_1'^i-P^{i+1}-C_1'^{i+1}$"
    else:
        exit("You overestimate my capabilities !")

    
    if not path.isfile(runDir + f"/data/wadley_kernel_{angle}_{res}A.npz"):

        # Get a worker number to position the progress bar
        global idxQueue
        thr_idx = idxQueue.get()
        setproctitle(f"RNANet statistics.py Worker {thr_idx+1} reproduce_wadley_results(carbon={carbon})")

        pbar = tqdm(total=2, desc=f"Worker {thr_idx+1}: eta/theta C{carbon} kernels", unit="kernel", position=thr_idx+1, leave=False)

        # Extract the angle values of c2'-endo and c3'-endo nucleotides
        with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal')
            df = pd.read_sql(f"""SELECT {angle}, th{angle} 
                                 FROM (
                                    SELECT chain_id FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
                                    WHERE chain.rfam_acc = 'unmappd' AND structure.resolution <= {res} AND issue = 0
                                 ) AS c NATURAL JOIN nucleotide
                                 WHERE puckering="C2'-endo" 
                                    AND {angle} IS NOT NULL 
                                    AND th{angle} IS NOT NULL;""", conn)
            c2_endo_etas = df[angle].values.tolist()
            c2_endo_thetas = df["th"+angle].values.tolist()
            df = pd.read_sql(f"""SELECT {angle}, th{angle} 
                                 FROM (
                                    SELECT chain_id FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
                                    WHERE chain.rfam_acc = 'unmappd' AND structure.resolution <= {res} AND issue = 0
                                 ) AS c NATURAL JOIN nucleotide 
                                 WHERE form = '.' 
                                    AND puckering="C3'-endo" 
                                    AND {angle} IS NOT NULL 
                                    AND th{angle} IS NOT NULL;""", conn)
            c3_endo_etas = df[angle].values.tolist()
            c3_endo_thetas = df["th"+angle].values.tolist()
        
        # Create arrays with (x,y) coordinates of the points
        values_c3 = np.vstack([c3_endo_etas, c3_endo_thetas])
        values_c2 = np.vstack([c2_endo_etas, c2_endo_thetas])

        # Approximate the Densité by a gaussian kernel
        kernel_c3 = st.gaussian_kde(values_c3)
        kernel_c2 = st.gaussian_kde(values_c2)

        # Create 100x100 regular (x,y,z) values for the plot
        xx, yy = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f_c3 = np.reshape(kernel_c3(positions).T, xx.shape)
        pbar.update(1)
        f_c2 = np.reshape(kernel_c2(positions).T, xx.shape)
        pbar.update(1)

        # Save the data to an archive for later use without the need to recompute
        np.savez(runDir + f"/data/wadley_kernel_{angle}_{res}A.npz",
                  c3_endo_e=c3_endo_etas, c3_endo_t=c3_endo_thetas,
                  c2_endo_e=c2_endo_etas, c2_endo_t=c2_endo_thetas,
                  kernel_c3=f_c3, kernel_c2=f_c2)
        pbar.close()
        idxQueue.put(thr_idx)
    else:
        setproctitle(f"RNANet statistics.py reproduce_wadley_results(carbon={carbon})")

        f = np.load(runDir + f"/data/wadley_kernel_{angle}_{res}A.npz")
        c2_endo_etas = f["c2_endo_e"]
        c3_endo_etas = f["c3_endo_e"]
        c2_endo_thetas = f["c2_endo_t"]
        c3_endo_thetas = f["c3_endo_t"]
        f_c3 = f["kernel_c3"]
        f_c2 = f["kernel_c2"]
        xx, yy = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]

    # notify(f"Kernel computed for {angle}/th{angle} (or loaded from file).")

    # exact counts:
    hist_c2, xedges, yedges = np.histogram2d(c2_endo_etas, c2_endo_thetas, bins=int(2*np.pi/0.1), 
                                             range=[[0, 2*np.pi], [0, 2*np.pi]])
    hist_c3, xedges, yedges = np.histogram2d(c3_endo_etas, c3_endo_thetas, bins=int(2*np.pi/0.1), 
                                             range=[[0, 2*np.pi], [0, 2*np.pi]])
    cmap = cm.get_cmap("jet")
    color_values = cmap(hist_c3.ravel()/hist_c3.max())

    for x, y, hist, f, l in zip( (c3_endo_etas, c2_endo_etas), 
                                 (c3_endo_thetas, c2_endo_thetas), 
                                 (hist_c3, hist_c2), 
                                 (f_c3, f_c2), ("c3","c2")):
        # cut hist and kernel
        hist_sup_thr = hist.mean() + sd_range[1]*hist.std()
        hist_cut = np.where( hist > hist_sup_thr, hist_sup_thr, hist)
        f_sup_thr = f.mean() + sd_range[1]*f.std()
        f_low_thr = f.mean() + sd_range[0]*f.std()
        f_cut = np.where(f > f_sup_thr, f_sup_thr, f)
        f_cut = np.where(f_cut < f_low_thr, 0, f_cut)
        levels = [ f.mean()+f.std(), f.mean()+2*f.std(), f.mean()+4*f.std()]

        # histogram:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        ax.bar3d(xpos.ravel(), ypos.ravel(), 0.0, 0.09, 0.09, hist_cut.ravel(), color=color_values, zorder="max")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(runDir + f"/results/figures/wadley_plots/wadley_hist_{angle}_{l}_{res}A.png")
        if show:
            fig.show()
        plt.close()

        # Smoothed joint distribution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, f_cut, cmap=cm.get_cmap("coolwarm"), linewidth=0, antialiased=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(runDir + f"/results/figures/wadley_plots/wadley_distrib_{angle}_{l}_{res}A.png")
        if show:
            fig.show()
        plt.close()

        # 2D Wadley plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.scatter(x, y, s=1, alpha=0.1)
        ax.contourf(xx, yy, f, alpha=0.5, cmap=cm.get_cmap("coolwarm"), levels=levels, extend="max")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(runDir + f"/results/figures/wadley_plots/wadley_{angle}_{l}_{res}A.png")
        if show:
            fig.show()
        plt.close()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")

    # print(f"[{worker_nbr}]\tComputed joint distribution of angles (C{carbon}) and saved the figures.")

@trace_unhandled_exceptions
def stats_len():
    """Plots statistics on chain lengths in RNA families.
    Uses all chains mapped to a family including copies, inferred or not.
    
    REQUIRES tables chain, nucleotide up to date.
    """

    setproctitle(f"RNANet statistics.py stats_len({res_thr})")
    
    # Get a worker number to position the progress bar
    global idxQueue
    thr_idx = idxQueue.get()

    cols = []
    lengths = []
    
    for f in tqdm(famlist, position=thr_idx+1, desc=f"Worker {thr_idx+1}: Average chain lengths", unit="family", leave=False):

        # Define a color for that family in the plot
        if f in LSU_set:
            cols.append("red") # LSU
        elif f in SSU_set:
            cols.append("blue") # SSU
        elif f in ["RF00001"]:
            cols.append("green")
        elif f in ["RF00005"]:
            cols.append("orange")
        else:
            cols.append("grey")

        # Get the lengths of chains
        with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal')
            l = [ x[0] for x in sql_ask_database(conn, f"""SELECT COUNT(index_chain) 
                                                            FROM (
                                                                SELECT chain_id 
                                                                FROM chain JOIN structure ON chain.structure_id = structure.pdb_id 
                                                                WHERE rfam_acc='{f}' AND resolution <= {res_thr}
                                                            ) NATURAL JOIN nucleotide 
                                                            GROUP BY chain_id;""", warn_every=0) ]
        lengths.append(l) # list of chain lengths from the family

    # Plot the figure
    fig = plt.figure(figsize=(10,3))
    ax = fig.gca()
    ax.hist(lengths, bins=100, stacked=True, log=True, color=cols, label=famlist)
    ax.set_xlabel("Sequence length (nucleotides)", fontsize=8)
    ax.set_ylabel("Number of 3D chains", fontsize=8)
    ax.set_xlim(left=-150)
    ax.tick_params(axis='both', which='both', labelsize=8)
    fig.tight_layout()

    # Draw the legend
    fig.subplots_adjust(right=0.78)
    filtered_handles = [mpatches.Patch(color='red'), mpatches.Patch(color='white'), mpatches.Patch(color='white'), mpatches.Patch(color='white'),
                        mpatches.Patch(color='blue'), mpatches.Patch(color='white'), mpatches.Patch(color='white'),
                        mpatches.Patch(color='green'), mpatches.Patch(color='white'),
                        mpatches.Patch(color='orange'), mpatches.Patch(color='white'),
                        mpatches.Patch(color='grey')]
    filtered_labels = ['Large Ribosomal Subunits', '(RF00002, RF02540,', 'RF02541, RF02543,', 'RF02546)',
                        'Small Ribosomal Subunits','(RF01960, RF00177,', 'RF02545)',
                       '5S rRNA', '(RF00001)', 
                       'tRNA', '(RF00005)', 
                       'Other']
    ax.legend(filtered_handles, filtered_labels, loc='right', 
                ncol=1, fontsize='small', bbox_to_anchor=(1.3, 0.5))

    # Save the figure
    fig.savefig(runDir + f"/results/figures/lengths_{res_thr}A.png")
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
    # notify("Computed sequence length statistics and saved the figure.")

def format_percentage(tot, x):
        if not tot:
            return '0 %'
        x = 100*x/tot
        if x >= 0.01:
            x = "%.2f" % x
        elif x == 0:
            return "0 %"
        else:
            x = "<.01"
        return x + '%'

@trace_unhandled_exceptions
def stats_freq():
    """Computes base frequencies in all RNA families.
    Uses all chains mapped to a family including copies, inferred or not.

    Outputs results/frequencies.csv
    REQUIRES tables chain, nucleotide up to date."""

    # Get a worker number to position the progress bar
    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} stats_freq()")

    # Initialize a Counter object for each family
    freqs = {}
    for f in famlist:
        freqs[f] = Counter()

    # List all nt_names happening within a RNA family and store the counts in the Counter
    for f in tqdm(famlist, position=thr_idx+1, desc=f"Worker {thr_idx+1}: Base frequencies", unit="family", leave=False):
        with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal')
            counts = dict(sql_ask_database(conn, f"SELECT nt_name, COUNT(nt_name) FROM (SELECT chain_id from chain WHERE rfam_acc='{f}') NATURAL JOIN nucleotide GROUP BY nt_name;", warn_every=0))
        freqs[f].update(counts)
    
    # Create a pandas DataFrame, and save it to CSV.
    df = pd.DataFrame()
    for f in tqdm(famlist, position=thr_idx+1, desc=f"Worker {thr_idx+1}: Base frequencies", unit="family", leave=False):
        tot = sum(freqs[f].values())
        df = pd.concat([ df, pd.DataFrame([[ format_percentage(tot, x) for x in freqs[f].values() ]], columns=list(freqs[f]), index=[f]) ])
    df = df.fillna(0)
    df.to_csv(runDir + "/results/frequencies.csv")    
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
    # notify("Saved nucleotide frequencies to CSV file.")

@trace_unhandled_exceptions
def parallel_stats_pairs(f):
    """Counts occurrences of intra-chain base-pair types in one RNA family

    REQUIRES tables chain, nucleotide up-to-date.""" 

    if path.isfile(runDir + "/data/"+f+"_pairs.csv") and path.isfile(runDir + "/data/"+f+"_counts.csv"):
        return

    # Get a worker number to position the progress bar
    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} p_stats_pairs({f})")

    chain_id_list = mappings_list[f]
    data = []
    sqldata = []
    for cid in tqdm(chain_id_list, position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} basepair types", unit="chain",leave=False):
        with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal')
            # Get comma separated lists of basepairs per nucleotide
            interactions = pd.DataFrame(
                            sql_ask_database(conn, f"SELECT nt_code as nt1, index_chain, paired, pair_type_LW FROM nucleotide WHERE chain_id='{cid}';"), 
                            columns = ["nt1", "index_chain", "paired", "pair_type_LW"]
                           )
        # expand the comma-separated lists in real lists
        expanded_list = pd.concat([ pd.DataFrame({  'nt1':[ row["nt1"] for x in row["paired"].split(',') ],
                                                    'index_chain':[ row['index_chain'] for x in row["paired"].split(',') ],
                                                    'paired':row['paired'].split(','), 
                                                    'pair_type_LW':row['pair_type_LW'].split(',') 
                                                }) 
                                    for _, row in interactions.iterrows() 
                                ]).reset_index(drop=True)

        # Add second nucleotide
        nt2 = []
        for _, row in expanded_list.iterrows():
            if row.paired in ['', '0']:
                nt2.append('')
            else:
                try:
                    n = expanded_list[expanded_list.index_chain == int(row.paired)].nt1.tolist()[0]
                    nt2.append(n)
                except IndexError:
                    print(cid, flush=True)
        try:
            expanded_list["nt2"] = nt2
        except ValueError:
            print(cid, flush=True)
            print(expanded_list, flush=True)
            return 0,0

        # keep only intra-chain interactions
        expanded_list = expanded_list[ ~expanded_list.paired.isin(['0','']) ]
        expanded_list["nts"] = expanded_list["nt1"] + expanded_list["nt2"]
        
        # Get basepair type
        expanded_list["basepair"] = np.where(expanded_list.nts.isin(["AU","UA"]), "AU",
                                        np.where(expanded_list.nts.isin(["GC","CG"]), "GC",
                                            np.where(expanded_list.nts.isin(["GU","UG"]), "Wobble","Other")
                                        )
                                    )
        expanded_list = expanded_list[["basepair", "pair_type_LW"]]

        # Update the database
        vlcnts = expanded_list.pair_type_LW.value_counts()
        sqldata.append(   ( vlcnts.at["cWW"]/2 if "cWW" in vlcnts.index else 0, 
                            vlcnts.at["cWH"] if "cWH" in vlcnts.index else 0, 
                            vlcnts.at["cWS"] if "cWS" in vlcnts.index else 0, 
                            vlcnts.at["cHH"]/2 if "cHH" in vlcnts.index else 0, 
                            vlcnts.at["cHS"] if "cHS" in vlcnts.index else 0, 
                            vlcnts.at["cSS"]/2 if "cSS" in vlcnts.index else 0, 
                            vlcnts.at["tWW"]/2 if "tWW" in vlcnts.index else 0, 
                            vlcnts.at["tWH"] if "tWH" in vlcnts.index else 0, 
                            vlcnts.at["tWS"] if "tWS" in vlcnts.index else 0, 
                            vlcnts.at["tHH"]/2 if "tHH" in vlcnts.index else 0, 
                            vlcnts.at["tHS"] if "tHS" in vlcnts.index else 0, 
                            vlcnts.at["tSS"]/2 if "tSS" in vlcnts.index else 0, 
                            int(sum(vlcnts.loc[[ str(x) for x in vlcnts.index if "." in str(x)]])/2), 
                            cid) )

        data.append(expanded_list)

    # Update the database
    with sqlite3.connect(runDir + "/results/RNANet.db", isolation_level=None) as conn:
        conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
        sql_execute(conn, """UPDATE chain SET pair_count_cWW = ?, pair_count_cWH = ?, pair_count_cWS = ?, pair_count_cHH = ?,
                                pair_count_cHS = ?, pair_count_cSS = ?, pair_count_tWW = ?, pair_count_tWH = ?, pair_count_tWS = ?, 
                                pair_count_tHH = ?, pair_count_tHS = ?, pair_count_tSS = ?, pair_count_other = ? WHERE chain_id = ?;""", many=True, data=sqldata, warn_every=0)

    # merge all the dataframes from all chains of the family
    expanded_list = pd.concat(data)

    # Count each pair type
    vcnts = expanded_list.pair_type_LW.value_counts()

    # Add these new counts to the family's counter
    cnt = Counter()
    cnt.update(dict(vcnts))

    # Create an output DataFrame
    f_df = pd.DataFrame([[ x for x in cnt.values() ]], columns=list(cnt), index=[f])
    f_df.to_csv(runDir + f"/data/{f}_counts.csv")
    expanded_list.to_csv(runDir + f"/data/{f}_pairs.csv")
    
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")

@trace_unhandled_exceptions
def to_id_matrix(f):
    """
    Runs esl-alipid on the filtered alignment to get an identity matrix.
    """
    if path.isfile("data/"+f+".npy"):
        return 0
  
    # Get a worker number to position the progress bar
    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} to_id_matrix({f})")

    if not path.isfile(f"{path_to_seq_data}/realigned/{f}_3d_only.stk"):
        warn(f"File not found: {path_to_seq_data}/realigned/{f}_3d_only.stk")
    align = AlignIO.read(f"{path_to_seq_data}/realigned/{f}_3d_only.stk", "stockholm")
    names = [ x.id for x in align if '[' in x.id ]
    del align
    
    pbar = tqdm(total = len(names)*(len(names)-1)*0.5, position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} idty matrix", unit="comparisons", leave=False)
    pbar.update(0)
    
    # Prepare the job
    process = subprocess.Popen(shlex.split(f"esl-alipid --rna --noheader --informat stockholm {path_to_seq_data}/realigned/{f}_3d_only.stk"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    id_matrix = np.zeros((len(names), len(names)))
    cnt = 0
    while not cnt or process.poll() is None:
        output = process.stdout.read()
        if output:
            lines = output.strip().split(b'\n')
            for l in lines:
                cnt += 1
                line = l.split()
                s1 = line[0].decode('utf-8')
                s2 = line[1].decode('utf-8')
                score = line[2].decode('utf-8')
                id1 = names.index(s1)
                id2 = names.index(s2)
                id_matrix[id1, id2] = float(score)
                pbar.update(1)
    if cnt != len(names)*(len(names)-1)*0.5:
        warn(f"{f} got {cnt} updates on {len(names)*(len(names)-1)*0.5}")
    if process.poll() != 0:
        l = process.stderr.read().strip().split(b'\n')
        warn("\n".join([ line.decode('utf-8') for line in l ]))
    pbar.close()

    np.save("data/"+f+".npy", id_matrix)

    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
    return 0

@trace_unhandled_exceptions
def seq_idty():
    """Computes identity matrices for each of the RNA families.
    
    REQUIRES temporary results files in data/*.npy
    REQUIRES tables chain, family up to date."""

    # load distance matrices
    fams_to_plot = [ f for f in famlist if f not in ignored ]
    fam_arrays = []
    for f in fams_to_plot:
        if path.isfile("data/"+f+".npy"):
            fam_arrays.append(np.load("data/"+f+".npy") / 100.0)  # normalize percentages in [0,1]
        else:
            warn("data/"+f+".npy not found !")
            fam_arrays.append(np.array([]))

    # Update database with identity percentages
    conn = sqlite3.connect(runDir + "/results/RNANet.db")
    conn.execute('pragma journal_mode=wal')
    for f, D in zip(fams_to_plot, fam_arrays):
        if not len(D): continue
        if D.shape[0] > 1:
            a = np.sum(D) * 2 / D.shape[0] / (D.shape[0] - 1)    # SUM(D) / (n(n-1)/2)
        else:
            a = D[0][0]
        conn.execute(f"UPDATE family SET idty_percent = {round(float(a),2)} WHERE rfam_acc = '{f}';")
    conn.commit()
    conn.close()

    # Plots plots plots
    fig, axs = plt.subplots(4,17, figsize=(17,5.75))
    axs = axs.ravel()
    [axi.set_axis_off() for axi in axs]
    im = "" # Just to declare the variable, it will be set in the loop
    for f, D, ax in zip(fams_to_plot, fam_arrays, axs):
        D = D + D.T         # Copy the lower triangle to upper, to get a symetrical matrix
        if D.shape[0] > 2:  # Cluster only if there is more than 2 sequences to organize
            D = 1.0 - D
            np.fill_diagonal(D, 0.0)
            condensedD = squareform(D)

            # Compute basic dendrogram by Ward's method
            Y = sch.linkage(condensedD, method='ward')
            Z = sch.dendrogram(Y, orientation='left', no_plot=True)

            # Reorganize rows and cols
            idx1 = Z['leaves']
            D = D[idx1[::-1],:]
            D = D[:,idx1[::-1]]
            D = 1.0 - D
        elif D.shape[0] == 2:
            np.fill_diagonal(D, 1.0) # the diagonal has been ignored until now
        ax.text(np.floor(D.shape[0]/2.0)-(0.5 if not D.shape[0]%2 else 0), -0.5, f + "\n(" + str(D.shape[0]) + " chains)", 
                fontsize=9, horizontalalignment = 'center', verticalalignment='bottom')
        im = ax.matshow(D, vmin=0, vmax=1)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    fig.colorbar(im, ax=axs[-4], shrink=0.8)
    fig.savefig(runDir + f"/results/figures/distances_{res_thr}.png")
    print("> Computed all identity matrices and saved the figure.", flush=True)

@trace_unhandled_exceptions
def stats_pairs():
    """Counts occurrences of intra-chain base-pair types in RNA families

    Creates a temporary results file in data/pair_counts.csv, and a results file in results/pairings.csv.
    REQUIRES tables chain, nucleotide up-to-date.""" 
    
    setproctitle(f"RNANet statistics.py stats_pairs()")

    def line_format(family_data):
        return family_data.apply(partial(format_percentage, sum(family_data)))

    if not path.isfile("data/pair_counts_{res_thr}.csv"):
        results = []
        allpairs = []
        for f in famlist:
            newpairs = pd.read_csv(runDir + f"/data/{f}_pairs.csv", index_col=0)
            fam_df = pd.read_csv(runDir + f"/data/{f}_counts.csv", index_col=0)
            results.append(fam_df)
            allpairs.append(newpairs)
            subprocess.run(["rm", "-f", runDir + f"/data/{f}_pairs.csv"])
            subprocess.run(["rm", "-f", runDir + f"/data/{f}_counts.csv"])
        all_pairs = pd.concat(allpairs)
        df = pd.concat(results).fillna(0)
        df.to_csv(runDir + f"/data/pair_counts_{res_thr}.csv")
        all_pairs.to_csv(runDir + f"/data/all_pairs_{res_thr}.csv")
    else:
        df = pd.read_csv(runDir + f"/data/pair_counts_{res_thr}.csv", index_col=0)
        all_pairs = pd.read_csv(runDir + f"/data/all_pairs_{res_thr}.csv", index_col=0)

    crosstab = pd.crosstab(all_pairs.pair_type_LW, all_pairs.basepair)
    col_list = [ x for x in df.columns if '.' in x ]

    # Remove not very well defined pair types (not in the 12 LW types)
    df['other'] = df[col_list].sum(axis=1)
    df.drop(col_list, axis=1, inplace=True)
    crosstab = crosstab.append(crosstab.loc[col_list].sum(axis=0).rename("non-LW"))
    
    # drop duplicate types
    # The twelve Leontis-Westhof types are
    # cWW cWH cWS cHH cHS cSS (do not count cHW cSW and cSH, they are the same as their opposites)
    # tWW tWH tWS tHH tHS tSS (do not count tHW tSW and tSH, they are the same as their opposites)
    df = df.drop([ x for x in [ "cHW", "tHW", "cSW", "tSW", "cHS", "tHS"] if x in df.columns], axis=1)
    crosstab = crosstab.loc[[ x for x in ["cWW","cWH","cWS","cHH","cHS","cSS","tWW","tWH","tWS","tHH","tHS","tSS","non-LW"] if x in crosstab.index]]
    df.loc[:,[x for x in ["cWW", "tWW", "cHH", "tHH", "cSS", "tSS", "other"] if x in df.columns] ] /= 2
    crosstab.loc[["cWW", "tWW", "cHH", "tHH", "cSS", "tSS", "non-LW"]] /= 2

    # Compute total row
    total_series = df.sum(numeric_only=True).rename("TOTAL")
    df = df.append(total_series)

    # format as percentages
    df = df.apply(line_format, axis=1)

    # reorder columns
    df.sort_values("TOTAL", axis=1, inplace=True, ascending=False)
    crosstab = crosstab[["AU", "GC", "Wobble", "Other"]]

    # Save to CSV
    df.to_csv(runDir + "/results/pair_types.csv")

    # Plot barplot of overall types
    ax = crosstab.plot(figsize=(8,5), kind='bar', stacked=True, log=False, fontsize=13)
    ax.set_ylabel("Number of observations (millions)", fontsize=13)
    ax.set_xlabel(None)
    plt.subplots_adjust(left=0.1, bottom=0.16, top=0.95, right=0.99)
    plt.savefig(runDir + f"/results/figures/pairings_{res_thr}.png")

    notify("Computed nucleotide statistics and saved CSV and PNG file.")

@trace_unhandled_exceptions
def per_chain_stats():
    """Computes per-chain frequencies and base-pair type counts.

    REQUIRES tables chain, nucleotide up to date. """
    
    setproctitle(f"RNANet statistics.py per_chain_stats()")

    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        # Compute per-chain nucleotide frequencies
        df = pd.read_sql("SELECT SUM(is_A) as A, SUM(is_C) AS C, SUM(is_G) AS G, SUM(is_U) AS U, SUM(is_other) AS O, chain_id FROM nucleotide GROUP BY chain_id;", conn)
        df["total"] = pd.Series(df.A + df.C + df.G + df.U + df.O, dtype=np.float64)
        df[['A','C','G','U','O']] = df[['A','C','G','U','O']].div(df.total, axis=0)
        df = df.drop("total", axis=1)

        # Set the values
        sql_execute(conn, "UPDATE chain SET chain_freq_A = ?, chain_freq_C = ?, chain_freq_G = ?, chain_freq_U = ?, chain_freq_other = ? WHERE chain_id= ?;",
                          many=True, data=list(df.to_records(index=False)), warn_every=10)
    print("> Updated the database with per-chain base frequencies", flush=True)

@trace_unhandled_exceptions
def general_stats():
    """
    Number of structures as function of the resolution threshold
    Number of Rfam families as function of the resolution threshold
    """

    setproctitle(f"RNANet statistics.py general_stats()")

    reqs = [
        # unique unmapped chains with no issues
        """ SELECT distinct pdb_id, chain_name, exp_method, resolution
            FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
            WHERE rfam_acc = 'unmappd' AND ISSUE=0;""",

        # unique mapped chains with no issues
        """ SELECT distinct pdb_id, chain_name, exp_method, resolution
            FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
            WHERE rfam_acc != 'unmappd' AND ISSUE=0;""",

        # mapped chains with no issues
        """ SELECT pdb_id, chain_name, inferred, rfam_acc, pdb_start, pdb_end, exp_method, resolution
            FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
            WHERE rfam_acc != 'unmappd' AND ISSUE=0;""",

        # mapped chains with no issues that are all inferred
        """ SELECT DISTINCT pdb_id, c.chain_name, exp_method, resolution
            FROM (
                SELECT inferred, rfam_acc, pdb_start, pdb_end, chain.structure_id, chain.chain_name, r.redundancy, r.inf_redundancy
                FROM chain 
                JOIN (SELECT structure_id, chain_name, COUNT(distinct rfam_acc) AS redundancy, SUM(inferred) AS inf_redundancy 
                        FROM chain 
                        WHERE rfam_acc != 'unmappd' AND issue=0 
                        GROUP BY structure_id, chain_name
                ) AS r ON chain.structure_id=r.structure_id AND chain.chain_name = r.chain_name 
                WHERE r.redundancy=r.inf_redundancy AND rfam_acc != 'unmappd' and issue=0
            ) AS c
            JOIN structure ON c.structure_id=structure.pdb_id;""",

        # Number of mapped chains (not inferred)
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 0);""",

        # Number of unique mapped chains (not inferred)
        """SELECT count(*) FROM (SELECT DISTINCT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 0);""",

        # Number of mapped chains (inferred)
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 1);""",

        # Number of unique mapped chains (inferred)
        """SELECT count(*) FROM (SELECT DISTINCT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 1);""",

        # Number of mapped chains inferred once
        """SELECT count(*) FROM (
                SELECT structure_id, chain_name, COUNT(DISTINCT rfam_acc) as c 
                FROM chain where rfam_acc!='unmappd' and inferred=1 
                GROUP BY structure_id, chain_name
            ) WHERE c=1;""",

        # Number of mapped chains inferred twice
        """select count(*) from (
                select structure_id, chain_name, count(distinct rfam_acc) as c 
                from chain where rfam_acc!='unmappd' and inferred=1 
                group by structure_id, chain_name
            ) where c=2;""",

        # Number of mapped chains inferred 3 times or more
        """select count(*) from (
                select structure_id, chain_name, count(distinct rfam_acc) as c 
                from chain where rfam_acc!='unmappd' and inferred=1 
                group by structure_id, chain_name
            ) where c>2;""",

        # Number of chains both mapped with and without inferrence
        """ SELECT COUNT(*) FROM (
                SELECT structure_id, chain_name, sum(inferred) AS s, COUNT(rfam_acc) AS c 
                FROM chain 
                WHERE rfam_acc!='unmappd' 
                GROUP BY structure_id, chain_name
            ) 
            WHERE s < c AND s > 0;""",
        
        # Number of mapped chains (total)
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd');""",

        # Number of unique mapped chains
        """SELECT count(*) FROM (SELECT DISTINCT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd');""",

        # Number of unmapped chains
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc = 'unmappd');""",
        
        # Number of mapped chains without issues (not inferred)
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 0 AND issue = 0);""",

        # Number of unique mapped chains without issues (not inferred)
        """SELECT count(*) FROM (SELECT DISTINCT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 0 AND issue = 0);""",

        # Number of mapped chains without issues (inferred)
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 1 AND issue=0);""",

        # Number of unique mapped chains without issues (inferred)
        """SELECT count(*) FROM (SELECT DISTINCT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND inferred = 1 AND issue=0);""",

        # Number of mapped chains without issues (total)
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND issue=0);""",

        # Number of unique mapped chains without issues
        """SELECT count(*) FROM (SELECT DISTINCT structure_id, chain_name FROM chain WHERE rfam_acc != 'unmappd' AND issue=0);""",

        # Number of unmapped chains without issues
        """SELECT count(*) FROM (SELECT structure_id, chain_name FROM chain WHERE rfam_acc = 'unmappd' AND issue=0);"""
    ]

    answers = []
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        for r in reqs:
            answers.append(pd.read_sql(r, conn))
    df_unique = answers[0]
    df_mapped_unique = answers[1]
    df_mapped_copies = answers[2]
    df_inferred_only_unique = answers[3]
    print()
    print("> found", answers[4].iloc[0][0], f"chains ({answers[5].iloc[0][0]} unique chains) that are mapped thanks to Rfam. Removing chains with issues, only {answers[15].iloc[0][0]} ({answers[16].iloc[0][0]} unique)")
    if answers[4].iloc[0][0] != answers[5].iloc[0][0]:
        print("\t> This happens because different parts of the same chain can be mapped to different families.")
    print("> found", answers[6].iloc[0][0], f"chains ({answers[7].iloc[0][0]} unique chains) that are mapped by inferrence. Removing chains with issues, only {answers[17].iloc[0][0]} ({answers[18].iloc[0][0]} unique).")
    print("\t> ", answers[8].iloc[0][0], "chains are mapped only once,")
    print("\t> ", answers[9].iloc[0][0], "are mapped to 2 families,")
    print("\t> ", answers[10].iloc[0][0], "are mapped to 3 or more.")
    print("> Among them,", answers[11].iloc[0][0], "chains are mapped both with families found on Rfam and by inferrence.")
    if answers[11].iloc[0][0]:
        print("\t> this is normal if you used option -f (--full-inference). Otherwise, there might be a problem.")
    print("> TOTAL:", answers[12].iloc[0][0], f"chains ({answers[13].iloc[0][0]} unique chains) mapped to a family. Removing chains with issues, only {answers[19].iloc[0][0]} ({answers[20].iloc[0][0]} unique).")
    print("> TOTAL:", answers[14].iloc[0][0], f"unmapped chains. Removing chains with issues, {answers[21].iloc[0][0]}.")
    if answers[14].iloc[0][0]:
        print("\t> this is normal if you used option --no-homology. Otherwise, there might be a problem.")
    print()

    ##########################################
    # plot N = f(resolution, exp_method)
    ##########################################

    methods = df_unique.exp_method.unique()

    fig, axs = plt.subplots(1+len(methods), 3, figsize=(15,5*(1+len(methods))), sharex=True)
    df_unique.sort_values('resolution', inplace=True, ignore_index=True)
    df_mapped_unique.sort_values('resolution', inplace=True, ignore_index=True)
    df_inferred_only_unique.sort_values('resolution', inplace=True, ignore_index=True)
    df_mapped_copies.sort_values('resolution', inplace=True, ignore_index=True)
    max_res = max(df_unique.resolution)
    max_structs = max(len(df_mapped_copies.index), len(df_unique.index))
    colors = np.linspace(0,1,1+len(methods))
    plt.xticks( np.arange(0, max_res+2, 2.0).tolist(),  np.arange(0, max_res+2, 2.0).tolist() )

    axs[0][0].grid(axis='y', ls='dotted', lw=1)
    axs[0][0].hist(df_unique.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 1, colors[0], 1), label='distribution')
    axs[0][0].hist(df_unique.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 0, colors[0], 0.5), cumulative=True, label='cumulative')
    axs[0][0].text(0.95*max_res, 0.95*len(df_unique.resolution), "%d " %  len(df_unique.resolution), 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
    axs[0][0].set_ylabel("ALL", fontsize=14)
    axs[0][0].set_title("Number of unique RNA chains", fontsize=14)
    axs[0][0].set_ylim((0, max_structs * 1.05))
    axs[0][0].legend(loc="lower right", fontsize=14)

    axs[0][1].grid(axis='y', ls='dotted', lw=1)
    axs[0][1].set_yticklabels([])
    axs[0][1].hist(df_mapped_unique.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 1, colors[0], 1), label='distribution')
    axs[0][1].hist(df_mapped_unique.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 0, colors[0], 0.5), cumulative=True, label='cumulative')
    axs[0][1].hist(df_inferred_only_unique.resolution, bins=np.arange(0, max_res, 0.5), fc=(0.2, 0, colors[0], 0.5), cumulative=True, label='only by inference')
    axs[0][1].text(0.95*max_res, 0.95*len(df_mapped_unique.resolution), "%d " %  len(df_mapped_unique.resolution), 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
    axs[0][1].set_title(r"Number of unique RNA chains\nmapped to $\geq 1$ family", fontsize=14)
    axs[0][1].set_ylim((0, max_structs * 1.05))
    axs[0][1].legend(loc="upper left", fontsize=14)

    axs[0][2].grid(axis='y', ls='dotted', lw=1)
    axs[0][2].set_yticklabels([])
    axs[0][2].hist(df_mapped_copies.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 1, colors[0], 1), label='distribution')
    axs[0][2].hist(df_mapped_copies.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 0, colors[0], 0.5), cumulative=True, label='cumulative')
    axs[0][2].hist(df_mapped_copies[df_mapped_copies.inferred == 1].resolution, bins=np.arange(0, max_res, 0.5), fc=(0.2, 0, colors[0], 0.5), cumulative=True, label='inferred')
    axs[0][2].text(0.95*max_res, 0.95*len(df_mapped_copies.resolution), "%d " %  len(df_mapped_copies.resolution), 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
    axs[0][2].set_title("Number of RNA chains mapped to a\nfamily (with copies)", fontsize=14)
    axs[0][2].legend(loc="upper left", fontsize=14)
    axs[0][2].set_ylim((0, max_structs * 1.05))

    for i,m in enumerate(methods):
        df_unique_m = df_unique[df_unique.exp_method == m]
        df_mapped_unique_m = df_mapped_unique[df_mapped_unique.exp_method == m]
        df_inferred_only_unique_m = df_inferred_only_unique[df_inferred_only_unique.exp_method == m]
        df_mapped_copies_m = df_mapped_copies[ df_mapped_copies.exp_method == m]
        max_structs = max(len(df_mapped_copies_m.index), len(df_unique_m.index))
        print("> found", max_structs, "structures with method", m, flush=True)

        axs[1+i][0].grid(axis='y', ls='dotted', lw=1)
        axs[1+i][0].hist(df_unique_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 1, colors[1+i], 1), label='distribution')
        axs[1+i][0].hist(df_unique_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 0, colors[1+i], 0.5), cumulative=True, label='cumulative')
        axs[1+i][0].text(0.95*max_res, 0.95*len(df_unique_m.resolution), "%d " %  len(df_unique_m.resolution), 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
        axs[1+i][0].set_ylim((0, max_structs * 1.05))
        axs[1+i][0].set_ylabel(m, fontsize=14)
        axs[1+i][0].legend(loc="lower right", fontsize=14)

        axs[1+i][1].grid(axis='y', ls='dotted', lw=1)
        axs[1+i][1].set_yticklabels([])
        axs[1+i][1].hist(df_mapped_unique_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 1, colors[1+i], 1), label='distribution')
        axs[1+i][1].hist(df_mapped_unique_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 0, colors[1+i], 0.5), cumulative=True, label='cumulative')
        axs[1+i][1].hist(df_inferred_only_unique_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0.2, 0, colors[1+i], 0.5), cumulative=True, label='only by inference')
        axs[1+i][1].text(0.95*max_res, 0.95*len(df_mapped_unique_m.resolution), "%d " %  len(df_mapped_unique_m.resolution), 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
        axs[1+i][1].set_ylim((0, max_structs * 1.05))
        axs[1+i][1].legend(loc="upper left", fontsize=14)
        
        axs[1+i][2].grid(axis='y', ls='dotted', lw=1)
        axs[1+i][2].set_yticklabels([])
        axs[1+i][2].hist(df_mapped_copies_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 1, colors[1+i], 1), label='distribution')
        axs[1+i][2].hist(df_mapped_copies_m.resolution, bins=np.arange(0, max_res, 0.5), fc=(0, 0, colors[1+i], 0.5), cumulative=True, label='cumulative')
        axs[1+i][2].hist(df_mapped_copies_m[df_mapped_copies_m.inferred == 1].resolution, bins=np.arange(0, max_res, 0.5), fc=(0.2, 0, colors[1+i], 0.5), cumulative=True, label='inferred')
        axs[1+i][2].text(0.95*max_res, 0.95*len(df_mapped_copies_m.resolution), "%d " %  len(df_mapped_copies_m.resolution), 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
        axs[1+i][2].set_ylim((0, max_structs * 1.05))
        axs[1+i][2].legend(loc="upper left", fontsize=14)
    
    axs[-1][0].set_xlabel("Structure resolution\n(Angströms, lower is better)", fontsize=14)
    axs[-1][1].set_xlabel("Structure resolution\n(Angströms, lower is better)", fontsize=14)
    axs[-1][2].set_xlabel("Structure resolution\n(Angströms, lower is better)", fontsize=14)

    fig.suptitle("Number of RNA chains by experimental method and resolution", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.98, wspace=0.05, 
                        hspace=0.05, bottom=0.05, top=0.92)
    fig.savefig(runDir + "/results/figures/resolutions.png")
    plt.close()

    ##########################################
    # plot Nfam = f(resolution, exp_method)
    ##########################################

    df_mapped_copies['n_fam'] = [ len(df_mapped_copies.rfam_acc[:i+1].unique()) for i in range(len(df_mapped_copies.index)) ]

    fig, axs = plt.subplots(1, 1+len(methods), figsize=(5*(1+len(methods)), 5))
    max_res = max(df_mapped_copies.resolution)
    max_fams = max(df_mapped_copies.n_fam)
    colors = np.linspace(0,1,1+len(methods))
    plt.xticks( np.arange(0, max_res+2, 2.0).tolist(),  np.arange(0, max_res+2, 2.0).tolist() )

    axs[0].grid(axis='y', ls='dotted', lw=1)
    axs[0].plot(df_mapped_copies.resolution, df_mapped_copies.n_fam)
    axs[0].text(0.95*max_res, 0.95*df_mapped_copies.n_fam.iloc[-1], "%d " %  df_mapped_copies.n_fam.iloc[-1], 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
    axs[0].set_title("ALL", fontsize=14)
    axs[0].set_xlabel("Structure resolution (Angströms)", fontsize=14)
    axs[0].set_ylabel("Number of Rfam families", fontsize=14)
    axs[0].set_ylim((0, max_res * 1.05))
    axs[0].set_ylim((0, max_fams * 1.05))
    
    for i,m in enumerate(methods):
        df_mapped_copies_m = df_mapped_copies[ df_mapped_copies.exp_method == m].drop("n_fam", axis=1).copy()
        df_mapped_copies_m['n_fam'] = [ len(df_mapped_copies_m.rfam_acc[:i+1].unique()) for i in range(len(df_mapped_copies_m.index)) ]
        print(">", df_mapped_copies_m.n_fam.iloc[-1], "different RNA families have a 3D structure solved by", m)

        axs[1+i].grid(axis='y', ls='dotted', lw=1)
        axs[1+i].plot(df_mapped_copies_m.resolution, df_mapped_copies_m.n_fam, )
        axs[1+i].text(0.95*max(df_mapped_copies_m.resolution), 0.95*df_mapped_copies_m.n_fam.iloc[-1], "%d " %  df_mapped_copies_m.n_fam.iloc[-1], 
                         horizontalalignment='right', verticalalignment='top', fontsize=14)
        axs[1+i].set_xlim((0, max_res * 1.05))
        axs[1+i].set_ylim((0, max_fams * 1.05))
        axs[1+i].set_xlabel("Structure resolution (Angströms)", fontsize=14)
        axs[1+i].set_title(m, fontsize=14)
        axs[1+i].set_yticklabels([])
    
    fig.suptitle("Number of RNA families used by experimental method and resolution", fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.05, 
                        hspace=0.05, bottom=0.12, top=0.84)
    fig.savefig(runDir + "/results/figures/Nfamilies.png")
    plt.close()

def par_distance_matrix(filelist, f, label, cm_coords, consider_all_atoms, s):
    
    # Identify the right 3D file
    filename = ''
    for file in filelist:
        if file.startswith(s.id.split("RF")[0].replace('-', '').replace('[', '_').replace(']', '_')):
            filename = path_to_3D_data + "rna_mapped_to_Rfam/" + file
            break
    if not len(filename):
        return None, None, None
    
    # Get the coordinates of every existing nt in the 3D file
    try:
        coordinates = nt_3d_centers(filename, consider_all_atoms)
        if not len(coordinates):
            # there is not nucleotides in the file, or no C1' atoms for example.
            warn("No C1' atoms in " + filename)
            return None, None, None
    except FileNotFoundError:
        return None, None, None


    # Get the coordinates of every position in the alignment
    nb_gap = 0
    coordinates_with_gaps = []
    for i, letter in enumerate(s.seq):
        if letter in "-.":
            nb_gap += 1
            coordinates_with_gaps.append(np.nan)
        else:
            coordinates_with_gaps.append(coordinates[i - nb_gap])
    
    # Build the pairwise distances
    d = np.zeros((len(s.seq), len(s.seq)), dtype=np.float32)
    for i in range(len(s.seq)):
        for j in range(len(s.seq)):
            if np.isnan(coordinates_with_gaps[i]).any() or np.isnan(coordinates_with_gaps[j]).any():
                d[i,j] = np.NaN
            else:
                d[i,j] = get_euclidian_distance(coordinates_with_gaps[i], coordinates_with_gaps[j])
    
    # Save the individual distance matrices
    # if f not in LSU_set and f not in SSU_set:
    np.savetxt(runDir + '/results/distance_matrices/' + f + '_'+ label + '/'+ s.id.strip("\'") + '.csv', d, delimiter=",", fmt="%.3f")
    
    # For the average and sd, we want to consider only positions of the consensus model. This means:
    #  - Add empty space when we have deletions
    #  - skip measures that correspond to insertions
    i = len(cm_coords)-1
    while cm_coords[i] is None:
        i -= 1
    family_end = int(cm_coords[i])
    i = 0
    while cm_coords[i] is None:
        i += 1
    family_start = int(cm_coords[i])
    # c = np.zeros((family_end, family_end), dtype=np.float32)    # new matrix of size of the consensus model for the family
    c = np.NaN * np.ones((family_end, family_end), dtype=np.float32)
    # set to NaN zones that never exist in the 3D data
    for i in range(family_start-1):
        for j in range(i, family_end):
            c[i,j] = np.NaN
            c[j,i] = np.NaN
    # copy the values ignoring insertions
    for i in range(len(s.seq)):
        if cm_coords[i] is None:
            continue
        for j in range(len(s.seq)):
            if j >= len(cm_coords):
                print(f"Issue with {s.id} mapped to {f} ({label}, {j}/{len(s.seq)}, {len(cm_coords)})")
            if cm_coords[j] is None:
                continue
            c[int(cm_coords[i])-1, int(cm_coords[j])-1] = d[i,j]
    # return the matrices counts, c, c^2
    return 1-np.isnan(c).astype(int), np.nan_to_num(c), np.nan_to_num(c*c)

@trace_unhandled_exceptions
def get_avg_std_distance_matrix(f, consider_all_atoms, multithread=False):
    np.seterr(divide='ignore') # ignore division by zero issues

    if consider_all_atoms:
        label = "base"
    else:
        label = "backbone"

    if not multithread:
        # This function call is for ONE worker.
        # Get a worker number for it to position the progress bar
        global idxQueue
        thr_idx = idxQueue.get()
        setproctitle(f"RNANet statistics.py Worker {thr_idx+1} {f} {label} distance matrices")

    os.makedirs(runDir + '/results/distance_matrices/' + f + '_' + label, exist_ok=True )   

    align = AlignIO.read(path_to_seq_data + f"realigned/{f}_3d_only.afa", "fasta")
    ncols = align.get_alignment_length()
    found = 0
    notfound = 0
    # retrieve the mappings between this family's alignment and the CM model:
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        r = sql_ask_database(conn, f"SELECT structure_id, '_1_', chain_name, '_', CAST(pdb_start AS TEXT), '-', CAST(pdb_end AS TEXT) FROM chain WHERE rfam_acc='{f}';")
        filelist = sorted([ ''.join(list(x))+'.cif' for x in r ])
        r = sql_ask_database(conn, f"SELECT cm_coord FROM align_column WHERE rfam_acc = '{f}' AND index_ali > 0 ORDER BY index_ali ASC;")
        cm_coords = [ x[0] for x in r ] # len(cm_coords) is the number of saved columns. There are many None values in the list.
        i = len(cm_coords)-1
        while cm_coords[i] is None:
            if i == 0:
                # Issue somewhere. Abort.
                warn(f"{f} has no mapping to CM. Ignoring distance matrix.")
                if not multithread:
                    idxQueue.put(thr_idx) # replace the thread index in the queue
                    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
                return 0
            i -= 1
        family_end = int(cm_coords[i])
    counts = np.zeros((family_end, family_end))
    avg = np.zeros((family_end, family_end))
    std = np.zeros((family_end, family_end))
    
    if not multithread:
        pbar = tqdm(total = len(align), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} {label} distance matrices", unit="chains", leave=False)
        pbar.update(0)
        for s in align:
            contrib, d, dsquared = par_distance_matrix(filelist, f, label, cm_coords, consider_all_atoms, s)
            if d is not None:
                found += 1
                counts += contrib
                avg += d
                std += dsquared
            else:
                notfound += 1
            pbar.update(1)
        pbar.close()
    else:
        # We split the work for one family on multiple workers.
        
        p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=nworkers)
        try:
            fam_pbar = tqdm(total=len(align), desc=f"{f} {label} pair distances", position=0, unit="chain", leave=True)
            # Apply work_pssm_remap to each RNA family
            for i, (contrib, d, dsquared) in enumerate(p.imap_unordered(partial(par_distance_matrix, filelist, f, label, cm_coords, consider_all_atoms), align, chunksize=1)):
                if d is not None:
                    found += 1
                    counts += contrib
                    avg += d
                    std += dsquared
                else:
                    notfound += 1
                fam_pbar.update(1)
            fam_pbar.close()
            p.close()
            p.join()
        except KeyboardInterrupt:
            warn("KeyboardInterrupt, terminating workers.", error=True)
            fam_pbar.close()
            p.terminate()
            p.join()
            exit(1)

    # Calculation of the average matrix
    avg = np.divide(avg, counts, where=counts>0, out=np.full_like(avg, np.NaN)) # Ultrafancy way to take avg/counts or NaN if counts is 0
    np.savetxt(runDir + '/results/distance_matrices/' + f + '_'+ label + '/' + f + '_average.csv' , avg, delimiter=",", fmt="%.3f")
    
    fig, ax = plt.subplots()
    im = ax.imshow(avg)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Angströms", rotation=-90, va="bottom")
    ax.set_title(f"Average distance between {f} residues (Angströms)")
    fig.tight_layout()
    fig.savefig(runDir + '/results/distance_matrices/' + f + '_'+ label + '/' + f + '_average.png', dpi=300)
    plt.close()

    # Calculation of the standard deviation matrix by the Huygens theorem
    std = np.divide(std, counts, where=counts>0, out=np.full_like(std, np.NaN))
    mask = np.invert(np.isnan(std))
    value = std[mask] - np.power(avg[mask], 2)
    if ((value[value<0] < -1e-2).any()):
        warn("Erasing very negative variance value !")
    value[value<0] = 0.0 # floating point problems !
    std[mask] = np.sqrt(value)
    np.savetxt(runDir + '/results/distance_matrices/' + f + '_'+ label + '/' + f + '_stdev.csv' , std, delimiter=",", fmt="%.3f")
    
    fig, ax = plt.subplots()
    im = ax.imshow(std)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Angströms", rotation=-90, va="bottom")
    ax.set_title(f"Standard deviation of distances between {f} residues (Angströms)")
    fig.tight_layout()
    fig.savefig(runDir + '/results/distance_matrices/' + f + '_'+ label + '/' + f + '_std.png', dpi=300)
    plt.close()

    # Save log
    with open(runDir + '/results/distance_matrices/' + f + '_'+ label + '/' + f + '.log', 'a') as logfile:
        logfile.write(str(found)+ " chains taken into account for computation. "+ str(notfound)+ " were not found/without atoms.\n")

    # Save associated nucleotide frequencies (off-topic but convenient to do it here)
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        df = pd.read_sql_query(f"SELECT freq_A, freq_C, freq_G, freq_U, freq_other, gap_percent, consensus FROM align_column WHERE rfam_acc = '{f}' AND index_ali > 0 ORDER BY index_ali ASC;", conn)
        df.to_csv(runDir + '/results/distance_matrices/' + f + '_'+ label + '/' + f + '_frequencies.csv', float_format="%.3f")

    if not multithread:
        idxQueue.put(thr_idx) # replace the thread index in the queue
        setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
    return 0

def log_to_pbar(pbar):
    def update(r):
        pbar.update(1)
    return update

def family_order(f):
    # sort the RNA families so that the plots are readable

    if f in LSU_set:
        return 4
    elif f in SSU_set:
        return 3
    elif f in ["RF00001"]:      #
        return 1                # put tRNAs and 5S rRNAs first,
    elif f in ["RF00005"]:      # because of the logarithmic scale of the lengths' figure, otherwise, they look tiny
        return 0                #
    else:
        return 2

def nt_3d_centers(cif_file, consider_all_atoms):
    """Return the nucleotides' coordinates, summarizing a nucleotide by only one point.
    If consider_all_atoms : barycentre is used
    else: C1' atom is the nucleotide

    Some chains have no C1' (e.g. 4v7f-3), therefore, an empty result is returned.
    """
    result  =[]
    structure = MMCIFParser().get_structure(cif_file, cif_file)
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if consider_all_atoms:
                    temp_list = []
                    for atom in residue:
                        temp_list.append(atom.get_coord())
                    lg = len(temp_list)
                    summ = np.sum(temp_list, axis = 0)
                    res_isobaricentre = [summ[0]/lg, summ[1]/lg, summ[2]/lg]
                    result.append([res_isobaricentre[0], res_isobaricentre[1], res_isobaricentre[2]])
                else:
                    coordinates = None
                    for atom in residue:
                        if atom.get_name() == "C1'":
                            coordinates = atom.get_coord()
                    if coordinates is None:
                        # Residue has no C1'
                        res = np.nan
                    else:
                        res = [coordinates[0], coordinates[1], coordinates[2]]
                    result.append(res)
    return(result)

def get_euclidian_distance(L1, L2):
    """Returns the distance between two points (coordinates in lists)
    """

    e = 0
    for i in range(len(L1)):
        e += float(L1[i] - L2[i])**2
    return np.sqrt(e)

def distance(coord1, coord2):
    """
    Returns the distance between two points using their coordinates (x, y, z)
    """
    return np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

def pos_b1(res) :
    """
    Returns the coordinates of virtual atom B1 (center of the first aromatic cycle)
    """
    coordb1=[]
    somme_x_b1=0
    somme_y_b1=0
    somme_z_b1=0
    moy_x_b1=0
    moy_y_b1=0
    moy_z_b1=0
    #different cases
    #some residues have 2 aromatic cycles 
    if res.get_resname() in ['A', 'G', '2MG', '7MG', 'MA6', '6IA', 'OMG' , '2MA', 'B9B', 'A2M', '1MA', 'E7G', 'P7G', 'B8W', 'B8K', 'BGH', '6MZ', 'E6G', 'MHG', 'M7A', 'M2G', 'P5P', 'G7M', '1MG', 'T6A', 'MIA', 'YG', 'YYG', 'I', 'DG', 'N79', '574', 'DJF', 'AET', '12A', 'ANZ', 'UY4'] :  
        c=0
        names=[]
        for atom in res : 
            if (atom.get_fullname() in ['N9', 'C8', 'N7', 'C4', 'C5']) :
                c=c+1
                names.append(atom.get_name())
                coord=atom.get_vector()
                somme_x_b1=somme_x_b1+coord[0]
                somme_y_b1=somme_y_b1+coord[1]
                somme_z_b1=somme_z_b1+coord[2]
            else : 
                c=c
        #calcul coord B1
        if c != 0 :
            moy_x_b1=somme_x_b1/c
            moy_y_b1=somme_y_b1/c
            moy_z_b1=somme_z_b1/c
            coordb1.append(moy_x_b1)
            coordb1.append(moy_y_b1)
            coordb1.append(moy_z_b1)
    #others have only one cycle
    if res.get_resname() in ['C', 'U', 'AG9', '70U', '1RN', 'RSP', '3AU', 'CM0', 'U8U', 'IU', 'E3C', '4SU', '5HM', 'LV2', 'LHH', '4AC', 'CH', 'Y5P', '2MU', '4OC', 'B8T', 'JMH', 'JMC', 'DC', 'B9H', 'UR3', 'I4U', 'B8Q', 'P4U', 'OMU', 'OMC', '5MU', 'H2U', 'CBV', 'M1Y', 'B8N', '3TD', 'B8H'] :
        c=0
        for atom in res :
            if (atom.get_fullname() in ['C6', 'N3', 'N1', 'C2', 'C4', 'C5']):
                c=c+1
                coord=atom.get_vector()
                somme_x_b1=somme_x_b1+coord[0]
                somme_y_b1=somme_y_b1+coord[1]
                somme_z_b1=somme_z_b1+coord[2]
        #calcul coord B1
        if c != 0 :
            moy_x_b1=somme_x_b1/c
            moy_y_b1=somme_y_b1/c
            moy_z_b1=somme_z_b1/c
            coordb1.append(moy_x_b1)
            coordb1.append(moy_y_b1)
            coordb1.append(moy_z_b1)
    return(coordb1)

def pos_b2(res):
    """
    Returns the coordinates of virtual atom B2 (center of the second aromatic cycle, if exists)
    """
    coordb2=[]
    somme_x_b2=0
    somme_y_b2=0
    somme_z_b2=0
    moy_x_b2=0
    moy_y_b2=0
    moy_z_b2=0

    if res.get_resname() in ['A', 'G', '2MG', '7MG', 'MA6', '6IA', 'OMG' , '2MA', 'B9B', 'A2M', '1MA', 'E7G', 'P7G', 'B8W', 'B8K', 'BGH', '6MZ', 'E6G', 'MHG', 'M7A', 'M2G', 'P5P', 'G7M', '1MG', 'T6A', 'MIA', 'YG', 'YYG', 'I', 'DG', 'N79', '574', 'DJF', 'AET', '12A', 'ANZ', 'UY4'] :  #2 cycles aromatiques
        c=0
        for atom in res :
            if atom.get_fullname() in ['C6', 'N3', 'N1', 'C2', 'C4', 'C5'] :
                c=c+1
                coord=atom.get_vector()
                somme_x_b2=somme_x_b2+coord[0]
                somme_y_b2=somme_y_b2+coord[1]
                somme_z_b2=somme_z_b2+coord[2]
        #calcul coord B2
        if c!=0 :
            moy_x_b2=somme_x_b2/c
            moy_y_b2=somme_y_b2/c
            moy_z_b2=somme_z_b2/c
            coordb2.append(moy_x_b2)
            coordb2.append(moy_y_b2)
            coordb2.append(moy_z_b2)
    return coordb2

def dist_atoms(f):
    '''
    Measures the distance between atoms linked by covalent bonds
    '''
    
    name=str.split(f,'.')[0]

    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} dist_atoms({f})")


    last_o3p=[] #o3 'of the previous nucleotide linked to the P of the current nucleotide
    
    liste_common=[]
    liste_purines=[]
    liste_pyrimidines=[]

    parser=MMCIFParser()
    s = parser.get_structure(f, os.path.abspath(path_to_3D_data+ "rna_only/" + f))
    
    
    chain = next(s[0].get_chains())#1 chain per file
    residues=list(chain.get_residues())
    pbar = tqdm(total=len(residues), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} dist_atoms", unit="residu", leave=False)
    pbar.update(0)
    for res in chain :
        
        # for residues A, G, C, U
        op3_p=[]
        p_op1=[]
        p_op2=[]
        p_o5p=[]
        o5p_c5p=[]
        c5p_c4p=[]
        c4p_o4p=[]
        o4p_c1p=[]
        c1p_c2p=[]
        c2p_o2p=[]
        c2p_c3p=[]
        c3p_o3p=[]
        c4p_c3p=[]
        
        #if res = A or G
        c1p_n9=None
        n9_c8=None
        c8_n7=None
        n7_c5=None
        c5_c6=None
        c6_n1=None
        n1_c2=None
        c2_n3=None
        n3_c4=None
        c4_n9=None
        c4_c5=None
        #if res=G
        c6_o6=None
        c2_n2=None
        #if res = A
        c6_n6=None

        #if res = C or U
        c1p_n1=None
        n1_c6=None
        c6_c5=None
        c5_c4=None
        c4_n3=None
        n3_c2=None
        c2_n1=None
        c2_o2=None
        #if res =C
        c4_n4=None
        #if res=U
        c4_o4=None
        last_o3p_p=None


        if res.get_resname()=='A' or res.get_resname()=='G' or res.get_resname()=='C' or res.get_resname()=='U' :
            #get the coordinates of the atoms
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_op3 = [ atom.get_coord() for atom in res if "OP3" in atom.get_fullname() ]
            atom_op1 = [ atom.get_coord() for atom in res if "OP1" in atom.get_fullname() ]
            atom_op2 = [ atom.get_coord() for atom in res if "OP2" in atom.get_fullname() ]
            atom_o5p= [ atom.get_coord() for atom in res if "O5'" in atom.get_fullname() ]
            atom_c5p = [ atom.get_coord() for atom in res if "C5'" in atom.get_fullname() ]
            atom_c4p = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ]
            atom_o4p = [ atom.get_coord() for atom in res if "O4'" in atom.get_fullname() ]
            atom_c3p = [ atom.get_coord() for atom in res if "C3'" in atom.get_fullname() ]
            atom_o3p = [ atom.get_coord() for atom in res if "O3'" in atom.get_fullname() ]
            atom_c2p = [ atom.get_coord() for atom in res if "C2'" in atom.get_fullname() ]
            atom_o2p = [ atom.get_coord() for atom in res if "O2'" in atom.get_fullname() ]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
            atom_n9 = [ atom.get_coord() for atom in res if "N9" in atom.get_fullname() ]
            atom_c8 = [ atom.get_coord() for atom in res if "C8" in atom.get_fullname() ]
            atom_n7 = [ atom.get_coord() for atom in res if "N7" in atom.get_fullname() ]
            atom_c5 = [ atom.get_coord() for atom in res if atom.get_name() ==  "C5"]
            atom_c6 = [ atom.get_coord() for atom in res if "C6" in atom.get_fullname() ]
            atom_o6 = [ atom.get_coord() for atom in res if "O6" in atom.get_fullname() ]
            atom_n6 = [ atom.get_coord() for atom in res if "N6" in atom.get_fullname() ]
            atom_n1 = [ atom.get_coord() for atom in res if "N1" in atom.get_fullname() ]
            atom_c2 = [ atom.get_coord() for atom in res if atom.get_name() ==  "C2"]
            atom_n2 = [ atom.get_coord() for atom in res if "N2" in atom.get_fullname() ]
            atom_o2 = [ atom.get_coord() for atom in res if atom.get_name() ==  "O2"]
            atom_n3 = [ atom.get_coord() for atom in res if "N3" in atom.get_fullname() ]
            atom_c4 = [ atom.get_coord() for atom in res if atom.get_name() ==  "C4" ]
            atom_n4 = [ atom.get_coord() for atom in res if "N4" in atom.get_fullname() ]
            atom_o4 = [ atom.get_coord() for atom in res if atom.get_name() == "O4"]
            

            if len(atom_op3)<1 or len(atom_p)<1 :#if no atom p or op3 in this chain
                op3_p=np.nan
            else :
                op3_p=distance(atom_op3[0], atom_p[0])

            if len(last_o3p)<1 or len(atom_p)<1 or f != f_prec :#if the file has changed, do not calculate the distance between o3 'of the previous file and p of the current file
                last_o3p_p=None
            else :
                if distance(last_o3p[0], atom_p[0])>3 :
                    last_o3p_p=None
                else :
                    last_o3p_p=distance(last_o3p[0], atom_p[0])#link with the previous nucleotide
            
            if len(atom_op1)<1 or len(atom_p)<1 :
                p_op1=None
            else :
                p_op1=distance(atom_op1[0], atom_p[0])

            if len(atom_op2)<1 or len(atom_p)<1 :
                p_op2=None
            else :
                p_op2=distance(atom_op2[0], atom_p[0])

            if len(atom_o5p)<1 or len(atom_p)<1 :
                p_o5p=None
            else :
                p_o5p=distance(atom_o5p[0], atom_p[0])

            if len(atom_o5p)<1 or len(atom_c5p)<1 :
                o5p_c5p=None
            else :
                o5p_c5p=distance(atom_o5p[0], atom_c5p[0])

            if len(atom_c5p)<1 or len(atom_c4p)<1 :
                c5p_c4p=None
            else :
                c5p_c4p=distance(atom_c5p[0], atom_c4p[0])

            if len(atom_c4p)<1 or len(atom_o4p)<1 :
                c4p_o4p=None
            else :
                c4p_o4p=distance(atom_c4p[0], atom_o4p[0])

            if len(atom_c4p)<1 or len(atom_c3p)<1 :
                c4p_c3p=None
            else :
                c4p_c3p=distance(atom_c4p[0], atom_c3p[0])

            if len(atom_o4p)<1 or len(atom_c1p)<1 :
                o4p_c1p=None
            else :
                o4p_c1p=distance(atom_o4p[0], atom_c1p[0])

            if len(atom_c1p)<1 or len(atom_c2p)<1 :
                c1p_c2p=None
            else :
                c1p_c2p=distance(atom_c1p[0], atom_c2p[0])

            if len(atom_c2p)<1 or len(atom_o2p)<1 :
                c2p_o2p=None
            else :
                c2p_o2p=distance(atom_c2p[0], atom_o2p[0])

            if len(atom_c2p)<1 or len(atom_c3p)<1 :
                c2p_c3p=None
            else :
                c2p_c3p=distance(atom_c2p[0], atom_c3p[0])

            if len(atom_c3p)<1 or len(atom_o3p)<1 :
                c3p_o3p=None
            else :
                c3p_o3p=distance(atom_c3p[0], atom_o3p[0])

            last_o3p=atom_o3p #o3' of this residue becomes the previous o3' of the following
            f_prec=f 
            
            #different cases for the aromatic cycles
            if res.get_resname()=='A' or res.get_resname()=='G': 
                '''
                computes the distances between atoms of aromatic cycles
                '''
                if len(atom_c1p)<1 or len(atom_n9)<1 :
                    c1p_n9=None
                else :
                    c1p_n9=distance(atom_c1p[0], atom_n9[0])

                if len(atom_n9)<1 or len(atom_c8)<1 :
                    n9_c8=None
                else :
                    n9_c8=distance(atom_n9[0], atom_c8[0])

                if len(atom_c8)<1 or len(atom_n7)<1 :
                    c8_n7=None
                else :
                    c8_n7=distance(atom_c8[0], atom_n7[0])

                if len(atom_n7)<1 or len(atom_c5)<1 :
                    n7_c5=None
                else :
                    n7_c5=distance(atom_n7[0], atom_c5[0])

                if len(atom_c5)<1 or len(atom_c6)<1 :
                    c5_c6=None
                else :
                    c5_c6=distance(atom_c5[0], atom_c6[0])

                if len(atom_c6)<1 or len(atom_o6)<1 :
                    c6_o6=None
                else :
                    c6_o6=distance(atom_c6[0], atom_o6[0])

                if len(atom_c6)<1 or len(atom_n6)<1 :
                    c6_n6=None
                else :
                    c6_n6=distance(atom_c6[0], atom_n6[0])

                if len(atom_c6)<1 or len(atom_n1)<1 :
                    c6_n1=None
                else :
                    c6_n1=distance(atom_c6[0], atom_n1[0])

                if len(atom_n1)<1 or len(atom_c2)<1 :
                    n1_c2=None
                else :
                    n1_c2=distance(atom_n1[0], atom_c2[0])
                
                if len(atom_c2)<1 or len(atom_n2)<1 :
                    c2_n2=None
                else :
                    c2_n2=distance(atom_c2[0], atom_n2[0])

                if len(atom_c2)<1 or len(atom_n3)<1 :
                    c2_n3=None
                else :
                    c2_n3=distance(atom_c2[0], atom_n3[0])

                if len(atom_n3)<1 or len(atom_c4)<1 :
                    n3_c4=None
                else :
                    n3_c4=distance(atom_n3[0], atom_c4[0])

                if len(atom_c4)<1 or len(atom_n9)<1 :
                    c4_n9=None
                else :
                    c4_n9=distance(atom_c4[0], atom_n9[0])

                if len(atom_c4)<1 or len(atom_c5)<1 :
                    c4_c5=None
                else :
                    c4_c5=distance(atom_c4[0], atom_c5[0])

            if res.get_resname()=='C' or res.get_resname()=='U' :
                if len(atom_c1p)<1 or len(atom_n1)<1 :
                    c1p_n1=None
                else :
                    c1p_n1=distance(atom_c1p[0], atom_n1[0])

                if len(atom_c6)<1 or len(atom_n1)<1 :
                    n1_c6=None
                else :
                    n1_c6=distance(atom_n1[0], atom_c6[0])
                
                if len(atom_c5)<1 or len(atom_c6)<1 :
                    c6_c5=None
                else :
                    c6_c5=distance(atom_c6[0], atom_c5[0])
                
                if len(atom_c4)<1 or len(atom_c5)<1 :
                    c5_c4=None
                else :
                    c5_c4=distance(atom_c5[0], atom_c4[0])
                
                if len(atom_n3)<1 or len(atom_c4)<1 :
                    c4_n3=None
                else :
                    c4_n3=distance(atom_c4[0], atom_n3[0])
                
                if len(atom_c2)<1 or len(atom_n3)<1 :
                    n3_c2=None
                else :
                    n3_c2=distance(atom_n3[0], atom_c2[0])
                
                if len(atom_c2)<1 or len(atom_o2)<1 :
                    c2_o2=None
                else :
                    c2_o2=distance(atom_c2[0], atom_o2[0])
                
                if len(atom_c2)<1 or len(atom_n1)<1 :
                    c2_n1=None
                else :
                    c2_n1=distance(atom_c2[0], atom_n1[0])

                if len(atom_c4)<1 or len(atom_n4)<1 :
                    c4_n4=None
                else :
                    c4_n4=distance(atom_c4[0], atom_n4[0])

                if len(atom_c4)<1 or len(atom_o4)<1:
                    c4_o4=None
                else :
                    c4_o4=distance(atom_c4[0], atom_o4[0])

            liste_common.append([res.get_resname(), last_o3p_p, op3_p, p_op1, p_op2, p_o5p, o5p_c5p, c5p_c4p, c4p_o4p, c4p_c3p, o4p_c1p, c1p_c2p, c2p_o2p, c2p_c3p, c3p_o3p] )
            liste_purines.append([c1p_n9, n9_c8, c8_n7, n7_c5, c5_c6, c6_o6, c6_n6, c6_n1, n1_c2, c2_n2, c2_n3, n3_c4, c4_n9, c4_c5])
            liste_pyrimidines.append([c1p_n1, n1_c6, c6_c5, c5_c4, c4_n3, n3_c2, c2_o2, c2_n1, c4_n4, c4_o4])
            pbar.update(1)

    df_comm=pd.DataFrame(liste_common, columns=["Residu", "O3'-P", "OP3-P", "P-OP1", "P-OP2", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-O4'", "C4'-C3'", "O4'-C1'", "C1'-C2'", "C2'-O2'", "C2'-C3'", "C3'-O3'"])
    df_pur=pd.DataFrame(liste_purines, columns=["C1'-N9", "N9-C8", "C8-N7", "N7-C5", "C5-C6", "C6-O6", "C6-N6", "C6-N1", "N1-C2", "C2-N2", "C2-N3", "N3-C4", "C4-N9", "C4-C5" ])
    df_pyr=pd.DataFrame(liste_pyrimidines, columns=["C1'-N1", "N1-C6", "C6-C5", "C5-C4", "C4-N3", "N3-C2", "C2-O2", "C2-N1", "C4-N4", "C4-O4"])
    df=pd.concat([df_comm, df_pur, df_pyr], axis = 1)
    pbar.close()
    
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")

    os.makedirs(runDir+"/results/all-atoms/distances/", exist_ok=True)
    df.to_csv(runDir+"/results/all-atoms/distances/" +'dist_atoms '+name+'.csv')


def concatenate(chemin, filename):
    '''
    Concatenates the dataframes of liste containing measures 
    and creates a new dataframe gathering all
    '''
    liste=os.listdir(runDir+chemin)
    df_0=pd.read_csv(os.path.abspath(runDir + chemin + liste[0]))
    del(liste[0])
    df_tot=df_0
    for f in liste:
        df=pd.read_csv(os.path.abspath(runDir + chemin + f))
        df_tot=pd.concat([df_tot, df], ignore_index=True)
    
    df_tot.to_csv(runDir + chemin + filename)

def dist_atoms_hire_RNA (f) :
    '''
    Measures the distance between the atoms of the HiRE-RNA model linked by covalent bonds
    '''
    name=str.split(f,'.')[0]
    liste_dist=[]
    last_c4p=[]
    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} dist_atoms_hire_RNA({f})")

    parser=MMCIFParser()
    s = parser.get_structure(name, os.path.abspath(path_to_3D_data +"rna_only/" + f))
    chain = next(s[0].get_chains())
    residues=list(chain.get_residues())
    pbar = tqdm(total=len(residues), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} dist_atoms_hire_RNA", unit="residu", leave=False)
    pbar.update(0)
    os.makedirs(runDir+"/results/HiRE-RNA/distances/", exist_ok=True)
    for res in chain :
        p_o5p=None
        o5p_c5p=None
        c5p_c4p=None
        c4p_c1p=None
        c1p_b1=None
        b1_b2=None
        last_c4p_p=np.nan
        
        if res.get_resname() not in ['ATP', 'CCC', 'A3P', 'A23', 'GDP', 'RIA'] : #several phosphate groups, ignore
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_o5p= [ atom.get_coord() for atom in res if "O5'" in atom.get_fullname() ]
            atom_c5p = [ atom.get_coord() for atom in res if "C5'" in atom.get_fullname() ]
            atom_c4p = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
            atom_b1=pos_b1(res)#position b1 to be calculated, depending on the case
            atom_b2=pos_b2(res)#position b2 to be calculated only for those with 2 cycles
            

            if len(last_c4p)<1 or len(atom_p)<1 or f!= f_prec:#link with the previous residue in the chain
                last_c4p_p=last_c4p_p
            else :
                if distance(last_c4p[0], atom_p[0])>5:
                    last_c4p_p=last_c4p_p
                else:
                    last_c4p_p=distance(last_c4p[0], atom_p[0])

            if len(atom_p)<1 or len(atom_o5p)<1 :
                p_o5p=p_o5p
            else :
                p_o5p=distance(atom_p[0], atom_o5p[0])
            
            if len(atom_c5p)<1 or len(atom_o5p)<1 :
                o5p_c5p=o5p_c5p
            else :
                o5p_c5p=distance(atom_o5p[0], atom_c5p[0])

            if len(atom_c5p)<1 or len(atom_c4p)<1 :
                c5p_c4p=c5p_c4p
            else :
                c5p_c4p=distance(atom_c5p[0], atom_c4p[0])

            if len(atom_c4p)<1 or len(atom_c1p)<1 :
                c4p_c1p=c4p_c1p
            else :
                c4p_c1p=distance(atom_c4p[0], atom_c1p[0])

            if len(atom_c1p)<1 or len(atom_b1)<1 :
                c1p_b1=c1p_b1
            else :
               
                c1p_b1=distance(atom_c1p[0], atom_b1)

            if len(atom_b1)<1 or len(atom_b2)<1 :
                b1_b2=b1_b2
            else :

                b1_b2=distance(atom_b1, atom_b2)
            
            last_c4p=atom_c4p
            f_prec=f
        

            liste_dist.append([res.get_resname(), last_c4p_p, p_o5p, o5p_c5p, c5p_c4p, c4p_c1p, c1p_b1, b1_b2])
            pbar.update(1)
    df=pd.DataFrame(liste_dist, columns=["Residu", "C4'-P", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-C1'", "C1'-B1", "B1-B2"])
    pbar.close()
    
    df.to_csv(runDir + '/results/HiRE-RNA/distances/' + 'dist_atoms_hire_RNA '+name+'.csv')
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
    
def conversion_angles(bdd): 
    '''
    Convert database torsion angles to degrees
    and put them in a list to reuse for statistics
    '''
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, bdd)
    baseDeDonnees = sqlite3.connect(db_path)
    curseur = baseDeDonnees.cursor()
    curseur.execute("SELECT chain_id, nt_name, alpha, beta, gamma, delta, epsilon, zeta, chi FROM nucleotide WHERE nt_name='A' OR nt_name='C' OR nt_name='G' OR nt_name='U' ;")
    liste=[]
    for nt in curseur.fetchall(): # retrieve the angle measurements and put them in a list
        liste.append(nt)
    angles_torsion=[]
    for nt in liste :
        angles_deg=[]
        angles_deg.append(nt[0]) #chain_id
        angles_deg.append(nt[1]) #nt_name
        for i in range (2,9): # on all angles
            angle=0
            if nt[i] == None : 
                angle=None
            elif nt[i]<=np.pi: #if angle value <pi, positive
                angle=(180/np.pi)*nt[i]
            elif np.pi < nt[i] <= 2*np.pi : #if value of the angle between pi and 2pi, negative
                angle=((180/np.pi)*nt[i])-360
            else :
                angle=nt[i] # dans le cas ou certains angles seraient en degres -> supprimer?
            angles_deg.append(angle)
        angles_torsion.append(angles_deg)
    return angles_torsion

def conversion_eta_theta(bdd):
    '''
    We repeat the operation for the pseudotorsion angles
    '''
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, bdd)
    baseDeDonnees = sqlite3.connect(db_path)
    curseur = baseDeDonnees.cursor()
    curseur.execute("SELECT chain_id, nt_name, eta, theta, eta_prime, theta_prime, eta_base, theta_base FROM nucleotide WHERE nt_name='A' OR nt_name='C' OR nt_name='G' OR nt_name='U';")
    liste=[]
    for nt in curseur.fetchall(): 
        liste.append(nt)
    angles_virtuels=[]
    for nt in liste :
        angles_deg=[]
        angles_deg.append(nt[0]) #chain_id
        angles_deg.append(nt[1]) #nt_name
        for i in range (2,8): 
            angle=0
            if nt[i] == None : 
                angle=None
            elif nt[i]<=np.pi:
                angle=(180/np.pi)*nt[i]
            elif np.pi < nt[i] <= 2*np.pi : 
                angle=((180/np.pi)*nt[i])-360
            else :
                angle=nt[i] 
            angles_deg.append(angle)
        angles_virtuels.append(angles_deg)
    return angles_virtuels

def angles_torsion_hire_RNA(f):
    '''
    Measures the torsion angles between the atoms of the HiRE-RNA model
    Saves the results in a dataframe
    '''
    name=str.split(f,'.')[0]
    liste_angles_torsion=[]

    last_o5p=[]
    last_c4p=[]
    last_c5p=[]
    last_c1p=[]
    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} angles_torsion_hire_RNA({f})")

    os.makedirs(runDir+"/results/HiRE-RNA/torsions/", exist_ok=True)

    parser=MMCIFParser()
    s = parser.get_structure(name, os.path.abspath(path_to_3D_data + "rna_only/" + f))
    chain = next(s[0].get_chains())
    residues=list(chain.get_residues())
    pbar = tqdm(total=len(residues), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} angles_torsion_hire_RNA", unit="residu", leave=False)
    pbar.update(0)

    for res in chain :
        p_o5_c5_c4=np.nan
        o5_c5_c4_c1=np.nan
        c5_c4_c1_b1=np.nan
        c4_c1_b1_b2=np.nan
        o5_c5_c4_psuiv=np.nan
        c5_c4_psuiv_o5suiv=np.nan
        c4_psuiv_o5suiv_c5suiv=np.nan
        c1_c4_psuiv_o5suiv=np.nan
        if res.get_resname() not in ['ATP', 'CCC', 'A3P', 'A23', 'GDP', 'RIA'] : # several phosphate groups
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_o5p= [ atom.get_coord() for atom in res if "O5'" in atom.get_fullname() ]
            atom_c5p = [ atom.get_coord() for atom in res if "C5'" in atom.get_fullname() ]
            atom_c4p = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
            atom_b1=pos_b1(res)
            atom_b2=pos_b2(res)

            if len(atom_p)<1 or len(atom_o5p)<1 or len(atom_c5p)<1 or len(atom_c4p)<1:
                p_o5_c5_c4=p_o5_c5_c4
            else :
                p=Vector(atom_p[0])
                o5p=Vector(atom_o5p[0])
                c5p=Vector(atom_c5p[0])
                c4p=Vector(atom_c4p[0])
                p_o5_c5_c4=calc_dihedral(p, o5p, c5p, c4p)*(180/np.pi)
      
            if len(atom_c1p)<1 or len(atom_o5p)<1 or len(atom_c5p)<1 or len(atom_c4p)<1:
                o5_c5_c4_c1=o5_c5_c4_c1
            else :
                o5p=Vector(atom_o5p[0])
                c5p=Vector(atom_c5p[0])
                c4p=Vector(atom_c4p[0])
                c1p=Vector(atom_c1p[0])
                o5_c5_c4_c1=calc_dihedral(o5p, c5p, c4p, c1p)*(180/np.pi)
  
            if len(atom_c1p)<1 or len(atom_b1)<1 or len(atom_c5p)<1 or len(atom_c4p)<1:
                c5_c4_c1_b1=c5_c4_c1_b1
            else :
                c5p=Vector(atom_c5p[0])
                c4p=Vector(atom_c4p[0])
                c1p=Vector(atom_c1p[0])
                b1=Vector(atom_b1)
                c5_c4_c1_b1=calc_dihedral(c5p, c4p, c1p, b1)*(180/np.pi)

            if len(atom_c1p)<1 or len(atom_b1)<1 or len(atom_b2)<1 or len(atom_c4p)<1:
                c4_c1_b1_b2=c4_c1_b1_b2
            else :
                c4p=Vector(atom_c4p[0])
                c1p=Vector(atom_c1p[0])
                b1=Vector(atom_b1)
                b2=Vector(atom_b2)
                c4_c1_b1_b2=calc_dihedral(c4p, c1p, b1, b2)*(180/np.pi)
            
            if len(last_o5p)<1 or len(atom_p)<1 or len(last_c5p)<1 or len(last_c4p)<1:
                o5_c5_c4_psuiv=o5_c5_c4_psuiv
            else :
                o5p_prec=Vector(last_o5p[0])
                c5p_prec=Vector(last_c5p[0])
                c4p_prec=Vector(last_c4p[0])
                p=Vector(atom_p[0])
                o5_c5_c4_psuiv=calc_dihedral(o5p_prec, c5p_prec, c4p_prec, p)*(180/np.pi)
              
            if len(atom_o5p)<1 or len(atom_p)<1 or len(last_c5p)<1 or len(last_c4p)<1:
                c5_c4_psuiv_o5suiv=c5_c4_psuiv_o5suiv
            else : 
                c5p_prec=Vector(last_c5p[0])
                c4p_prec=Vector(last_c4p[0])
                p=Vector(atom_p[0])
                o5p=Vector(atom_o5p[0])
                c5_c4_psuiv_o5suiv=calc_dihedral(c5p_prec, c4p_prec, p, o5p)*(180/np.pi)
     
            if len(atom_o5p)<1 or len(atom_p)<1 or len(atom_c5p)<1 or len(last_c4p)<1:
                c4_psuiv_o5suiv_c5suiv=c4_psuiv_o5suiv_c5suiv
            else :
                c4p_prec=Vector(last_c4p[0])
                p=Vector(atom_p[0])
                o5p=Vector(atom_o5p[0])
                c5p=Vector(atom_c5p[0])
                c4_psuiv_o5suiv_c5suiv=calc_dihedral(c4p_prec, p, o5p, c5p)*(180/np.pi)
 
            if len(atom_o5p)<1 or len(atom_p)<1 or len(last_c1p)<1 or len(last_c4p)<1:
                c1_c4_psuiv_o5suiv=c1_c4_psuiv_o5suiv
            else :
                c1p_prec=Vector(last_c1p[0])
                c4p_prec=Vector(last_c4p[0])
                p=Vector(atom_p[0])
                o5p=Vector(atom_o5p[0])
                c1_c4_psuiv_o5suiv=calc_dihedral(c1p_prec, c4p_prec, p, o5p)*(180/np.pi)
            last_o5p=atom_o5p
            last_c4p=atom_c4p
            last_c5p=atom_c5p
            last_c1p=atom_c1p
            liste_angles_torsion.append([res.get_resname(), p_o5_c5_c4, o5_c5_c4_c1, c5_c4_c1_b1, c4_c1_b1_b2, o5_c5_c4_psuiv, c5_c4_psuiv_o5suiv, c4_psuiv_o5suiv_c5suiv, c1_c4_psuiv_o5suiv])
            pbar.update(1)
    df=pd.DataFrame(liste_angles_torsion, columns=["Residu", "P-O5'-C5'-C4'", "O5'-C5'-C4'-C1'", "C5'-C4'-C1'-B1", "C4'-C1'-B1-B2", "O5'-C5'-C4'-P°", "C5'-C4'-P°-O5'°", "C4'-P°-O5'°-C5'°", "C1'-C4'-P°-O5'°"])
    pbar.close()

    df.to_csv(runDir + '/results/HiRE-RNA/torsions/' + 'angles_torsion_hire_RNA '+name+'.csv')
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")

def angles_plans_hire_RNA(f):
    '''
    Measures the plane angles involving C1' and B1 atoms 
    Saves the results in a dataframe
    '''
    name=str.split(f,'.')[0]
    liste_angles_plans=[]
    last_p=[]
    last_c1p=[]

    global idxQueue
    thr_idx = idxQueue.get()

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} angles_plans_hire_RNA({f})")

    os.makedirs(runDir+"/results/HiRE-RNA/angles/", exist_ok=True)

    parser=MMCIFParser()
    s = parser.get_structure(name, os.path.abspath(path_to_3D_data + "rna_only/" + f))
    chain = next(s[0].get_chains())
    residues=list(chain.get_residues())
    pbar = tqdm(total=len(residues), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} angles_torsion_hire_RNA", unit="residu", leave=False)
    pbar.update(0)
    for res in chain :
        p_c1p_psuiv=np.nan
        c1p_psuiv_c1psuiv=np.nan
        if res.get_resname() not in ['ATP', 'CCC', 'A3P', 'A23', 'GDP', 'RIA'] :
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]

            if len(last_p)<1 or len(last_c1p)<1 or len(atom_p)<1 :
                p_c1p_psuiv=p_c1p_psuiv
            else :
                p_prec=Vector(last_p[0])
                c1p_prec=Vector(last_c1p[0])
                p=Vector(atom_p[0])
                p_c1p_psuiv=calc_angle(p_prec, c1p_prec, p)*(180/np.pi)

            if len(atom_c1p)<1 or len(last_c1p)<1 or len(atom_p)<1:
                c1p_psuiv_c1psuiv=c1p_psuiv_c1psuiv
            else :
                c1p_prec=Vector(last_c1p[0])
                p=Vector(atom_p[0])
                c1p=Vector(atom_c1p[0])
                c1p_psuiv_c1psuiv=calc_angle(c1p_prec, p, c1p)*(180/np.pi)

            last_p=atom_p
            last_c1p=atom_c1p
            liste_angles_plans.append([res.get_resname(), p_c1p_psuiv, c1p_psuiv_c1psuiv])
            pbar.update(1)
    df=pd.DataFrame(liste_angles_plans, columns=["Residu", "P-C1'-P°", "C1'-P°-C1'°"])
    pbar.close()
    
    
    df.to_csv(runDir + '/results/HiRE-RNA/angles/' + 'angles_plans_hire_RNA '+name+'.csv')
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")
    
def histogram(data, name_data, x, y, nb):
    '''
    Plot histograms
    '''
    
    plt.hist(data,color="green",edgecolor='black', linewidth=1.2,bins=50, density=True)
    plt.xlabel(x)
    plt.ylabel(y)

def GMM_histo(data, name_data, x, y) :
    '''
    Plot Gaussian-Mixture-Model on histograms
    '''
    histogram(data, name_data, x, y, len(data))#plot the histogram
    
    n_max = 8    # number of possible values for n_components
    n_components_range = np.arange(n_max)+1
    aic = []
    bic = []
    maxlogv=[]
    md=np.array(data).reshape(-1,1)
    # construction of models and calculation of criteria
    nb_components=1
    nb_log_max=n_components_range[0]
    log_max=0
    # chooses the number of components based on the maximum likelihood value (maxlogv)
    for n_comp in n_components_range:
        gmm = GaussianMixture(n_components=n_comp).fit(md)
        aic.append(abs(gmm.aic(md)))
        bic.append(abs(gmm.bic(md)))
        maxlogv.append(gmm.lower_bound_)
        if gmm.lower_bound_== max(maxlogv) : # takes the maximum
            nb_components=n_comp
            # if there is convergence, keep the first maximum found
            if abs(gmm.lower_bound_-log_max)<0.02 : #threshold=0.02
                nb_components=nb_log_max
                break
        log_max=max(maxlogv)
        nb_log_max=n_comp

    # plot with the appropriate number of components
    obs=np.array(data).reshape(-1,1)
    g = GaussianMixture(n_components=nb_components)
    g.fit(obs)
    weights = g.weights_
    means = g.means_
    covariances = g.covariances_

    D = obs.ravel()
    xmin = D.min()
    xmax = D.max()
    x = np.linspace(xmin,xmax,1000)
    colors=['red', 'blue', 'gold', 'cyan', 'magenta', 'white', 'black', 'green']
    # prepare the dictionary to save the parameters
    summary_data={}
    summary_data["measure"]= name_data
    summary_data["weights"]=[]
    summary_data["means"]=[]
    summary_data["std"]=[]
    # plot
    for i in range(nb_components):
        mean = means[i]
        sigma = np.sqrt(covariances[i])
        weight = weights[i]
        plt.plot(x,(weights[i]*st.norm.pdf(x,mean,sigma))[0], c=colors[i])
        summary_data["means"].append(str(mean))
        summary_data["std"].append(str(sigma))
        summary_data["weights"].append(str(weight))
    axes=plt.gca()
    plt.title("Histogramme " +name_data+ " avec GMM pour " +str(nb_components)+ " composantes (" + str(len(data))+" valeurs)")
    plt.savefig("Histogramme " +name_data+ " avec GMM pour " +str(nb_components)+ " composantes (" + str(len(data))+" valeurs).png")
    plt.close()
    # save in a json
    with open (name_data + " .json", 'w', encoding='utf-8') as f:
	    json.dump(summary_data, f, indent=4)

def GMM_histo_without_saving(data, name_data, x, y) :
    '''
    Plot Gaussian-Mixture-Model on histograms
    But without saving and close the figure
    '''
    histogram(data, name_data, x, y, len(data))#plot the histogram
    
    n_max = 8    # number of possible values for n_components
    n_components_range = np.arange(n_max)+1
    aic = []
    bic = []
    maxlogv=[]
    md=np.array(data).reshape(-1,1)
    # construction of models and calculation of criteria
    nb_components=1
    nb_log_max=n_components_range[0]
    log_max=0
    # chooses the number of components based on the maximum likelihood value (maxlogv)
    for n_comp in n_components_range:
        gmm = GaussianMixture(n_components=n_comp).fit(md)
        aic.append(abs(gmm.aic(md)))
        bic.append(abs(gmm.bic(md)))
        maxlogv.append(gmm.lower_bound_)
        if gmm.lower_bound_== max(maxlogv) : # takes the maximum
            nb_components=n_comp
            # if there is convergence, keep the first maximum found
            if abs(gmm.lower_bound_-log_max)<0.02 : #threshold=0.02
                nb_components=nb_log_max
                break
        log_max=max(maxlogv)
        nb_log_max=n_comp

    # plot with the appropriate number of components
    obs=np.array(data).reshape(-1,1)
    g = GaussianMixture(n_components=nb_components)
    g.fit(obs)
    weights = g.weights_
    means = g.means_
    covariances = g.covariances_

    D = obs.ravel()
    xmin = D.min()
    xmax = D.max()
    x = np.linspace(xmin,xmax,1000)
    colors=['red', 'blue', 'gold', 'cyan', 'magenta', 'white', 'black', 'green']
    # prepare the dictionary to save the parameters
    summary_data={}
    summary_data["measure"]= name_data
    summary_data["weights"]=[]
    summary_data["means"]=[]
    summary_data["std"]=[]
    # plot
    for i in range(nb_components):
        mean = means[i]
        sigma = np.sqrt(covariances[i])
        weight = weights[i]
        plt.plot(x,(weights[i]*st.norm.pdf(x,mean,sigma))[0], c=colors[i])
        summary_data["means"].append(str(mean))
        summary_data["std"].append(str(sigma))
        summary_data["weights"].append(str(weight))
    axes=plt.gca()
    plt.title("Histogramme " +name_data+ " avec GMM pour " +str(nb_components)+ " composantes (" + str(len(data))+" structures)")
    # save in a json
    with open (name_data + " .json", 'w', encoding='utf-8') as f:
	    json.dump(summary_data, f, indent=4)

def GMM_tot(data, name_data, couleur) :
    '''
    Plot the sum of the Gaussians (without the histograms)
    '''
    
    n_max = 8    # number of possible values for n_components
    n_components_range = np.arange(n_max)+1
    aic = []
    bic = []
    maxlogv=[]
    md=np.array(data).reshape(-1,1)
    # construction of models and calculation of criteria
    nb_components=1
    nb_log_max=n_components_range[0]
    log_max=0
    for n_comp in n_components_range:
        gmm = GaussianMixture(n_components=n_comp).fit(md)
        aic.append(abs(gmm.aic(md)))
        bic.append(abs(gmm.bic(md)))
        maxlogv.append(gmm.lower_bound_)
        if gmm.lower_bound_== max(maxlogv) :
            nb_components=n_comp
            if abs(gmm.lower_bound_-log_max)<0.02 :
                nb_components=nb_log_max
                break
        log_max=max(maxlogv)
        nb_log_max=n_comp

    obs=np.array(data).reshape(-1,1)
    g = GaussianMixture(n_components=nb_components)
    g.fit(obs)
    weights = g.weights_
    means = g.means_
    covariances = g.covariances_

    D = obs.ravel()
    xmin = D.min()
    xmax = D.max()
    x = np.linspace(xmin,xmax,1000)
    summary_data={}
    summary_data["measure"]= name_data
    summary_data["means"]=[]
    summary_data["std"]=[]
    courbes=[]

    for i in range(nb_components):
        mean = means[i]
        sigma = np.sqrt(covariances[i])
        courbes.append((weights[i]*st.norm.pdf(x,mean,sigma))[0])
        summary_data["means"].append(str(mean))
        summary_data["std"].append(str(sigma))
    axes=plt.gca()
    os.makedirs(runDir+"/results/figures/GMM/", exist_ok=True)
    plt.plot(x, sum(courbes), c=couleur, label=name_data)
    plt.ylabel("Densité")
    plt.legend()

def graph_dist_atoms():
    '''
    Draw the figures representing the data on the measurements of distances between atoms
    '''

    df=pd.read_csv(os.path.abspath(runDir + "/results/all-atoms/distances/dist_atoms.csv"))

    last_o3p_p=list(df["O3'-P"][~ np.isnan(df["O3'-P"])])
    #print(last_o3p_p)
    op3_p=list(df["OP3-P"][~ np.isnan(df["OP3-P"])])
    p_op1=list(df["P-OP1"][~ np.isnan(df["P-OP1"])])
    p_op2=list(df["P-OP2"][~ np.isnan(df["P-OP2"])])
    p_o5p=list(df["P-O5'"][~ np.isnan(df["P-O5'"])])
    o5p_c5p=list(df["O5'-C5'"][~ np.isnan(df["O5'-C5'"])])
    c5p_c4p=list(df["C5'-C4'"][~ np.isnan(df["C5'-C4'"])])
    c4p_o4p=list(df["C4'-O4'"][~ np.isnan(df["C4'-O4'"])])
    o4p_c1p=list(df["O4'-C1'"][~ np.isnan(df["O4'-C1'"])])
    c1p_c2p=list(df["C1'-C2'"][~ np.isnan(df["C1'-C2'"])])
    c2p_o2p=list(df["C2'-O2'"][~ np.isnan(df["C2'-O2'"])])
    c2p_c3p=list(df["C2'-C3'"][~ np.isnan(df["C2'-C3'"])])
    c3p_o3p=list(df["C3'-O3'"][~ np.isnan(df["C3'-O3'"])])
    c4p_c3p=list(df["C4'-C3'"][~ np.isnan(df["C4'-C3'"])])
    
    #if res = A ou G
    c1p_n9=list(df["C1'-N9"][~ np.isnan(df["C1'-N9"])])
    n9_c8=list(df["N9-C8"][~ np.isnan(df["N9-C8"])])
    c8_n7=list(df["C8-N7"][~ np.isnan(df["C8-N7"])])
    n7_c5=list(df["N7-C5"][~ np.isnan(df["N7-C5"])])
    c5_c6=list(df["C5-C6"][~ np.isnan(df["C5-C6"])])
    c6_n1=list(df["C6-N1"][~ np.isnan(df["C6-N1"])])
    n1_c2=list(df["N1-C2"][~ np.isnan(df["N1-C2"])])
    c2_n3=list(df["C2-N3"][~ np.isnan(df["C2-N3"])])
    n3_c4=list(df["N3-C4"][~ np.isnan(df["N3-C4"])])
    c4_n9=list(df["C4-N9"][~ np.isnan(df["C4-N9"])])
    c4_c5=list(df["C4-C5"][~ np.isnan(df["C4-C5"])])
    #if res=G
    c6_o6=list(df["C6-O6"][~ np.isnan(df["C6-O6"])])
    c2_n2=list(df["C2-N2"][~ np.isnan(df["C2-N2"])])
    #if res = A
    c6_n6=list(df["C6-N6"][~ np.isnan(df["C6-N6"])])

    #if res = C ou U
    c1p_n1=list(df["C1'-N1"][~ np.isnan(df["C1'-N1"])])
    n1_c6=list(df["N1-C6"][~ np.isnan(df["N1-C6"])])
    c6_c5=list(df["C6-C5"][~ np.isnan(df["C6-C5"])])
    c5_c4=list(df["C5-C4"][~ np.isnan(df["C5-C4"])])
    c4_n3=list(df["C4-N3"][~ np.isnan(df["C4-N3"])])
    n3_c2=list(df["N3-C2"][~ np.isnan(df["N3-C2"])])
    c2_n1=list(df["C2-N1"][~ np.isnan(df["C2-N1"])])
    c2_o2=list(df["C2-O2"][~ np.isnan(df["C2-O2"])])
    #if res =C
    c4_n4=list(df["C4-N4"][~ np.isnan(df["C4-N4"])])
    #if res=U
    c4_o4=list(df["C4-O4"][~ np.isnan(df["C4-O4"])])

    os.makedirs(runDir+"/results/figures/all-atoms/distances/commun/", exist_ok=True)
    os.chdir(runDir+"/results/figures/all-atoms/distances/commun/")
    # draw figures for atoms common to all nucleotides
    GMM_histo(last_o3p_p, "O3'-P", "Distance(Angström)", "Densité")
    if len(op3_p) > 0 :
        GMM_histo(op3_p, "OP3-P", "Distance(Angström)", "Densité")
    GMM_histo(p_op1, "P-OP1", "Distance(Angström)", "Densité")
    GMM_histo(p_op2, "P-OP2", "Distance(Angström)", "Densité")
    
    GMM_histo(p_o5p, "P-O5'", "Distance(Angström)", "Densité")
    GMM_histo(o5p_c5p, "O5'-C5'", "Distance(Angström)", "Densité")
    GMM_histo(c5p_c4p, "C5'-C4'", "Distance(Angström)", "Densité")
    GMM_histo(c4p_o4p, "C4'-O4'", "Distance(Angström)", "Densité")
    GMM_histo(c4p_c3p, "C4'-C3'", "Distance(Angström)", "Densité")
    GMM_histo(c3p_o3p, "C3'-O3'", "Distance(Angström)", "Densité")
    GMM_histo(o4p_c1p, "O4'-C1'", "Distance(Angström)", "Densité")
    GMM_histo(c1p_c2p, "C1'-C2'", "Distance(Angström)", "Densité")
    GMM_histo(c2p_c3p, "C2'-C3'", "Distance(Angström)", "Densité")
    GMM_histo(c2p_o2p, "C2'-O2'", "Distance(Angström)", "Densité")

    if len(op3_p) > 0 :
        GMM_tot(op3_p, "OP3-P", 'lightcoral')
    GMM_tot(p_op1, "P-OP1", 'gold')
    GMM_tot(p_op2, "P-OP2", 'lightseagreen')
    GMM_tot(last_o3p_p, "O3'-P", 'saddlebrown')
    GMM_tot(p_o5p, "P-O5'", 'darkturquoise')
    GMM_tot(o5p_c5p, "O5'-C5'", 'darkkhaki')
    GMM_tot(c5p_c4p, "C5'-C4'", 'indigo')
    GMM_tot(c4p_o4p, "C4'-O4'", 'maroon')
    GMM_tot(c4p_c3p, "C4'-C3'", 'burlywood')
    GMM_tot(c3p_o3p, "C3'-O3'", 'steelblue')
    GMM_tot(o4p_c1p, "O4'-C1'", 'tomato')
    GMM_tot(c1p_c2p, "C1'-C2'", 'darkolivegreen')
    GMM_tot(c2p_c3p, "C2'-C3'", 'orchid')
    GMM_tot(c2p_o2p, "C2'-O2'", 'deeppink')
    axes=plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angström)")
    plt.title("GMM des distances entre atomes communs ")
    plt.savefig(runDir + "/results/figures/all-atoms/distances/commun/" + "GMM des distances entre atomes communs .png")
    plt.close()

    os.makedirs(runDir+"/results/figures/all-atoms/distances/purines/", exist_ok=True)
    os.chdir(runDir+"/results/figures/all-atoms/distances/purines/")
    # purines
    GMM_histo(c1p_n9, "C1'-N9", "Distance(Angström)", "Densité")
    GMM_histo(n9_c8, "N9-C8", "Distance(Angström)", "Densité")
    GMM_histo(c8_n7, "C8-N7", "Distance(Angström)", "Densité")
    GMM_histo(n7_c5, "N7-C5", "Distance(Angström)", "Densité")
    GMM_histo(c5_c6, "C5-C6", "Distance(Angström)", "Densité")
    GMM_histo(c6_o6, "C6-O6", "Distance(Angström)", "Densité")
    GMM_histo(c6_n6, "C6-N6", "Distance(Angström)", "Densité")
    GMM_histo(c6_n1, "C6-N1", "Distance(Angström)", "Densité")
    GMM_histo(n1_c2, "N1-C2", "Distance(Angström)", "Densité")
    GMM_histo(c2_n2, "C2-N2", "Distance(Angström)", "Densité")
    GMM_histo(c2_n3, "C2-N3", "Distance(Angström)", "Densité")
    GMM_histo(n3_c4, "N3-C4", "Distance(Angström)", "Densité")
    GMM_histo(c4_n9, "C4-N9", "Distance(Angström)", "Densité")
    GMM_histo(c4_c5, "C4-C5", "Distance(Angström)", "Densité")

    GMM_tot(c1p_n9, "C1'-N9", 'lightcoral')
    GMM_tot(n9_c8, "N9-C8", 'gold')
    GMM_tot(c8_n7, "C8-N7", 'lightseagreen')
    GMM_tot(n7_c5, "N7-C5", 'saddlebrown')
    GMM_tot(c5_c6, "C5-C6", 'darkturquoise')
    GMM_tot(c6_o6, "C6-O6", 'darkkhaki')
    GMM_tot(c6_n6, "C6-N6", 'indigo')
    GMM_tot(c6_n1, "C6-N1", 'maroon')
    GMM_tot(n1_c2, "N1-C2", 'burlywood')
    GMM_tot(c2_n2, "C2-N2", 'steelblue')
    GMM_tot(c2_n3, "C2-N3", 'tomato')
    GMM_tot(n3_c4, "N3-C4", 'darkolivegreen')
    GMM_tot(c4_n9, "C4-N9", 'orchid')
    GMM_tot(c4_c5, "C4-C5", 'deeppink')
    axes=plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angström)")
    plt.title("GMM des distances entre atomes des cycles purines", fontsize=10)
    plt.savefig(runDir+ "/results/figures/all-atoms/distances/purines/" + "GMM des distances entre atomes des cycles purines.png")
    plt.close()

    os.makedirs(runDir+"/results/figures/all-atoms/distances/pyrimidines/", exist_ok=True)
    os.chdir(runDir+"/results/figures/all-atoms/distances/pyrimidines/")
    # pyrimidines

    GMM_histo(c1p_n1, "C1'-N1", "Distance(Angström)", "Densité")
    GMM_histo(n1_c6, "N1-C6", "Distance(Angström)", "Densité")
    GMM_histo(c6_c5, "C6-C5", "Distance(Angström)", "Densité")
    GMM_histo(c5_c4, "C5-C4", "Distance(Angström)", "Densité")
    GMM_histo(c4_n3, "C4-N3", "Distance(Angström)", "Densité")
    GMM_histo(n3_c2, "N3-C2", "Distance(Angström)", "Densité")
    GMM_histo(c2_o2, "C2-O2", "Distance(Angström)", "Densité")
    GMM_histo(c2_n1, "C2-N1", "Distance(Angström)", "Densité")
    GMM_histo(c4_n4, "C4-N4", "Distance(Angström)", "Densité")
    GMM_histo(c4_o4, "C4-O4", "Distance(Angström)", "Densité")

    GMM_tot(c1p_n1, "C1'-N1", 'lightcoral')
    GMM_tot(n1_c6, "N1-C6", 'gold')
    GMM_tot(c6_c5, "C6-C5", 'lightseagreen')
    GMM_tot(c5_c4, "C5-C4", 'deeppink')
    GMM_tot(c4_n3, "C4-N3", 'red')
    GMM_tot(n3_c2, "N3-C2", 'lime')
    GMM_tot(c2_o2, "C2-O2", 'indigo')
    GMM_tot(c2_n1, "C2-N1", 'maroon')
    GMM_tot(c4_n4, "C4-N4", 'burlywood')
    GMM_tot(c4_o4, "C4-O4", 'steelblue')
    axes=plt.gca()
    #axes.set_xlim(1, 2)
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angström)")
    plt.title("GMM des distances entre atomes des cycles pyrimidines", fontsize=10)
    plt.savefig(runDir + "/results/figures/all-atoms/distances/pyrimidines/" + "GMM des distances entre atomes des cycles pyrimidines.png")
    plt.close()
    
def graph_dist_atoms_h_RNA():
    '''
    Draw the figures representing the data on the measurements of distances between atoms of the HiRE-RNA model
    '''
    df=pd.read_csv(os.path.abspath(runDir + "/results/HiRE-RNA/distances/dist_atoms_hire_RNA.csv"))  

    last_c4p_p=list(df["C4'-P"][~ np.isnan(df["C4'-P"])])
    p_o5p=list(df["P-O5'"][~ np.isnan(df["P-O5'"])])
    o5p_c5p=list(df["O5'-C5'"][~ np.isnan(df["O5'-C5'"])])
    c5p_c4p=list(df["C5'-C4'"][~ np.isnan(df["C5'-C4'"])])
    c4p_c1p=list(df["C4'-C1'"][~ np.isnan(df["C4'-C1'"])])
    c1p_b1=list(df["C1'-B1"][~ np.isnan(df["C1'-B1"])])
    b1_b2=list(df["B1-B2"][~ np.isnan(df["B1-B2"])])

    os.makedirs(runDir + "/results/figures/HiRE-RNA/distances/", exist_ok=True)
    os.chdir(runDir + "/results/figures/HiRE-RNA/distances/")


    GMM_histo(o5p_c5p, "O5'-C5'", "Distance(Angström)", "Densité")
    GMM_histo(b1_b2, "B1-B2", "Distance(Angström)", "Densité")
    GMM_histo(c1p_b1, "C1'-B1", "Distance(Angström)", "Densité")
    GMM_histo(c5p_c4p, "C5'-C4'", "Distance(Angström)", "Densité")
    GMM_histo(c4p_c1p, "C4'-C1'", "Distance(Angström)", "Densité")
    GMM_histo(p_o5p, "P-O5'", "Distance(Angström)", "Densité")
    GMM_histo(last_c4p_p, "C4'-P", "Distance(Angström)", "Densité")
    
    GMM_tot(o5p_c5p, "O5'-C5'", 'lightcoral')
    GMM_tot(b1_b2, "B1-B2", 'limegreen')
    GMM_tot(c1p_b1, "C1'-B1", 'tomato')
    GMM_tot(c5p_c4p, "C5'-C4'", 'aquamarine')
    GMM_tot(c4p_c1p, "C4'-C1'", 'goldenrod')
    GMM_tot(p_o5p, "P-O5'", 'darkcyan')
    GMM_tot(last_c4p_p, "C4'-P", 'deeppink')
    axes=plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angström)")
    plt.title("GMM des distances entre atomes HiRE-RNA")
    plt.savefig(runDir + "/results/figures/HiRE-RNA/distances/" + "GMM des distances entre atomes HiRE-RNA.png")
    plt.close()

def graph_angles_torsion():
    '''
    Separates the torsion angle measurements by angle type and plots the figures representing the data
    '''
    # we create lists to store the values ​​of each angle
    alpha=[]
    beta=[]
    gamma=[]
    delta=[]
    epsilon=[]
    zeta=[]
    chi = []
    for angles_deg in conversion_angles("/home/atabot/RNANet.db") : #chemin à modifier 
        alpha.append(angles_deg[2])
        beta.append(angles_deg[3])
        gamma.append(angles_deg[4])
        delta.append(angles_deg[5])
        epsilon.append(angles_deg[6])
        zeta.append(angles_deg[7])
        chi.append(angles_deg[8])

    # we remove the null values
    alpha=[i for i in alpha if i != None]
    beta=[i for i in beta if i != None]
    gamma=[i for i in gamma if i != None]
    delta=[i for i in delta if i != None]
    epsilon=[i for i in epsilon if i != None]
    zeta=[i for i in zeta if i != None]
    chi=[i for i in chi if i != None]

    os.makedirs(runDir + "/results/figures/all-atoms/torsions/", exist_ok=True)
    os.chdir(runDir + "/results/figures/all-atoms/torsions/")

    '''
    We plot the GMMs with histogram for each angle
    We create the corresponding json with the means and standard deviations of each Gaussian
    We draw the figure grouping the GMMs of all angles without histogram to compare them with each other
    '''

    GMM_histo(alpha, "Alpha", "Angle(degré)", "Densité")
    GMM_histo(beta, "Beta", "Angle(degré)", "Densité")
    GMM_histo(gamma, "Gamma", "Angle(degré)", "Densité")
    GMM_histo(delta, "Delta", "Angle(degré)", "Densité")
    GMM_histo(epsilon, "Epsilon", "Angle(degré)", "Densité")
    GMM_histo(zeta, "Zeta", "Angle(degré)", "Densité")
    GMM_histo(chi, "Xhi", "Angle(degré)", "Densité")

    GMM_tot(alpha, "Alpha", 'red')
    GMM_tot(beta, "Beta", 'firebrick')
    GMM_tot(gamma, "Gamma", 'limegreen')
    GMM_tot(delta, "Delta", 'darkslateblue')
    GMM_tot(epsilon, "Epsilon", 'goldenrod')
    GMM_tot(zeta, "Zeta", 'teal')
    GMM_tot(chi, "Xhi", 'hotpink')
    plt.xlabel("Angle(Degré)")
    plt.title("GMM des angles de torsion")
    plt.savefig("GMM des angles de torsion.png")
    plt.close()

def graph_eta_theta():
    '''
    Separates the pseudotorsion angle measurements by angle type and plots the figures representing the data
    '''
    eta=[]
    theta=[]
    eta_prime=[]
    theta_prime=[]
    eta_base=[]
    theta_base=[]

    for angles_deg in conversion_eta_theta(runDir + "/results/RNANet.db"): 
        eta.append(angles_deg[2])
        theta.append(angles_deg[3])
        eta_prime.append(angles_deg[4])
        theta_prime.append(angles_deg[5])
        eta_base.append(angles_deg[6])
        theta_base.append(angles_deg[7])

    eta=[i for i in eta if i != None]
    theta=[i for i in theta if i != None]
    eta_prime=[i for i in eta_prime if i != None]
    theta_prime=[i for i in theta_prime if i != None]
    eta_base=[i for i in eta_base if i != None]
    theta_base=[i for i in theta_base if i != None]


    os.makedirs(runDir + "/results/figures/Pyle/pseudotorsions/", exist_ok=True)
    os.chdir(runDir + "/results/figures/Pyle/pseudotorsions/")

    GMM_histo(eta, "Eta", "Angle(degré)", "Densité")
    GMM_histo(theta, "Theta", "Angle(degré)", "Densité")
    GMM_histo(eta_prime, "Eta'", "Angle(degré)", "Densité")
    GMM_histo(theta_prime, "Theta'", "Angle(degré)", "Densité")
    GMM_histo(eta_base, "Eta''", "Angle(degré)", "Densité")
    GMM_histo(theta_base, "Theta''", "Angle(degré)", "Densité")

    GMM_tot(eta, "Eta", 'mediumaquamarine')
    GMM_tot(theta, "Theta", 'darkorchid')
    GMM_tot(eta_prime, "Eta'", 'cyan')
    GMM_tot(theta_prime, "Theta'", 'crimson')
    GMM_tot(eta_base, "Eta''", 'royalblue')
    GMM_tot(theta_base, "Theta''", 'palevioletred')
    plt.xlabel("Angle(Degré)")
    plt.title("GMM des angles de pseudotorsion")
    plt.savefig("GMM des angles de pseudotorsion.png")
    plt.close()

def graph_torsion_h_RNA():
    
    df=pd.read_csv(os.path.abspath(runDir + "/results/HiRE-RNA/torsions/angles_torsion_hire_RNA.csv"))  

    p_o5_c5_c4=list(df["P-O5'-C5'-C4'"][~ np.isnan(df["P-O5'-C5'-C4'"])])
    o5_c5_c4_c1=list(df["O5'-C5'-C4'-C1'"][~ np.isnan(df["O5'-C5'-C4'-C1'"])])
    c5_c4_c1_b1=list(df["C5'-C4'-C1'-B1"][~ np.isnan(df["C5'-C4'-C1'-B1"])])
    c4_c1_b1_b2=list(df["C4'-C1'-B1-B2"][~ np.isnan(df["C4'-C1'-B1-B2"])])
    o5_c5_c4_psuiv=list(df["O5'-C5'-C4'-P°"][~ np.isnan(df["O5'-C5'-C4'-P°"])])
    c5_c4_psuiv_o5suiv=list(df["C5'-C4'-P°-O5'°"][~ np.isnan(df["C5'-C4'-P°-O5'°"])])
    c4_psuiv_o5suiv_c5suiv=list(df["C4'-P°-O5'°-C5'°"][~ np.isnan(df["C4'-P°-O5'°-C5'°"])])
    c1_c4_psuiv_o5suiv=list(df["C1'-C4'-P°-O5'°"][~ np.isnan(df["C1'-C4'-P°-O5'°"])])

    os.makedirs(runDir + "/results/figures/HiRE-RNA/torsions/", exist_ok=True)
    os.chdir(runDir + "/results/figures/HiRE-RNA/torsions/")

    GMM_histo(p_o5_c5_c4, "P-O5'-C5'-C4'", "Angle(Degré)", "Densité")
    GMM_histo(o5_c5_c4_c1, "O5'-C5'-C4'-C1'", "Angle(Degré)", "Densité")
    GMM_histo(c5_c4_c1_b1, "C5'-C4'-C1'-B1", "Angle(Degré)", "Densité")
    GMM_histo(c4_c1_b1_b2, "C4'-C1'-B1-B2", "Angle(Degré)", "Densité")
    GMM_histo(o5_c5_c4_psuiv, "O5'-C5'-C4'-P°", "Angle(Degré)", "Densité")
    GMM_histo(c5_c4_psuiv_o5suiv, "C5'-C4'-P°-O5'°", "Angle(Degré)", "Densité")
    GMM_histo(c4_psuiv_o5suiv_c5suiv, "C4'-P°-O5'°-C5'°", "Angle(Degré)", "Densité")
    GMM_histo(c1_c4_psuiv_o5suiv, "C1'-C4'-P°-O5'°", "Angle(Degré)", "Densité")

    GMM_tot(p_o5_c5_c4, "P-O5'-C5'-C4'", 'darkred')
    GMM_tot(o5_c5_c4_c1, "O5'-C5'-C4'-C1'", 'chocolate')
    GMM_tot(c5_c4_c1_b1, "C5'-C4'-C1'-B1", 'mediumvioletred')
    GMM_tot(c4_c1_b1_b2, "C4'-C1'-B1-B2", 'cadetblue')
    GMM_tot(o5_c5_c4_psuiv, "O5'-C5'-C4'-P°", 'darkkhaki')
    GMM_tot(c5_c4_psuiv_o5suiv, "C5'-C4'-P°-O5'°", 'springgreen')
    GMM_tot(c4_psuiv_o5suiv_c5suiv, "C4'-P°-O5'°-C5'°", 'indigo')
    GMM_tot(c1_c4_psuiv_o5suiv, "C1'-C4'-P°-O5'°", 'gold')
    plt.xlabel("Angle(Degré)")
    plt.title("GMM des angles de torsion (hire-RNA)")
    plt.savefig("GMM des angles de torsion (hire-RNA).png")
    plt.close()

def graph_plans_h_RNA():

    df=pd.read_csv(os.path.abspath(runDir + "/results/HiRE-RNA/angles/angles_plans_hire_RNA.csv"))  

    p_c1p_psuiv=list(df["P-C1'-P°"][~ np.isnan(df["P-C1'-P°"])])
    c1p_psuiv_c1psuiv=list(df["C1'-P°-C1'°"][~ np.isnan(df["C1'-P°-C1'°"])])


    os.makedirs(runDir + "/results/figures/Pyle/angles/", exist_ok=True)
    os.chdir(runDir + "/results/figures/Pyle/angles/")

    GMM_histo(p_c1p_psuiv, "P-C1'-P°", "Angle(Degré)", "Densité")
    GMM_histo(c1p_psuiv_c1psuiv, "C1'-P°-C1'°", "Angle(Degré)", "Densité")

    GMM_tot(p_c1p_psuiv, "P-C1'-P°", 'firebrick')
    GMM_tot(c1p_psuiv_c1psuiv, "C1'-P°-C1'°", 'seagreen')
    plt.xlabel("Angle(Degré)")
    plt.title("GMM des angles plans (hire-RNA)")
    plt.savefig("GMM des angles plans (hire-RNA).png")
    plt.close()

'''
Functions for making measurements on pairings
'''

def dictio(ld):
    '''
    creation of a dictionary
    key = pdb identifier of the structure
    value = list of RNA chains
    '''
    dictionnaire=dict()
    pdb=[]
    for f in ld:
        pdb_id=str.split(f, '_')[0]
        if pdb_id not in pdb: #we create a list of distinct structures
            pdb.append(pdb_id)
    for pdb_id in pdb:#for all structures found
        liste_chaines=[]
        for f in ld:
            if (len(f)<10): #unmapped to a Rfam family
                chaine=str.split(f, '_')
                if chaine[0]==pdb_id:
                    id_chain=chaine[2]
                    liste_chaines.append(id_chain)
        if liste_chaines != []:
            dictionnaire[pdb_id]=liste_chaines
    return (dictionnaire)

def dist_pointes(res, pair):
    '''
    measure of the distance between the tips of the paired nucleotides (B1 / B1 or B1 / B2 or B2 / B2)
    '''
    dist=[]
    d=0
    if res.get_resname()=='A' or res.get_resname()=='G' :# different cases if 1 aromatic cycle or 2
        atom_res=pos_b2(res)
        if pair.get_resname()=='A' or pair.get_resname()=='G' :
            atom_pair=pos_b2(pair)
        if pair.get_resname()=='C' or pair.get_resname()=='U' :
            atom_pair=pos_b1(pair)

    if res.get_resname()=='C' or res.get_resname()=='U' :
        atom_res=pos_b1(res)
        if pair.get_resname()=='A' or pair.get_resname()=='G' :
            atom_pair=pos_b2(pair)
        if pair.get_resname()=='C' or pair.get_resname()=='U' :
            atom_pair=pos_b1(pair)
    
    dist=distance(atom_res, atom_pair)

    return dist

def angle_c1_b1(res,pair):
    '''
    measurement of the plane angles formed by the vectors C1-> B1 of the paired nucleotides
    '''
    if res.get_resname()=='A' or res.get_resname()=='G' or res.get_resname()=='C' or res.get_resname()=='U' :
        atom_c4_res = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ] 
        atom_c1p_res = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
        atom_b1_res=pos_b1(res)
        c4_res=Vector(atom_c4_res[0])
        c1_res=Vector(atom_c1p_res[0])
        b1_res=Vector(atom_b1_res)
        if pair.get_resname()=='A' or pair.get_resname()=='G' or pair.get_resname()=='C' or pair.get_resname()=='U' :
            atom_c4_pair = [ atom.get_coord() for atom in pair if "C4'" in atom.get_fullname() ]
            atom_c1p_pair = [ atom.get_coord() for atom in pair if "C1'" in atom.get_fullname() ]
            atom_b1_pair=pos_b1(pair)
            c4_pair=Vector(atom_c4_pair[0])
            c1_pair=Vector(atom_c1p_pair[0])
            b1_pair=Vector(atom_b1_pair)
            #we calculate the 4 plane angles including these vectors
            
            a=calc_angle(c4_res, c1_res, b1_res)*(180/np.pi)
            b=calc_angle(c1_res, b1_res, b1_pair)*(180/np.pi)
            c=calc_angle(b1_res, b1_pair, c1_pair)*(180/np.pi)
            d=calc_angle(b1_pair, c1_pair, c4_pair)*(180/np.pi)
            

    angles=[a, b, c, d]
    return angles

def graphe(type_LW, angle_1, angle_2, angle_3, angle_4, distance):
    '''
    function to plot the statistical figures you want
    By type of pairing:
    Superposition of GMMs of plane angles
    Superposition of the histogram and the GMM of the distances
    all in the same window
    '''

    figure = plt.figure(figsize = (10, 10))
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0, hspace = 0.5)
   
    plt.subplot(2, 1, 1)

    if len(angle_1) > 0 :
        GMM_tot(angle_1, "C4'-C1'-B1", 'cyan' )
    if len(angle_2) > 0 :
        GMM_tot(angle_2, "C1'-B1-B1pair", 'magenta')
    if len(angle_3) > 0 :
        GMM_tot(angle_3, "B1-B1pair-C1'pair", "yellow")
    if len(angle_4) > 0 :
        GMM_tot(angle_4, "B1pair-C1'pair-C4'pair", 'olive')
    plt.xlabel("Angle(degré)")
    plt.title("GMM des angles plans pour les appariements " +type_LW , fontsize=10)

    plt.subplot(2, 1, 2)
    if len(distance)>0 :
        GMM_histo_without_saving(distance, "Distance pointes " + type_LW, "Distance (Angström)", "Densité")

    plt.savefig("Mesures appariements " +type_LW+ ".png" )
    plt.close()

def appariements(chain, df):
    '''
    Cleanup of the dataset
    measurements of distances and angles between paired nucleotides in the chain
    '''
    liste_dist=[]
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

    pairs=df[['index_chain', 'old_nt_resnum', 'paired', 'pair_type_LW']] # columns we keep
    for i in range(pairs.shape[0]): #we remove the lines where no pairing (NaN in paired)
            index_with_nan=pairs.index[pairs.iloc[:,2].isnull()]
            pairs.drop(index_with_nan, 0, inplace=True)

    paired_int=[]
    for i in pairs.index:# convert values ​​from paired to integers or lists of integers
        paired=pairs.at[i, 'paired']
        if type(paired) is np.int64 or type(paired) is np.float64:
            paired_int.append(int(paired))
        else :  #strings
            if len(paired)<3 : #a single pairing
                paired_int.append(int(paired))         
            else : #several pairings
                paired=paired.split(',')
                l=[int(i) for i in paired]
                paired_int.append(l)

    pair_type_LW_bis=[]
    for j in pairs.index:
        pair_type_LW = pairs.at[j, 'pair_type_LW']
        if len(pair_type_LW)<4 : #a single pairing
            pair_type_LW_bis.append(pair_type_LW)
        else : #several pairings
            pair_type_LW=pair_type_LW.split(',')
            l=[i for i in pair_type_LW]
            pair_type_LW_bis.append(pair_type_LW)

    #addition of these new columns
    pairs.insert(4, "paired_int", paired_int, True)
    pairs.insert(5, "pair_type_LW_bis", pair_type_LW_bis, True)
    
    indexNames=pairs[pairs['paired_int'] == 0].index
    pairs.drop(indexNames, inplace=True)#deletion of lines with a 0 in paired_int (matching to another RNA chain)

    for i in pairs.index:
        '''
        calculations for each row of the pairs dataset
        '''
        resseq=pairs.at[i, 'old_nt_resnum'] #number of the residue in the chain
        #code to delete letters in old_nt_resnum
        icode_res=' '
        if type(resseq) is str:
            if resseq[0] != '-' :
                while resseq.isdigit() is False:
                    l=len(resseq)
                    if icode_res==' ':
                        icode_res=resseq[l-1]
                    else :
                        icode_res=resseq[l-1]+icode_res
                    resseq=resseq[:l-1]
        resseq=int(resseq)
        index=pairs.at[i, 'index_chain'] 
        type_LW=pairs.at[i, 'pair_type_LW_bis'] #pairing type
        num_paired=pairs.at[i, 'paired_int'] #number (index_chain) of the paired nucleotide
        

        if type(num_paired) is int or type(num_paired) is np.int64:
            l=pairs[pairs['index_chain']==num_paired].index.to_list()
            
            resnum=pairs.at[l[0], 'old_nt_resnum']
            icode_pair=' '
            if type(resnum) is str:
                if resnum[0] != '-' :
                    while resnum.isdigit() is False:
                        l=len(resnum)
                        if icode_pair==' ':
                            icode_pair=resnum[l-1]
                        else :
                            icode_pair=resnum[l-1]+icode_pair
                        resnum=resnum[:l-1]
                        
                resnum=int(resnum)
            try :
                d=dist_pointes(chain[(' ',resseq, icode_res)], chain[(' ', resnum, icode_pair)]) #calculation of the distance between the tips of the paired nucleotides
                angle=angle_c1_b1(chain[(' ', resseq, icode_res)], chain[(' ', resnum, icode_pair)])
                if d != 0.0:
                    liste_dist.append([chain, type_LW, resseq, resnum, d,  angle[0], angle[1], angle[2], angle[3]])   
            except :
                pass
        else : 
            for j in range(len(num_paired)): #if several pairings, process them one by one
                if num_paired[j] != 0 :
                    l=pairs[pairs['index_chain']==num_paired[j]].index.to_list()
                    
                    resnum=pairs.at[l[0], 'old_nt_resnum']
                    
                    icode_pair=' '
                    if type(resnum) is str:
                        if resnum[0] != '-' :
                            while resnum.isdigit() is False:
                                l=len(resnum)
                                if icode_pair==' ':
                                    icode_pair=resnum[l-1]
                                else :
                                    icode_pair=resnum[l-1]+icode_pair
                                resnum=resnum[:l-1]
                        resnum=int(resnum)
                    try :
                        d=dist_pointes(chain[(' ', resseq, icode_res)], chain[(' ', resnum, icode_pair)])
                        angle=angle_c1_b1(chain[(' ', resseq, icode_res)], chain[(' ', resnum, icode_pair)])
                        if d != 0.0:
                            liste_dist.append([chain, type_LW[j], resseq, resnum, d, angle[0], angle[1], angle[2], angle[3]])
                    except:
                        pass

    return(liste_dist)

def parse(cle):
    '''
    For each pdb structure in the dictionary, get its chains and do the pairing measurements on them
    '''
    l=[]
    os.makedirs(runDir + "/results/basepairs/", exist_ok=True)
    with warnings.catch_warnings():
        # Ignore the PDB problems. This mostly warns that some chain is discontinuous.
        warnings.simplefilter('ignore', Bio.PDB.PDBExceptions.PDBConstructionWarning)
        warnings.simplefilter('ignore', Bio.PDB.PDBExceptions.BiopythonWarning)
        parser=MMCIFParser()
        s = parser.get_structure(cle, os.path.abspath(path_to_3D_data + "RNAcifs/" + cle +".cif")) # parse the original structure
        for model in s:
            for valeur in dictionnaire[cle]:
                if len(valeur)>2: # if several RNA chains in the same structure
                    df_tot=[]
                    for id_chain in valeur:
                        for data in ld:
                            if (len(data)<10): #unmapped
                                chaine=str.split(data, '_')
                                if chaine[0]==cle and chaine[2]==id_chain:
                                    df=pd.read_csv(os.path.abspath(path_to_3D_data +"datapoints/" + data))
                                    if df['index_chain'][0]==1:#ignore files with numbering errors
                                        l=appariements(model[id_chain], df)
                else : #if only one RNA chain
                    for data in ld:
                        if (len(data)<10): #unmapped
                            chaine=str.split(data, '_')
                            if chaine[0]==cle and chaine[2]==valeur:
                                df=pd.read_csv(os.path.abspath(path_to_3D_data + "datapoints/" + data))
                                if df['index_chain'][0]==1:
                                    l=appariements(model[valeur], df)
                df_calc=pd.DataFrame(l, columns=["Chaine", "type LW", "Resseq", "Num paired", "Distance", "C4'-C1'-B1", "C1'-B1-B1pair", "B1-B1pair-C1'pair", "B1pair-C1'pair-C4'pair"])
                df_calc.to_csv(runDir + "/results/basepairs/"+'basepairs '+cle+'_'+valeur+'.csv')

def graph_basepairs():
    df=pd.read_csv(os.path.abspath(runDir + "/results/basepairs/basepairs.csv"))

    cWW=df[df['type LW']=='cWW']
    cWW_dist=list(cWW["Distance"])
    cWW_angle_1=list(cWW["C4'-C1'-B1"])
    cWW_angle_2=list(cWW["C1'-B1-B1pair"])
    cWW_angle_3=list(cWW["B1-B1pair-C1'pair"])
    cWW_angle_4=list(cWW["B1pair-C1'pair-C4'pair"])
    tWW=df[df['type LW']=='tWW']
    tWW_dist=list(tWW["Distance"])
    tWW_angle_1=list(tWW["C4'-C1'-B1"])
    tWW_angle_2=list(tWW["C1'-B1-B1pair"])
    tWW_angle_3=list(tWW["B1-B1pair-C1'pair"])
    tWW_angle_4=list(tWW["B1pair-C1'pair-C4'pair"])
    cWH=df[df['type LW']=='cWH']
    cWH_dist=list(cWH["Distance"])
    cWH_angle_1=list(cWH["C4'-C1'-B1"])
    cWH_angle_2=list(cWH["C1'-B1-B1pair"])
    cWH_angle_3=list(cWH["B1-B1pair-C1'pair"])
    cWH_angle_4=list(cWH["B1pair-C1'pair-C4'pair"])
    tWH=df[df['type LW']=='tWH']
    tWH_dist=list(tWH["Distance"])
    tWH_angle_1=list(tWH["C4'-C1'-B1"])
    tWH_angle_2=list(tWH["C1'-B1-B1pair"])
    tWH_angle_3=list(tWH["B1-B1pair-C1'pair"])
    tWH_angle_4=list(tWH["B1pair-C1'pair-C4'pair"])
    cHW=df[df['type LW']=='cHW']
    cHW_dist=list(cHW["Distance"])
    cHW_angle_1=list(cHW["C4'-C1'-B1"])
    cHW_angle_2=list(cHW["C1'-B1-B1pair"])
    cHW_angle_3=list(cHW["B1-B1pair-C1'pair"])
    cHW_angle_4=list(cHW["B1pair-C1'pair-C4'pair"])
    tHW=df[df['type LW']=='tHW']
    tHW_dist=list(tHW["Distance"])
    tHW_angle_1=list(tHW["C4'-C1'-B1"])
    tHW_angle_2=list(tHW["C1'-B1-B1pair"])
    tHW_angle_3=list(tHW["B1-B1pair-C1'pair"])
    tHW_angle_4=list(tHW["B1pair-C1'pair-C4'pair"])
    cWS=df[df['type LW']=='cWS']
    cWS_dist=list(cWS["Distance"])
    cWS_angle_1=list(cWS["C4'-C1'-B1"])
    cWS_angle_2=list(cWS["C1'-B1-B1pair"])
    cWS_angle_3=list(cWS["B1-B1pair-C1'pair"])
    cWS_angle_4=list(cWS["B1pair-C1'pair-C4'pair"])
    tWS=df[df['type LW']=='tWS']
    tWS_dist=list(tWS["Distance"])
    tWS_angle_1=list(tWS["C4'-C1'-B1"])
    tWS_angle_2=list(tWS["C1'-B1-B1pair"])
    tWS_angle_3=list(tWS["B1-B1pair-C1'pair"])
    tWS_angle_4=list(tWS["B1pair-C1'pair-C4'pair"])
    cSW=df[df['type LW']=='cSW']
    cSW_dist=list(cSW["Distance"])
    cSW_angle_1=list(cSW["C4'-C1'-B1"])
    cSW_angle_2=list(cSW["C1'-B1-B1pair"])
    cSW_angle_3=list(cSW["B1-B1pair-C1'pair"])
    cSW_angle_4=list(cSW["B1pair-C1'pair-C4'pair"])
    tSW=df[df['type LW']=='tSW']
    tSW_dist=list(tSW["Distance"])
    tSW_angle_1=list(tSW["C4'-C1'-B1"])
    tSW_angle_2=list(tSW["C1'-B1-B1pair"])
    tSW_angle_3=list(tSW["B1-B1pair-C1'pair"])
    tSW_angle_4=list(tSW["B1pair-C1'pair-C4'pair"])
    cHH=df[df['type LW']=='cHH']
    cHH_dist=list(cHH["Distance"])
    cHH_angle_1=list(cHH["C4'-C1'-B1"])
    cHH_angle_2=list(cHH["C1'-B1-B1pair"])
    cHH_angle_3=list(cHH["B1-B1pair-C1'pair"])
    cHH_angle_4=list(cHH["B1pair-C1'pair-C4'pair"])
    tHH=df[df['type LW']=='tHH']
    tHH_dist=list(tHH["Distance"])
    tHH_angle_1=list(tHH["C4'-C1'-B1"])
    tHH_angle_2=list(tHH["C1'-B1-B1pair"])
    tHH_angle_3=list(tHH["B1-B1pair-C1'pair"])
    tHH_angle_4=list(tHH["B1pair-C1'pair-C4'pair"])
    cSH=df[df['type LW']=='cSH']
    cSH_dist=list(cSH["Distance"])
    cSH_angle_1=list(cSH["C4'-C1'-B1"])
    cSH_angle_2=list(cSH["C1'-B1-B1pair"])
    cSH_angle_3=list(cSH["B1-B1pair-C1'pair"])
    cSH_angle_4=list(cSH["B1pair-C1'pair-C4'pair"])
    tSH=df[df['type LW']=='tSH']
    tSH_dist=list(tSH["Distance"])
    tSH_angle_1=list(tSH["C4'-C1'-B1"])
    tSH_angle_2=list(tSH["C1'-B1-B1pair"])
    tSH_angle_3=list(tSH["B1-B1pair-C1'pair"])
    tSH_angle_4=list(tSH["B1pair-C1'pair-C4'pair"])
    cHS=df[df['type LW']=='cHS']
    cHS_dist=list(cHS["Distance"])
    cHS_angle_1=list(cHS["C4'-C1'-B1"])
    cHS_angle_2=list(cHS["C1'-B1-B1pair"])
    cHS_angle_3=list(cHS["B1-B1pair-C1'pair"])
    cHS_angle_4=list(cHS["B1pair-C1'pair-C4'pair"])
    tHS=df[df['type LW']=='tHS']
    tHS_dist=list(tHS["Distance"])
    tHS_angle_1=list(tHS["C4'-C1'-B1"])
    tHS_angle_2=list(tHS["C1'-B1-B1pair"])
    tHS_angle_3=list(tHS["B1-B1pair-C1'pair"])
    tHS_angle_4=list(tHS["B1pair-C1'pair-C4'pair"])
    cSS=df[df['type LW']=='cSS']
    cSS_dist=list(cSS["Distance"])
    cSS_angle_1=list(cSS["C4'-C1'-B1"])
    cSS_angle_2=list(cSS["C1'-B1-B1pair"])
    cSS_angle_3=list(cSS["B1-B1pair-C1'pair"])
    cSS_angle_4=list(cSS["B1pair-C1'pair-C4'pair"])
    tSS=df[df['type LW']=='tSS']
    tSS_dist=list(tSS["Distance"])
    tSS_angle_1=list(tSS["C4'-C1'-B1"])
    tSS_angle_2=list(tSS["C1'-B1-B1pair"])
    tSS_angle_3=list(tSS["B1-B1pair-C1'pair"])
    tSS_angle_4=list(tSS["B1pair-C1'pair-C4'pair"])

    os.makedirs(runDir + "/results/figures/basepairs/", exist_ok=True)
    os.chdir(runDir + "/results/figures/basepairs/")

    graphe('cWW', cWW_angle_1, cWW_angle_2, cWW_angle_3, cWW_angle_4, cWW_dist)
    graphe('tWW', tWW_angle_1, tWW_angle_2, tWW_angle_3, tWW_angle_4, tWW_dist)
    graphe('cWH', cWH_angle_1, cWH_angle_2, cWH_angle_3, cWH_angle_4, cWH_dist)
    graphe('tWH', tWH_angle_1, tWH_angle_2, tWH_angle_3, tWH_angle_4, tWH_dist)
    graphe('cHW', cHW_angle_1, cHW_angle_2, cHW_angle_3, cHW_angle_4, cHW_dist)
    graphe('tHW', tHW_angle_1, tHW_angle_2, tHW_angle_3, tHW_angle_4, tHW_dist)
    graphe('tWS', tWS_angle_1, tWS_angle_2, tWS_angle_3, tWS_angle_4, tWS_dist)
    graphe('cWS', cWS_angle_1, cWS_angle_2, cWS_angle_3, cWS_angle_4, cWS_dist)
    graphe('tSW', tSW_angle_1, tSW_angle_2, tSW_angle_3, tSW_angle_4, tSW_dist)
    graphe('cSW', cSW_angle_1, cSW_angle_2, cSW_angle_3, cSW_angle_4, cSW_dist)
    graphe('cHH', cHH_angle_1, cHH_angle_2, cHH_angle_3, cHH_angle_4, cHH_dist)
    graphe('tHH', tHH_angle_1, tHH_angle_2, tHH_angle_3, tHH_angle_4, tHH_dist)
    graphe('cSH', cSH_angle_1, cSH_angle_2, cSH_angle_3, cSH_angle_4, cSH_dist)
    graphe('tSH', tSH_angle_1, tSH_angle_2, tSH_angle_3, tSH_angle_4, tSH_dist)
    graphe('cHS', cHS_angle_1, cHS_angle_2, cHS_angle_3, cHS_angle_4, cHS_dist)
    graphe('tHS', tHS_angle_1, tHS_angle_2, tHS_angle_3, tHS_angle_4, tHS_dist)
    graphe('cSS', cSS_angle_1, cSS_angle_2, cSS_angle_3, cSS_angle_4, cSS_dist)
    graphe('tSS', tSS_angle_1, tSS_angle_2, tSS_angle_3, tSS_angle_4, tSS_dist)
    
    nc=len(cWW)+len(cHH)+len(cSS)+len(cWH)+len(cHW)+len(cWS)+len(cSW)+len(cHS)+len(cSH)
    GMM_tot(cWW_dist, "cWW", 'lightcoral')
    GMM_tot(cHH_dist, "cHH", 'lightseagreen')
    GMM_tot(cSS_dist, "cSS", 'black')
    GMM_tot(cWH_dist, "cWH", 'goldenrod')
    GMM_tot(cHW_dist, "cHW", 'olive')
    GMM_tot(cWS_dist, "cWS", 'steelblue')  
    GMM_tot(cSW_dist, "cSW", 'silver')
    GMM_tot(cHS_dist, "cHS", 'deeppink')
    GMM_tot(cSH_dist, "cSH", 'navy')
    plt.xlabel('Distance (Angström)')
    plt.title("GMM des distances entre pointes des nucléotides pour les appariements cis ("+str(nc)+ " valeurs)", fontsize=9)
    plt.savefig("GMM des distances entre pointes des nucléotides pour les appariements cis (" +str(nc)+ " valeurs).png")
    plt.close()

    nt=len(tWW)+len(tHH)+len(tSS)+len(tWH)+len(tHW)+len(tWS)+len(tSW)+len(tHS)+len(tSH)
    GMM_tot(tWW_dist, "tWW", 'sienna')
    GMM_tot(tHH_dist, "tHH", 'maroon')
    GMM_tot(tSS_dist, "tSS", 'orange') 
    GMM_tot(tWH_dist, "tWH", 'mediumaquamarine') 
    GMM_tot(tHW_dist, "tHW", 'tomato')
    GMM_tot(tWS_dist, "tWS", 'indigo')
    GMM_tot(tSW_dist, "tSW", 'orchid')
    GMM_tot(tHS_dist, "tHS", 'tan')
    GMM_tot(tSH_dist, "tSH", 'lime')
    plt.xlabel('Distance (Angström)')
    plt.title("GMM des distances entre pointes des nucléotides pour les appariements trans ("+str(nt)+ " valeurs)", fontsize=9)
    plt.savefig("GMM des distances entre pointes des nucléotides pour les appariements trans (" +str(nt)+ " valeurs).png")
    plt.close()
    

if __name__ == "__main__":

    os.makedirs(runDir + "/results/figures/", exist_ok=True)

    # parse options
    DELETE_OLD_DATA = False
    DO_WADLEY_ANALYSIS = False
    DO_AVG_DISTANCE_MATRIX = False
    try:
        opts, _ = getopt.getopt( sys.argv[1:], "r:h", [ "help", "from-scratch", "wadley", "distance-matrices", "resolution=", "3d-folder=", "seq-folder=" ])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for opt, arg in opts:

        if opt == "-h" or opt == "--help":
            print(  "RNANet statistics, a script to build a multiscale RNA dataset from public data\n"
                    "Developped by Louis Becquey an Khodor Hannoush, 2020/2021")
            print()
            print("Options:")
            print("-h [ --help ]\t\t\tPrint this help message")
            print()
            print("-r 20.0 [ --resolution=20.0 ]\tCompute statistics using chains of resolution 20.0A or better.")
            print("--3d-folder=…\t\t\tPath to a folder containing the 3D data files. Required subfolders should be:"
                    "\n\t\t\t\t\tdatapoints/\t\tFinal results in CSV file format.")
            print("--seq-folder=…\t\t\tPath to a folder containing the sequence and alignment files. Required subfolder:"
                    "\n\t\t\t\t\trealigned/\t\tSequences, covariance models, and alignments by family")
            print("--from-scratch\t\t\tDo not use precomputed results from past runs, recompute everything")
            print("--distance-matrices\t\tCompute average distance between nucleotide pairs for each family.")
            print("--wadley\t\t\tReproduce Wadley & al 2007 clustering of pseudotorsions.")

            sys.exit()
        elif opt == '--version':
            print("RNANet statistics 1.5 beta")
            sys.exit()
        elif opt == "-r" or opt == "--resolution":
            assert float(arg) > 0.0 and float(arg) <= 20.0 
            res_thr = float(arg)
        elif opt=='--3d-folder':
            path_to_3D_data = path.abspath(arg)
            if path_to_3D_data[-1] != '/':
                path_to_3D_data += '/'
        elif opt=='--seq-folder':
            path_to_seq_data = path.abspath(arg)
            if path_to_seq_data[-1] != '/':
                path_to_seq_data += '/'
        elif opt=='--from-scratch':
            DELETE_OLD_DATA = True
            DO_WADLEY_ANALYSIS = True
        elif opt=="--distance-matrices":
            DO_AVG_DISTANCE_MATRIX = True
        elif opt=='--wadley':
            DO_WADLEY_ANALYSIS = True
    

    # Load mappings. famlist will contain only families with structures at this resolution threshold.
    '''
    print("Loading mappings list...")
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        n_unmapped_chains = sql_ask_database(conn, "SELECT COUNT(*) FROM chain WHERE rfam_acc='unmappd' AND issue=0;")[0][0]
        families = pd.read_sql(f"""SELECT rfam_acc, count(*) as n_chains 
                                    FROM chain JOIN structure
                                    ON chain.structure_id = structure.pdb_id
                                    WHERE issue = 0 AND resolution <= {res_thr} AND rfam_acc != 'unmappd'
                                    GROUP BY rfam_acc;
                                """, conn)
    families.drop(families[families.n_chains == 0].index, inplace=True)
    mappings_list = {}
    for k in families.rfam_acc:
        mappings_list[k] = [ x[0] for x in sql_ask_database(conn,  f"""SELECT chain_id 
                                                                        FROM chain JOIN structure ON chain.structure_id=structure.pdb_id 
                                                                        WHERE rfam_acc='{k}' AND issue=0 AND resolution <= {res_thr};""") ]
    famlist = families.rfam_acc.tolist()
    ignored = families[families.n_chains < 3].rfam_acc.tolist()
    famlist.sort(key=family_order)

    print(f"Found {len(famlist)} families with chains of resolution {res_thr}A or better.")
    if len(ignored):
        print(f"Idty matrices: Ignoring {len(ignored)} families with only one chain:", " ".join(ignored)+'\n')
    '''
    if DELETE_OLD_DATA:
        for f in famlist:
            subprocess.run(["rm","-f", runDir + f"/data/{f}.npy", runDir + f"/data/{f}_pairs.csv", runDir + f"/data/{f}_counts.csv"])
        if DO_WADLEY_ANALYSIS:
            subprocess.run(["rm","-f", runDir + f"/data/wadley_kernel_eta_{res_thr}.npz", runDir + f"/data/wadley_kernel_eta_prime_{res_thr}.npz", runDir + f"/data/pair_counts_{res_thr}.csv"])
        if DO_AVG_DISTANCE_MATRIX:
            subprocess.run(["rm", "-rf", runDir + f"/results/distance_matrices/"])

    # Prepare the multiprocessing execution environment
    nworkers = min(read_cpu_number()-1, 40)
    thr_idx_mgr = Manager()
    idxQueue = thr_idx_mgr.Queue()
    for i in range(nworkers):
        idxQueue.put(i)

    # Define the tasks
    joblist = []
    '''
    if n_unmapped_chains and DO_WADLEY_ANALYSIS:
        joblist.append(Job(function=reproduce_wadley_results, args=(1, False, (1,4), res_thr)))
        joblist.append(Job(function=reproduce_wadley_results, args=(4, False, (1,4), res_thr)))
    if DO_AVG_DISTANCE_MATRIX:
        extracted_chains = []
        for file in os.listdir(path_to_3D_data + "rna_mapped_to_Rfam"):
            if os.path.isfile(os.path.join(path_to_3D_data + "rna_mapped_to_Rfam", file)):
                e1 = file.split('_')[0]
                e2 = file.split('_')[1]
                e3 = file.split('_')[2]
                extracted_chains.append(e1 + '[' + e2 + ']' + '-' + e3)
        for f in [ x for x in famlist if (x not in LSU_set and x not in SSU_set) ]:    # Process the rRNAs later only 3 by 3
            joblist.append(Job(function=get_avg_std_distance_matrix, args=(f, True, False)))
            joblist.append(Job(function=get_avg_std_distance_matrix, args=(f, False, False)))
    joblist.append(Job(function=stats_len)) # Computes figures
    joblist.append(Job(function=stats_freq)) # updates the database
    
    for f in famlist:
        joblist.append(Job(function=parallel_stats_pairs, args=(f,))) # updates the database
        if f not in ignored:
            joblist.append(Job(function=to_id_matrix, args=(f,))) # updates the database
    '''

    
    '''
    f_prec=os.listdir(path_to_3D_data + "rna_only")[0]
    #exit()
    for f in os.listdir(path_to_3D_data + "rna_only")[:100]:
        joblist.append(Job(function=dist_atoms, args=(f,)))
    '''
    '''
    ld=os.listdir(path_to_3D_data +'datapoints')[:1000]
    if '4zdo_1_E' in ld :
        ld.remove('4zdo_1_E')#cas particuliers
    if '4zdp_1_E' in ld :
        ld.remove('4zdp_1_E')
    dictionnaire=dictio(ld)
    
    for cle in dictionnaire.keys():
        joblist.append(Job(function=parse, args=(cle,)))
    '''
    
    #exit()
    '''
    f_prec=os.listdir(path_to_3D_data + "rna_only")[0]

    for f in os.listdir(path_to_3D_data + "rna_only")[:100]: 
        joblist.append(Job(function=dist_atoms, args=(f,)))
        joblist.append(Job(function=dist_atoms_hire_RNA, args=(f,)))
        joblist.append(Job(function=angles_torsion_hire_RNA, args=(f,)))
        joblist.append(Job(function=angles_plans_hire_RNA, args=(f,)))
    
    '''
    
    p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=nworkers)
    pbar = tqdm(total=len(joblist), desc="Stat jobs", position=0, unit="job", leave=True)

    try:
        for j in joblist:
            p.apply_async(j.func_, args=j.args_, callback=log_to_pbar(pbar))
        p.close()
        p.join()
        pbar.close()
    except KeyboardInterrupt:
        warn("KeyboardInterrupt, terminating workers.", error=True)
        p.terminate()
        p.join()
        pbar.close()
        exit(1)
    except:
        print("Something went wrong")

    # Now process the memory-heavy tasks family by family
    if DO_AVG_DISTANCE_MATRIX:
        for f in LSU_set:
            get_avg_std_distance_matrix(f, True, True)
            get_avg_std_distance_matrix(f, False, True)
        for f in SSU_set:
            get_avg_std_distance_matrix(f, True, True)
            get_avg_std_distance_matrix(f, False, True)

    print()
    print()

    
    # finish the work after the parallel portions
    '''
    per_chain_stats()
    seq_idty()
    stats_pairs()
    if n_unmapped_chains:
        general_stats()
    '''
    concatenate('/results/all-atoms/distances/', 'dist_atoms.csv')    
    graph_dist_atoms()
    concatenate('/results/HiRE-RNA/distances/', 'dist_atoms_hire_RNA.csv')
    graph_dist_atoms_h_RNA()
    
    concatenate('/results/HiRE-RNA/torsions/', 'angles_torsion_hire_RNA.csv')
    concatenate('/results/HiRE-RNA/angles/', 'angles_plans_hire_RNA.csv')
    graph_torsion_h_RNA()
    graph_plans_h_RNA()
    
    graph_angles_torsion()
    graph_eta_theta()
    
    concatenate('/results/basepairs/', 'basepairs.csv')
    graph_basepairs()
    
