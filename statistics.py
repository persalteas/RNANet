#!/usr/bin/python3

# This file computes additional statistics over the produced dataset.
# Run this file if you want the base counts, pair-type counts, identity percents, etc
# in the database.

import getopt, os, pickle, sqlite3, shlex, subprocess, sys, warnings
import numpy as np
import pandas as pd
import threading as th
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from mpl_toolkits.mplot3d import axes3d
from Bio import AlignIO, SeqIO
from Bio.PDB.MMCIFParser import MMCIFParser
from functools import partial
from multiprocessing import Pool, Manager
from os import path
from tqdm import tqdm
from collections import Counter
from setproctitle import setproctitle
from RNAnet import Job, read_cpu_number, sql_ask_database, sql_execute, warn, notify, init_worker, trace_unhandled_exceptions

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

        # Approximate the density by a gaussian kernel
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

def par_distance_matrix(filelist, f, label, consider_all_atoms, s):
    
    # Identify the right 3D file
    filename = ''
    for file in filelist:
        if file.startswith(s.id.replace('-', '').replace('[', '_').replace(']', '_')):
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
                d[i,j] = np.nan
            else:
                d[i,j] = get_euclidian_distance(coordinates_with_gaps[i], coordinates_with_gaps[j])
    
    if f not in LSU_set and f not in SSU_set:
        np.savetxt(runDir + '/results/distance_matrices/' + f + '_'+ label + '/'+ s.id.strip("\'") + '.csv', d, delimiter=",", fmt="%.3f")
    return 1-np.isnan(d).astype(int), np.nan_to_num(d), np.nan_to_num(d*d)

@trace_unhandled_exceptions
def get_avg_std_distance_matrix(f, consider_all_atoms, multithread=False):
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
    counts = np.zeros((ncols, ncols))
    avg = np.zeros((ncols, ncols))
    std = np.zeros((ncols, ncols))
    found = 0
    notfound = 0
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        r = sql_ask_database(conn, f"SELECT structure_id, '_1_', chain_name, '_', CAST(pdb_start AS TEXT), '-', CAST(pdb_end AS TEXT) FROM chain WHERE rfam_acc='{f}';")
        filelist = sorted([ ''.join(list(x))+'.cif' for x in r ])
    
    if not multithread:
        pbar = tqdm(total = len(align), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} {label} distance matrices", unit="chains", leave=False)
        pbar.update(0)
        for s in align:
            contrib, d, dsquared = par_distance_matrix(filelist, f, label, consider_all_atoms, s)
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
            for i, (contrib, d, dsquared) in enumerate(p.imap_unordered(partial(par_distance_matrix, filelist, f, label, consider_all_atoms), align, chunksize=1)):
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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")

        # Calculation of the average matrix
        try:
            avg = avg/counts
            assert (counts >0).all()
        except RuntimeWarning as e:
            # there will be division by 0 if count is 0 at some position = gap only column
            pass
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
        try:
            std = np.sqrt(std/counts - np.power(avg/counts, 2))
            assert (counts >0).all()
        except RuntimeWarning as e:
            # there will be division by 0 if count is 0 at some position = gap only column
            pass
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
                    "Developped by Louis Becquey (louis.becquey@univ-evry.fr), 2020")
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
            print("RNANet statistics 1.3 beta")
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
    
    if DELETE_OLD_DATA:
        for f in famlist:
            subprocess.run(["rm","-f", runDir + f"/data/{f}.npy", runDir + f"/data/{f}_pairs.csv", runDir + f"/data/{f}_counts.csv"])
        if DO_WADLEY_ANALYSIS:
            subprocess.run(["rm","-f", runDir + f"/data/wadley_kernel_eta_{res_thr}.npz", runDir + f"/data/wadley_kernel_eta_prime_{res_thr}.npz", runDir + f"/data/pair_counts_{res_thr}.csv"])
        if DO_AVG_DISTANCE_MATRIX:
            subprocess.run(["rm", "-rf", runDir + f"/results/distance_matrices/"])

    # Prepare the multiprocessing execution environment
    nworkers = min(read_cpu_number()-1, 32)
    thr_idx_mgr = Manager()
    idxQueue = thr_idx_mgr.Queue()
    for i in range(nworkers):
        idxQueue.put(i)

    # Define the tasks
    joblist = []
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
    per_chain_stats()
    seq_idty()
    stats_pairs()
    if n_unmapped_chains:
        general_stats()
