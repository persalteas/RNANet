#!/usr/bin/python3.8
import os, pickle, sys
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
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO, SeqIO
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from os import path
from collections import Counter
from RNAnet import read_cpu_number


path_to_3D_data = "/nhome/siniac/lbecquey/Data/RNA/3D/"
path_to_seq_data = "/nhome/siniac/lbecquey/Data/RNA/sequences/"

if len(sys.argv) > 1:
    path_to_3D_data = path.abspath(sys.argv[1])
    path_to_seq_data = path.abspath(sys.argv[2])

class DataPoint():
    def __init__(self, path_to_textfile):
        self.df = pd.read_csv(path_to_textfile, sep=',', header=0, engine="c", index_col=0)
        self.family = path_to_textfile.split('.')[-1]
        self.chain_label = path_to_textfile.split('.')[-2].split('/')[-1]

def load_rna_frome_file(path_to_textfile):
    return DataPoint(path_to_textfile)

def reproduce_wadley_results(points, show=False, carbon=4, sd_range=(1,4)):
    """
    Plot the joint distribution of pseudotorsion angles, in a Ramachandran-style graph.
    See Wadley & Pyle (2007)

    Arguments:
    show: True or False, call plt.show() at this end or not
    filter_helical: None, "form", "zone", or "both"
                    None: do not remove helical nucleotide
                    "form": remove nucleotides if they belong to a A, B or Z form stem
                    "zone": remove nucleotides falling in an arbitrary zone (see zone argument)
                    "both": remove nucleotides fulfilling one or both of the above conditions
    carbon: 1 or 4, use C4' (eta and theta) or C1' (eta_prime and theta_prime)
    sd_range: tuple, set values below avg + sd_range[0] * stdev to 0,
                     and values above avg + sd_range[1] * stdev to avg + sd_range[1] * stdev.
                     This removes noise and cuts too high peaks, to clearly see the clusters.
    """
    worker_nbr = 1 + (carbon==1)

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

    if not path.isfile(f"data/wadley_kernel_{angle}.npz"):
        c2_endo_etas = []
        c3_endo_etas = []
        c2_endo_thetas = []
        c3_endo_thetas = []
        for p in tqdm(points, desc="Loading eta/thetas", position=worker_nbr, leave=False):
            df = p.df.loc[(p.df[angle].isna()==False) & (p.df["th"+angle].isna()==False), ["form","puckering", angle,"th"+angle]]
            c2_endo_etas   += list(df.loc[ (df.puckering=="C2'-endo"), angle ].values)
            c3_endo_etas   += list(df.loc[ (df.form=='.') & (df.puckering=="C3'-endo"), angle ].values)
            c2_endo_thetas += list(df.loc[ (df.puckering=="C2'-endo"), "th"+angle ].values)
            c3_endo_thetas += list(df.loc[ (df.form=='.') & (df.puckering=="C3'-endo"), "th"+angle ].values)

        xx, yy = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        values_c3 = np.vstack([c3_endo_etas, c3_endo_thetas])
        kernel_c3 = st.gaussian_kde(values_c3)
        f_c3 = np.reshape(kernel_c3(positions).T, xx.shape)
        values_c2 = np.vstack([c2_endo_etas, c2_endo_thetas])
        kernel_c2 = st.gaussian_kde(values_c2)
        f_c2 = np.reshape(kernel_c2(positions).T, xx.shape)

        # Uncomment to save the data to an archive for later use without the need to recompute
        np.savez(f"data/wadley_kernel_{angle}.npz",
                  c3_endo_e=c3_endo_etas, c3_endo_t=c3_endo_thetas,
                  c2_endo_e=c2_endo_etas, c2_endo_t=c2_endo_thetas,
                  kernel_c3=f_c3, kernel_c2=f_c2)
    else:
        f = np.load(f"data/wadley_kernel_{angle}.npz")
        c2_endo_etas = f["c2_endo_e"]
        c3_endo_etas = f["c3_endo_e"]
        c2_endo_thetas = f["c2_endo_t"]
        c3_endo_thetas = f["c3_endo_t"]
        f_c3 = f["kernel_c3"]
        f_c2 = f["kernel_c2"]
        xx, yy = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]

    # print(f"[{worker_nbr}]\tKernel computed (or loaded from file).")

    # exact counts:
    hist_c2, xedges, yedges = np.histogram2d(c2_endo_etas, c2_endo_thetas, bins=int(2*np.pi/0.1), range=[[0, 2*np.pi], [0, 2*np.pi]])
    hist_c3, xedges, yedges = np.histogram2d(c3_endo_etas, c3_endo_thetas, bins=int(2*np.pi/0.1), range=[[0, 2*np.pi], [0, 2*np.pi]])
    color_values = cm.jet(hist_c3.ravel()/hist_c3.max())

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
        levels = [f.mean()+f.std(), f.mean()+2*f.std(), f.mean()+4*f.std()]

        # histogram:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        ax.bar3d(xpos.ravel(), ypos.ravel(), 0.0, 0.09, 0.09, hist_cut.ravel(), color=color_values, zorder="max")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(f"results/figures/wadley_plots/wadley_hist_{angle}_{l}.png")
        if show:
            fig.show()

        # Smoothed joint distribution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, f_cut, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(f"results/figures/wadley_plots/wadley_distrib_{angle}_{l}.png")
        if show:
            fig.show()

        # 2D Wadley plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.scatter(x, y, s=1, alpha=0.1)
        ax.contourf(xx, yy, f_cut, alpha=0.5, cmap=cm.coolwarm, levels=levels, extend="max")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(f"results/figures/wadley_plots/wadley_{angle}_{l}.png")
        if show:
            fig.show()
    # print(f"[{worker_nbr}]\tComputed joint distribution of angles (C{carbon}) and saved the figures.")

def stats_len(mappings_list, points):
    cols = []
    lengths = []
    for f in tqdm(sorted(mappings_list.keys()), desc="Chain length by family", position=3, leave=False):
        if f in ["RF02540","RF02541","RF02543"]:
            cols.append("red") # LSU
        elif f in ["RF00177","RF01960","RF01959","RF02542"]:
            cols.append("blue") # SSU
        elif f in ["RF00001"]:
            cols.append("green")
        elif f in ["RF00002"]:
            cols.append("purple")
        elif f in ["RF00005"]:
            cols.append("orange")
        else:
            cols.append("grey")
        l = []
        for r in points:
            if r.family != f: continue
            l.append(len(r.df['nt_code']))
        lengths.append(l)

    fig = plt.figure(figsize=(10,3))
    ax = fig.gca()
    ax.hist(lengths, bins=100, stacked=True, log=True, color=cols, label=sorted(mappings_list.keys()))
    ax.set_xlabel("Sequence length (nucleotides)")
    ax.set_ylabel("Number of 3D chains")
    fig.tight_layout()
    filtered_handles = [mpatches.Patch(color='red'), mpatches.Patch(color='white'),
                        mpatches.Patch(color='blue'), mpatches.Patch(color='white'),
                        mpatches.Patch(color='green'), mpatches.Patch(color='purple'),
                        mpatches.Patch(color='orange'), mpatches.Patch(color='grey')]
    filtered_labels = ['Large Ribosomal Subunits', '(RF02540, RF02541, RF02543)','Small Ribosomal Subunits','(RF01960, RF00177)',
                       '5S rRNA (RF00001)', '5.8S rRNA (RF00002)', 'tRNA (RF00005)', 'Other']
    ax.legend(filtered_handles, filtered_labels, loc='best', ncol=2)
    fig.savefig("results/lengths.png")
    # print("[3]\tComputed sequence length statistics and saved the figure.")

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

def stats_freq(mappings_list, points):
    freqs = {}
    for f in mappings_list.keys():
        freqs[f] = Counter()

    for r in tqdm(points, desc="Nucleotide frequencies", position=4, leave=False):
        freqs[r.family].update(dict(r.df['nt_name'].value_counts()))
    
    df = pd.DataFrame()
    for f in sorted(mappings_list.keys()):
        tot = sum(freqs[f].values())
        df = pd.concat([ df, pd.DataFrame([[ format_percentage(tot, x) for x in freqs[f].values() ]], columns=list(freqs[f]), index=[f]) ])
    df = df.fillna(0)
    df.to_csv("results/frequencies.csv")

    # print("[4]\tComputed nucleotide statistics and saved CSV file.")

def stats_pairs(mappings_list, points):

    def line_format(family_data):
        return family_data.apply(partial(format_percentage, sum(family_data)))

    # Create a Counter() object by family
    freqs = {}
    for f in mappings_list.keys():
        freqs[f] = Counter()

    # Iterate over data points
    if not path.isfile("data/pair_counts.csv"):
        for r in tqdm(points, desc="Leontis-Westhof basepair stats", position=5, leave=False):
            # Skip if linear piece of RNA
            if r.df.pair_type_LW.isna().all():
                continue 

            # Count each pair type within the molecule
            vcnts = pd.concat(
                                [   pd.Series(row['pair_type_LW'].split(',')) 
                                    for _, row in r.df.dropna(subset=["pair_type_LW"]).iterrows() ]
                            ).reset_index(drop=True).value_counts()

            # Add these new counts to the family's counter
            freqs[r.family].update(dict(vcnts))
        
        # Create the output dataframe
        df = pd.DataFrame()
        for f in sorted(mappings_list.keys()):
            df = pd.concat([ df, pd.DataFrame([[ x for x in freqs[f].values() ]], columns=list(freqs[f]), index=[f]) ])
        df = df.fillna(0)
        df.to_csv("data/pair_counts.csv")
    else:
        df = pd.read_csv("data/pair_counts.csv", index_col=0)


    # Remove not very well defined pair types (not in the 12 LW types)
    col_list = [ x for x in df.columns if '.' in x ]
    df['other'] = df[col_list].sum(axis=1)
    df.drop(col_list, axis=1, inplace=True)

    # Compute total row
    total_series = df.sum(numeric_only=True).rename("TOTAL")
    df = df.append(total_series)

    # format as percentages
    df = df.apply(line_format, axis=1)

    # reorder columns
    df.sort_values("TOTAL", axis=1, inplace=True, ascending=False)

    # Save to CSV
    df.to_csv("results/pairings.csv")

    # Plot barplot of overall types
    total_series.sort_values(ascending=False, inplace=True)
    total_series.apply(lambda x: x/2.0) # each interaction was counted twice because one time per extremity
    ax = total_series.plot(figsize=(5,3), kind='bar', log=True, ylim=(1e4,5000000) )
    ax.set_ylabel("Number of observations")
    plt.subplots_adjust(bottom=0.2, right=0.99)
    plt.savefig("results/pairings.png")

    # print("[5]\tComputed nucleotide statistics and saved CSV and PNG file.")

def to_dist_matrix(f):
    if path.isfile("data/"+f+".npy"):
        return 0

    dm = DistanceCalculator('identity')
    with open(path_to_seq_data+"realigned/"+f+"++.afa") as al_file:
        al = AlignIO.read(al_file, "fasta")[-len(mappings_list[f]):]
    idty = dm.get_distance(al).matrix # list of lists
    del al
    l = len(idty)
    np.save("data/"+f+".npy", np.array([ idty[i] + [0]*(l-1-i) if i<l-1 else idty[i]  for i in range(l) ]))
    del idty
    return 0

def seq_idty(mappings_list):
    famlist = sorted([ x for x in mappings_list.keys() if len(mappings_list[x]) > 1 ])
    ignored = []
    for x in mappings_list.keys():
        if len(mappings_list[x]) == 1:
            ignored.append(x)
    if len(ignored):
        print("Idty matrices: Ignoring families with only one chain:", " ".join(ignored)+'\n')

    # compute distance matrices
    p = Pool(processes=8)
    pbar = tqdm(total=len(famlist), desc="Families idty matrices", position=0, leave=False)
    for i, _ in enumerate(p.imap_unordered(to_dist_matrix, famlist)):
        pbar.update(1)
    pbar.close()
    p.close()
    p.join()

    # load them
    fam_arrays = []
    for f in famlist:
        if path.isfile("data/"+f+".npy"):
            fam_arrays.append(np.load("data/"+f+".npy"))
        else:
            fam_arrays.append([])

    fig, axs = plt.subplots(5,13, figsize=(15,9))
    axs = axs.ravel()
    [axi.set_axis_off() for axi in axs]
    for f, D, ax in zip(famlist, fam_arrays, axs):
        if not len(D): continue
        if D.shape[0] > 2:  # Cluster only if there is more than 2 sequences to organize
            D = D + D.T     # Copy the lower triangle to upper, to get a symetrical matrix
            condensedD = squareform(D)

            # Compute basic dendrogram by Ward's method
            Y = sch.linkage(condensedD, method='ward')
            Z = sch.dendrogram(Y, orientation='left', no_plot=True)

            # Reorganize rows and cols
            idx1 = Z['leaves']
            D = D[idx1,:]
            D = D[:,idx1[::-1]]
        im = ax.matshow(1.0 - D, vmin=0, vmax=1, origin='lower') # convert to identity matrix 1 - D from distance matrix D
        ax.set_title(f + "\n(" + str(len(mappings_list[f]))+ " chains)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.colorbar(im, ax=axs[-1], shrink=0.8)
    fig.savefig(f"results/distances.png")
    # print("[6]\tComputed identity matrices and saved the figure.")

if __name__ == "__main__":

    #################################################################
    #               LOAD ALL FILES
    #################################################################
    os.makedirs("results/figures/wadley_plots/", exist_ok=True)

    print("Loading mappings list...")
    mappings_list = pd.read_csv("results/mappings_list.csv", sep=',', index_col=0).to_dict(orient='list')
    for k in mappings_list.keys():
        mappings_list[k] = [ x for x in mappings_list[k] if str(x) != 'nan' ]

    print("Loading datapoints from file...")
    if path.isfile("data/rnapoints.dat"):
        with open("data/rnapoints.dat", 'rb') as f:
            rna_points = pickle.load(f)
    else:
        rna_points = []
        filelist = [path_to_3D_data+"/datapoints/"+f for f in os.listdir(path_to_3D_data+"/datapoints") ]
        p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=read_cpu_number())
        pbar = tqdm(total=len(filelist), desc="RNA files", position=0, leave=False)
        for i, rna in enumerate(p.imap_unordered(load_rna_frome_file, filelist)):
            rna_points.append(rna)
            pbar.update(1)
        pbar.close()
        p.close()
        p.join()
        with open("data/rnapoints.dat", "wb") as f:
            pickle.dump(rna_points, f)
    npoints = len(rna_points)
    print(npoints, "RNA files loaded.")

    #################################################################
    #               Define threads for the tasks
    #################################################################
    threads = [
        th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 1}),
        th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 4}),
        th.Thread(target=partial(stats_len, mappings_list), args=[rna_points]),
        th.Thread(target=partial(stats_freq, mappings_list), args=[rna_points]),
        th.Thread(target=partial(stats_pairs, mappings_list), args=[rna_points]),
        th.Thread(target=seq_idty, args=[mappings_list])
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

