#!/usr/bin/python3.8
import os
import numpy as np
import pandas as pd
import threading as th
import seaborn as sb
import scipy.stats as st
import matplotlib.pyplot as plt
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from mpl_toolkits.mplot3d import axes3d
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO, SeqIO
from matplotlib import cm 
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from os import path
from RNAnet import read_cpu_number


if os.path.isdir("/home/ubuntu/"): # this is the IFB-core cloud
    path_to_3D_data = "/mnt/Data/RNA/3D/"
    path_to_seq_data = "/mnt/Data/RNA/sequences/"
elif os.path.isdir("/home/persalteas"): # this is my personal workstation
    path_to_3D_data = "/home/persalteas/Data/RNA/3D/"
    path_to_seq_data = "/home/persalteas/Data/RNA/sequences/"
elif os.path.isdir("/home/lbecquey"): # this is the IBISC server
    path_to_3D_data = "/home/lbecquey/Data/RNA/3D/"
    path_to_seq_data = "/home/lbecquey/Data/RNA/sequences/"
elif os.path.isdir("/nhome/siniac/lbecquey"): # this is the office PC
    path_to_3D_data = "/nhome/siniac/lbecquey/Data/RNA/3D/"
    path_to_seq_data = "/nhome/siniac/lbecquey/Data/RNA/sequences/"
else:
    print("I don't know that machine... I'm shy, maybe you should introduce yourself ?")
    exit(1)

class DataPoint():
    def __init__(self, path_to_textfile):
        self.df = pd.read_csv(path_to_textfile, sep=',', header=0, engine="c", index_col=0)
        self.family = path_to_textfile.split('.')[-1]
        self.chain_label = path_to_textfile.split('.')[-2].split('/')[-1]

def load_rna_frome_file(path_to_textfile):
    return DataPoint(path_to_textfile)

def reproduce_wadley_results(points, show=False, filter_helical=None, carbon=4, zone=(2.7,3.3,3.5,4.5)):
    """
    Plot the joint distribution of pseudotorsion angles, in a Ramachandran-style graph.
    See Wadley & Pyle (2007)
    """

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

    all_etas = []
    all_thetas = []
    all_forms = []
    c = 0
    for p in points:
        all_etas += list(p.df[angle].values)
        all_thetas += list(p.df['th'+angle].values)
        all_forms += list(p.df['form'].values)
        if (len([ x for x in p.df[angle].values if x < 0 or x > 7]) or 
            len([ x for x in p.df['th'+angle].values if x < 0 or x > 7])):
            c += 1
    if c:
        print(c,"points on",len(points),"have non-radian angles !")

    print("combining etas and thetas...")
    warn = ""
    if not filter_helical:
        alldata = [ (e, t) 
                    for e, t in zip(all_etas, all_thetas) 
                    if ('nan' not in str((e,t)))  ]
    elif filter_helical == "form":
        alldata = [ (e, t) 
                    for e, t, f in zip(all_etas, all_thetas, all_forms) 
                    if ('nan' not in str((e,t))) 
                    and f == '.' ]
        warn = "(helical nucleotides removed)"
        print(len(alldata), "couples of non-helical nts found in", len(all_etas))
    elif filter_helical == "zone":
        alldata = [ (e, t) 
                    for e, t in zip(all_etas, all_thetas) 
                    if ('nan' not in str((e,t))) 
                    and not (e>zone[0] and e<zone[1] and t>zone[2] and t<zone[3]) ]
        warn = "(massive peak of helical nucleotides removed in red zone)"
        print(len(alldata), "couples of non-helical nts found in", len(all_etas))
    elif filter_helical == "both":
        alldata = [ (e, t) 
                    for e, t, f in zip(all_etas, all_thetas, all_forms) 
                    if ('nan' not in str((e,t))) 
                    and f == '.'
                    and not (e>zone[0] and e<zone[1] and t>zone[2] and t<zone[3]) ]
        warn = "(helical nucleotide and massive peak in the red zone removed)"
        print(len(alldata), "couples of non-helical nts found in", len(all_etas))

    x = np.array([ p[0] for p in alldata ])
    y = np.array([ p[1] for p in alldata ])
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    sign_threshold = np.mean(f) + np.std(f)
    z = np.where(f < sign_threshold, 0.0, f)
    z_inc = np.where(f < sign_threshold, sign_threshold, f)

    # histogram : 
    fig, axs = plt.subplots(1,3, figsize=(18, 6))
    ax = fig.add_subplot(131)
    ax.cla()
    
    plt.axhline(y=0, alpha=0.5, color='black')
    plt.axvline(x=0, alpha=0.5, color='black')
    plt.scatter(x, y, s=1, alpha=0.1)
    plt.contourf(xx, yy, z, cmap=cm.BuPu, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if filter_helical in ["zone","both"]:
        ax.add_patch(ptch.Rectangle((zone[0],zone[2]),zone[1]-zone[0],zone[3]-zone[2], linewidth=1, edgecolor='r', facecolor='#ff000080'))

    ax = fig.add_subplot(132, projection='3d')
    ax.cla()
    ax.plot_surface(xx, yy, z_inc, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_title(f"\"Wadley plot\" of {len(alldata)} nucleotides\nJoint distribution of pseudotorsions in 3D RNA structures\n" + warn)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax = fig.add_subplot(133, projection='3d')
    ax.cla()
    hist, xedges, yedges = np.histogram2d(x, y, bins=200, range=[[xmin, xmax], [ymin, ymax]])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    ax.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.2, 0.2, hist.ravel(), zsort='average')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f"results/wadley_{angle}_{filter_helical}.png")
    if show:
        plt.show()

def stats_len(mappings_list, points):

    lengths = {}
    full_lengths = {}
    for f in sorted(mappings_list.keys()):
        lengths[f] = []
        full_lengths[f] = []
        for r in points:
            if r.family != f: continue
            nt_codes = r.df['nt_code'].values.tolist()
            lengths[f].append(len(nt_codes))
            full_lengths[f].append(len([ c for c in nt_codes if c != '-']))

    # then for all families
    lengths["all"] = []
    full_lengths["all"] = []
    for r in points:
        nt_codes = r.df['nt_code'].values.tolist()
        lengths["all"].append(len(nt_codes))
        full_lengths["all"].append(len([ c for c in nt_codes if c != '-']))
    dlengths = pd.DataFrame.from_dict(lengths, orient='index').transpose().drop(["all"], axis='columns').dropna(axis='columns', thresh=2)
    dfulllengths = pd.DataFrame.from_dict(full_lengths, orient='index').transpose().drop(["all"], axis='columns').dropna(axis='columns', thresh=2)
    print(dlengths.head())


    axs = dlengths.plot.hist(figsize=(10, 15), bins=range(0,650,50), sharex=True, sharey=True, subplots=True, layout=(12,6), legend=False, log=True)
    # for ax, f in zip(axs, sorted(mappings_list.keys())):
    #     ax.text(600,150, str(len([ x for x in lengths[f] if x != np.NaN ])), fontsize=14)
    plt.savefig("results/length_distribs.png")

    axs = dfulllengths.plot.hist(figsize=(10, 15), bins=range(0,650,50), sharex=True, sharey=True, subplots=True, layout=(12,6), legend=False, log=True)
    # for ax, f in zip(axs, sorted(mappings_list.keys())):
    #     ax.text(600,150, str(len([ x for x in lengths[f] if x != np.NaN ])), fontsize=14)
    plt.savefig("results/full_length_distribs.png")

def to_dist_matrix(f):
    print(f)
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
    fam_arrays = []
    for f in sorted(mappings_list.keys()):
        if path.isfile("data/"+f+".npy"):
            fam_arrays.append(np.load("data/"+f+".npy"))
        else:
            # to_dist_matrix(f)
            # fam_arrays.append(np.load("data/"+f+".npy"))
            fam_arrays.append([])

    fig, axs = plt.subplots(11,7, figsize=(25,25))
    axs = axs.ravel()
    [axi.set_axis_off() for axi in axs]
    for f, D, ax in zip(sorted(mappings_list.keys()), fam_arrays, axs):
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
        im = ax.matshow(D, vmin=0, vmax=1, origin='lower')
        ax.set_title(f)
    fig.suptitle("Distance matrices of sequences from various families\nclustered by sequence identity (Ward's method)", fontsize="18")
    fig.tight_layout() 
    fig.subplots_adjust(top=0.92)
    fig.colorbar(im, ax=axs.tolist(), shrink=0.98)
    fig.savefig(f"results/distances.png")



if __name__ == "__main__":

    #TODO: compute nt frequencies, chain lengths

    #################################################################
    #               LOAD ALL FILES
    #################################################################

    print("Loading mappings list...")
    mappings_list = pd.read_csv(path_to_seq_data + "realigned/mappings_list.csv", sep=',', index_col=0).to_dict(orient='list')
    for k in mappings_list.keys():
        mappings_list[k] = [ x for x in mappings_list[k] if str(x) != 'nan' ]

    # print("Loading datapoints from file...")
    # rna_points = []
    # filelist = [path_to_3D_data+"/datapoints/"+f for f in os.listdir(path_to_3D_data+"/datapoints") if ".log" not in f and ".gz" not in f]
    # p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=read_cpu_number())
    # pbar = tqdm(total=len(filelist), desc="RNA files", position=0, leave=True)
    # for i, rna in enumerate(p.imap_unordered(load_rna_frome_file, filelist)):
    #     rna_points.append(rna)
    #     pbar.update(1)
    # pbar.close()
    # p.close()
    # p.join()
    # npoints = len(rna_points)
    # print(npoints, "RNA files loaded.")

    #################################################################
    #               Define threads for the tasks
    #################################################################
    # wadley_thr = []
    # wadley_thr.append(th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 1, 'filter_helical': "zone"}))
    # wadley_thr.append(th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 1, 'filter_helical': "form"}))
    # wadley_thr.append(th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 1, 'filter_helical': "both"}))
    # wadley_thr.append(th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 4, 'filter_helical': "form"}))
    # wadley_thr.append(th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 4, 'filter_helical': "form"}))
    # wadley_thr.append(th.Thread(target=reproduce_wadley_results, args=[rna_points], kwargs={'carbon': 4, 'filter_helical': "both"}))
    # seq_len_thr = th.Thread(target=partial(stats_len, mappings_list), args=[rna_points])
    # dist_thr = th.Thread(target=seq_idty, args=[mappings_list])

    # for t in wadley_thr:
    #     t.start()
    # seq_len_thr.start()
    # dist_thr.start()

    # for t in wadley_thr:
    #     t.join()
    # seq_len_thr.join()
    # dist_thr.join()


    # reproduce_wadley_results(rna_points)
    seq_idty(mappings_list)
    # stats_len(mappings_list, rna_points)

    