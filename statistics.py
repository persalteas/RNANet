#!/usr/bin/python3.8
import os
import numpy as np
import pandas as pd
import threading as th
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm 
from tqdm import tqdm
from multiprocessing import Pool
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

def load_rna_frome_file(path_to_textfile):
    return pd.read_csv(path_to_textfile, sep=',', header=0, engine="c", index_col=0)

def reproduce_wadley_results(dfs, show=True):
    all_etas = []
    all_thetas = []
    all_forms = []
    c = 0
    for df in dfs:
        all_etas += list(df['eta'].values)
        all_thetas += list(df['theta'].values)
        all_forms += list(df['form'].values)
        if (len([ x for x in df['eta'].values if x < 0 or x > 7]) or 
            len([ x for x in df['theta'].values if x < 0 or x > 7])):
            c += 1
    print(c,"points on",len(dfs),"have non-radian angles !")


    print("combining etas and thetas...")
    # # increase all the angles by 180°
    # alldata = [ ((e+360)%360-180, (t+360)%360-180) 
    #             for e, t in zip(all_etas, all_thetas) 
    #             if ('nan' not in str((e,t))) 
    #             and not(e<-150 and t<-110) and not (e>160 and t<-110) ]
    alldata = [ (e, t) 
                for e, t, f in zip(all_etas, all_thetas, all_forms) 
                if ('nan' not in str((e,t))) 
                and f == '.' ]
    print(len(alldata), "couples of non-helical nts found.")

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
    
    plt.axhline(y=0, alpha=0.5, color='black')
    plt.axvline(x=0, alpha=0.5, color='black')
    plt.scatter(x, y, s=1, alpha=0.1)
    plt.contourf(xx, yy, z, cmap=cm.BuPu, alpha=0.5)
    ax.set_xlabel("$\\eta'=C_1'^{i-1}-P^i-C_1'^i-P^{i+1}$")
    ax.set_ylabel("$\\theta'=P^i-C_1'^i-P^{i+1}-C_1'^{i+1}$")
    # ax.add_patch(ptch.Rectangle((-20,0),50,70, linewidth=1, edgecolor='r', facecolor='#ff000080'))

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(xx, yy, z_inc, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_title("\"Wadley plot\"\n$\\eta'$, $\\theta'$ pseudotorsions in 3D RNA structures\n(Massive peak removed in the red zone, = double helices)")
    ax.set_xlabel("$\\eta'=C_1'^{i-1}-P^i-C_1'^i-P^{i+1}$")
    ax.set_ylabel("$\\theta'=P^i-C_1'^i-P^{i+1}-C_1'^{i+1}$")

    ax = fig.add_subplot(133, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=300, range=[[xmin, xmax], [ymin, ymax]])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    ax.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.5, 0.5, hist.ravel(), zsort='average')
    ax.set_xlabel("$\\eta'=C_1'^{i-1}-P^i-C_1'^i-P^{i+1}$")
    ax.set_ylabel("$\\theta'=P^i-C_1'^i-P^{i+1}-C_1'^{i+1}$")
    plt.savefig("results/clusters_rot180.png")
    if show:
        plt.show()

def stats_len(dfs):
    lengths = []
    full_lengths = []
    for r in dfs:
        nt_codes = r['nt_code'].values.tolist()
        lengths.append(len(nt_codes))
        full_lengths.append(len([ c for c in nt_codes if c != '-']))



if __name__ == "__main__":

    #TODO: compute nt frequencies, chain lengths

    #################################################################
    #               LOAD ALL FILES
    #################################################################
    print("Loading mappings list...")
    mappings_list = pd.read_csv(path_to_seq_data + "realigned/mappings_list.csv", sep=',', index_col=0).to_dict()

    print("Loading datapoints from file...")
    filelist = [path_to_3D_data+"/datapoints/"+f for f in os.listdir(path_to_3D_data+"/datapoints") if ".log" not in f and ".gz" not in f]
    rna_points = []
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=read_cpu_number())
    pbar = tqdm(total=len(filelist), desc="RNA files", position=0, leave=True)
    for i, rna in enumerate(p.imap_unordered(load_rna_frome_file, filelist)):
        rna_points.append(rna)
        pbar.update(1)
    pbar.close()
    p.close()
    p.join()
    npoints = len(rna_points)
    print(npoints, "RNA files loaded.")

    #################################################################
    #               Define threads for the tasks
    #################################################################
    wadley_thr = th.Thread(target=reproduce_wadley_results, args=[rna_points])


    wadley_thr.start()
    wadley_thr.join()

    