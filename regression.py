#!/usr/bin/python3.8
# This file is supposed to propose regression models on the computation time and mem usage of the re-alignment jobs.
# Light jobs are monitored by the Monitor class in RNAnet.py, and the measures are saved in jobstats.csv.
# This was done to guess the amount of memory required to re-align the large ribosomal subunit families RF02541 and RF02543.
# INFO: Our home hardware was a 32-core VM with 50GB RAM + 8GB Swap.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy, os
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

jobstats = pd.read_csv("data/jobstats.csv", sep=",")
families = pd.read_csv("data/statistics.csv", sep=",")

computed_families = []
comptimes = []
maxmem = []
nchains = []
maxlengths = []

for index, fam in jobstats.iterrows():
    if fam["max_mem"] != -1 and fam["comp_time"] != -1:
        rfam_acc = fam["label"].split(' ')[1]
        computed_families.append(rfam_acc)
        comptimes.append(fam["comp_time"])
        maxmem.append(fam["max_mem"])
        nchains.append(
            families.loc[families["rfam_acc"] == rfam_acc, "total_seqs"].values[0])
        maxlengths.append(
            families.loc[families["rfam_acc"] == rfam_acc, "maxlength"].values[0])

comptimes = [x/3600 for x in comptimes]  # compte en heures
maxlengths = [x/1000 for x in maxlengths]  # compte en kB
maxmem = [x/1024/1024 for x in maxmem]  # compte en MB

summary = pd.DataFrame({"family": computed_families, "n_chains": nchains,
                        "max_length(kB)": maxlengths, "comp_time(h)": comptimes, "max_mem(MB)": maxmem})
summary.sort_values("max_length(kB)", inplace=True)
summary.to_csv("results/summary.csv")

# ========================================================
# Plot the data
# ========================================================

fig = plt.figure(figsize=(12,8), dpi=100)

plt.subplot(231)
plt.scatter(summary.n_chains, summary["max_mem(MB)"])
plt.xlabel("Number of sequences")
plt.ylabel("Peak memory (MB)")

plt.subplot(232)
plt.scatter(summary["max_length(kB)"], summary["max_mem(MB)"])
plt.xlabel("Maximum length of sequences (kB)")
plt.ylabel("Peak memory (MB)")

ax = fig.add_subplot(233, projection='3d')
ax.scatter(summary.n_chains, summary["max_length(kB)"], summary["max_mem(MB)"])
ax.set_xlabel("Number of sequences")
ax.set_ylabel("Maximum length of sequences (kB)")
ax.set_zlabel("Peak memory (MB)")

plt.subplot(234)
plt.scatter(summary.n_chains, summary["comp_time(h)"])
plt.xlabel("Number of sequences")
plt.ylabel("Computation time (h)")

plt.subplot(235)
plt.scatter(summary["max_length(kB)"], summary["comp_time(h)"])
plt.xlabel("Maximum length of sequences (kB)")
plt.ylabel("Computation time (h)")

ax = fig.add_subplot(236, projection='3d')
ax.scatter(summary.n_chains, summary["max_length(kB)"], summary["comp_time(h)"])
ax.set_xlabel("Number of sequences")
ax.set_ylabel("Maximum length of sequences (kB)")
ax.set_zlabel("Computation time (h)")

plt.subplots_adjust(wspace=0.4)
plt.savefig("results/realign_jobs_performance.png")

# # ========================================================
# # Linear Regression of max_mem as function of max_length
# # ========================================================

# # With scikit-learn
# model = LinearRegression(normalize=True, n_jobs=-1)
# model.fit(summary["max_length(kB)"].values.reshape(-1, 1), summary["max_mem(MB)"])
# b0 = model.intercept_
# b1 = model.coef_[0]
# print(f"peak_mem = {b0:.0f} + {b1:.0f} * max_length")

# # with scipy
# coeffs = scipy.optimize.curve_fit(  lambda t, B0, B1: B0+np.exp(B1*t), 
#                                     summary["max_length(kB)"].values, 
#                                     summary["max_mem(MB)"].values
#                                  )[0]
# print(f"peak_mem = {coeffs[0]:.0f} + e^({coeffs[1]:.0f} * max_length)")
# coeffs_log = scipy.optimize.curve_fit(  lambda t, B0, B1: B0+B1*np.log(t),
#                                         summary["max_length(kB)"].values, 
#                                         summary["max_mem(MB)"].values,
#                                         p0=(400, 12000)
#                                      )[0]
# print(f"peak_mem = {coeffs_log[0]:.0f} + {coeffs_log[1]:.0f} * log(max_length)")

# # Re-plot
# x = np.linspace(0, 10, 1000)
# plt.figure()
# plt.scatter(summary["max_length(kB)"], summary["max_mem(MB)"])
# plt.xlabel("Maximum length of sequences (kB)")
# plt.ylabel("Peak memory (MB)")
# plt.plot(x, b0 + b1*x, "-r", label="linear fit")
# plt.plot(x, coeffs[0] + np.exp(coeffs[1]*x), "-g", label="expo fit")
# plt.plot(x, coeffs_log[0] + coeffs_log[1]*np.log(x), "-b", label="log fit")
# plt.legend()
# plt.savefig("results/regression/memory_linear_model.png")

# # ========================================================
# # Linear Regression of comp_time as function of n_chains
# # ========================================================

# # With scikit-learn
# model = LinearRegression(normalize=True, n_jobs=-1)
# model.fit(summary.n_chains.values.reshape(-1, 1), summary["comp_time(h)"])
# b0 = model.intercept_
# b1 = model.coef_[0]
# print(f"comp_time = {b0:.3f} + {b1:.3f} * n_chains")

# # Re-plot
# x = np.linspace(0, 500000, 1000)
# plt.figure()
# plt.scatter(summary.n_chains, summary["comp_time(h)"])
# plt.xlabel("Number of sequences")
# plt.ylabel("Computation time (h)")
# plt.plot(x, b0 + b1*x, "-r", label="linear fit")
# plt.legend()
# plt.savefig("results/regression/comp_time_linear_model.png")
