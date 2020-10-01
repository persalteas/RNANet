#!/usr/bin/python3.8
# This file is supposed to propose regression models on the computation time and mem usage of the re-alignment jobs.
# Jobs are monitored by the Monitor class in RNAnet.py, and the measures are saved in jobstats.csv.
# This was done to guess the amount of memory required to re-align the large ribosomal subunit families RF02541 and RF02543.
# INFO: Our home hardware was a 32-core VM with 50GB RAM

# The conclusion of this was to move to SINA for ribosomal subunits. 
# However, this was before we use cmalign with --small, which is after all required for RF00005, RF00382 and RF01852 
# (we do not understand why the two last very small families require that much memory). 
# Feedback would be appreciated on wether it is better to 
#   - Use a specialised database (SILVA) : better alignments (we guess?), but two kind of jobs
#   - Use cmalign --small everywhere (homogeneity)
# Moreover, --small requires --nonbanded --cyk, which means the output alignement is the optimally scored one. 
# To date, we trust Infernal as the best tool to realign ncRNA. Is it ?

# Contact: louis.becquey@univ-evry.fr (PhD student), fariza.tahi@univ-evry.fr (PI)

# Running this file is not required to compute the dataset.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy, os, sqlite3
# from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
pd.set_option('display.max_rows', None)

LSU_set = ["RF00002", "RF02540", "RF02541", "RF02543", "RF02546"]   # From Rfam CLAN 00112
SSU_set = ["RF00177", "RF02542",  "RF02545", "RF01959", "RF01960"]  # From Rfam CLAN 00111

with sqlite3.connect(os.getcwd()+"/results/RNANet.db") as conn:
    df = pd.read_sql("SELECT rfam_acc, max_len, nb_total_homol, comput_time, comput_peak_mem FROM family;", conn)

to_remove = [ f for f in df.rfam_acc if f in LSU_set+SSU_set ]
df = df.set_index('rfam_acc').drop(to_remove)
print(df)

# ========================================================
# Plot the data
# ========================================================

fig = plt.figure(figsize=(12,8), dpi=100)

plt.subplot(231)
plt.scatter(df.nb_total_homol, df.comput_peak_mem)
plt.xlabel("Number of sequences")
plt.ylabel("Peak memory (MB)")

plt.subplot(232)
plt.scatter(df.max_len, df.comput_peak_mem)
plt.xlabel("Maximum length of sequences ")
plt.ylabel("Peak memory (MB)")

ax = fig.add_subplot(233, projection='3d')
ax.scatter(df.nb_total_homol, df.max_len, df.comput_peak_mem)
ax.set_xlabel("Number of sequences")
ax.set_ylabel("Maximum length of sequences ")
ax.set_zlabel("Peak memory (MB)")

plt.subplot(234)
plt.scatter(df.nb_total_homol, df.comput_time)
plt.xlabel("Number of sequences")
plt.ylabel("Computation time (s)")

plt.subplot(235)
plt.scatter(df.max_len, df.comput_time)
plt.xlabel("Maximum length of sequences ")
plt.ylabel("Computation time (s)")

ax = fig.add_subplot(236, projection='3d')
ax.scatter(df.nb_total_homol, df.max_len, df.comput_time)
ax.set_xlabel("Number of sequences")
ax.set_ylabel("Maximum length of sequences ")
ax.set_zlabel("Computation time (s)")

plt.subplots_adjust(wspace=0.4)
plt.savefig(os.getcwd()+"/results/cmalign_jobs_performance.png")

# # ========================================================
# # Linear Regression of max_mem as function of max_length
# # ========================================================

# # With scikit-learn
# model = LinearRegression(normalize=True, n_jobs=-1)
# model.fit(df.max_len.values.reshape(-1, 1), df.comput_peak_mem)
# b0 = model.intercept_
# b1 = model.coef_[0]
# print(f"peak_mem = {b0:.0f} + {b1:.0f} * max_length")

# # with scipy
# coeffs = scipy.optimize.curve_fit(  lambda t, B0, B1: B0+np.exp(B1*t), 
#                                     df.max_len.values, 
#                                     df.comput_peak_mem.values
#                                  )[0]
# print(f"peak_mem = {coeffs[0]:.0f} + e^({coeffs[1]:.0f} * max_length)")
# coeffs_log = scipy.optimize.curve_fit(  lambda t, B0, B1: B0+B1*np.log(t),
#                                         df.max_len.values, 
#                                         df.comput_peak_mem.values,
#                                         p0=(400, 12000)
#                                      )[0]
# print(f"peak_mem = {coeffs_log[0]:.0f} + {coeffs_log[1]:.0f} * log(max_length)")

# # Re-plot
# x = np.linspace(0, 10, 1000)
# plt.figure()
# plt.scatter(df.max_len, df.comput_peak_mem)
# plt.xlabel("Maximum length of sequences ")
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
# model.fit(df.nb_total_homol.values.reshape(-1, 1), df.comput_time)
# b0 = model.intercept_
# b1 = model.coef_[0]
# print(f"comp_time = {b0:.3f} + {b1:.3f} * n_chains")

# # Re-plot
# x = np.linspace(0, 500000, 1000)
# plt.figure()
# plt.scatter(df.nb_total_homol, df.comput_time)
# plt.xlabel("Number of sequences")
# plt.ylabel("Computation time (s)")
# plt.plot(x, b0 + b1*x, "-r", label="linear fit")
# plt.legend()
# plt.savefig("results/regression/comp_time_linear_model.png")
