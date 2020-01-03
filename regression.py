#!/usr/bin/python3
# This file is supposed to propose regression models on the computation time and mem usage of the re-alignment jobs.
# Light jobs are monitored by the Monitor class in RNAnet.py, and the measures are saved in jobstats.csv.
# This was done to guess the amount of memory required to re-align the large ribosomal subunit families RF02541 and RF02543.
# INFO: Our home hardware was a 24-core VM with 50GB RAM + 8GB Swap.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
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

nchains = [x/1000 for x in nchains]  # compte en milliers de séquences
comptimes = [x/3600 for x in comptimes]  # compte en heures
maxlengths = [x/1000 for x in maxlengths]  # compte en kB
maxmem = [x/1024/1024 for x in maxmem]  # compte en MB

summary = pd.DataFrame({"family": computed_families, "n_chains": nchains,
                        "max_length": maxlengths, "comp_time": comptimes, "max_mem": maxmem})
summary.sort_values("max_length", inplace=True)
summary.to_csv("summary.csv")

# ========================================================
# Plot the data
# ========================================================

fig = plt.figure(dpi=100)
plt.subplot(231)
plt.scatter(summary.n_chains, summary.max_mem)
plt.xlabel("Number of sequences (x1000 seqs)")
plt.ylabel("Peak memory (MB)")
plt.subplot(232)
plt.scatter(summary.max_length, summary.max_mem)
plt.xlabel("Maximum length of sequences (kB)")
plt.ylabel("Peak memory (MB)")
ax = fig.add_subplot(233, projection='3d')
ax.scatter(summary.n_chains, summary.max_length, summary.max_mem)
ax.set_xlabel("Number of sequences (x1000 seqs)")
ax.set_ylabel("Maximum length of sequences (kB)")
ax.set_zlabel("Peak memory (MB)")
plt.subplot(234)
plt.scatter(summary.n_chains, summary.comp_time)
plt.xlabel("Number of sequences (x1000 seqs)")
plt.ylabel("Computation time (h)")
plt.subplot(235)
plt.scatter(summary.max_length, summary.comp_time)
plt.xlabel("Maximum length of sequences (kB)")
plt.ylabel("Computation time (h)")
ax = fig.add_subplot(236, projection='3d')
ax.scatter(summary.n_chains, summary.max_length, summary.comp_time)
ax.set_xlabel("Number of sequences (x1000 seqs)")
ax.set_ylabel("Maximum length of sequences (kB)")
ax.set_zlabel("Computation time (h)")
plt.show()

# ========================================================
# Linear Regression of max_mem as function of max_length
# ========================================================

# With scikit-learn
model = LinearRegression(normalize=True, n_jobs=-1)
model.fit(np.array(summary.max_length).reshape(-1, 1), summary.max_mem)
b0 = model.intercept_
b1 = model.coef_[0]
print(f"peak_mem = {b0:.0f} + {b1:.0f} * max_length")

# with scipy
coeffs = scipy.optimize.curve_fit(lambda t, B0, B1: B0+np.exp(B1*t),
                                  np.array(summary.max_length[:-3]), np.array(summary.max_mem[:-3]))[0]
print(f"peak_mem = {coeffs[0]:.0f} + e^({coeffs[1]:.0f} * max_length)")
coeffs_log = scipy.optimize.curve_fit(lambda t, B0, B1: B0+B1*np.log(t),
                                      np.array(summary.max_length), np.array(summary.max_mem), p0=(400, 12000))[0]
print(
    f"peak_mem = {coeffs_log[0]:.0f} + {coeffs_log[1]:.0f} * log(max_length)")

# Re-plot
x = np.linspace(0, 10, 1000)
plt.figure()
plt.scatter(summary.max_length, summary.max_mem)
plt.xlabel("Maximum length of sequences (kB)")
plt.ylabel("Peak memory (MB)")
plt.plot(x, b0 + b1*x, "-r", label="linear fit")
plt.plot(x, coeffs[0] + np.exp(coeffs[1]*x), "-g", label="expo fit on [:-3]")
plt.plot(x, coeffs_log[0] + coeffs_log[1]*np.log(x), "-b", label="log fit")
plt.ylim(0, 60000)
plt.legend()
plt.show()

print("Estimated mem required to compute RF02543 and its 11kB sequences:",
      model.predict(np.array([11]).reshape(-1, 1)))

# ========================================================
# Linear Regression of comp_time as function of n_chains
# ========================================================

# With scikit-learn
model = LinearRegression(normalize=True, n_jobs=-1)
model.fit(np.array(summary.n_chains).reshape(-1, 1), summary.comp_time)
b0 = model.intercept_
b1 = model.coef_[0]
print(f"comp_time = {b0:.3f} + {b1:.3f} * n_chains")
print("Estimated computation time required for RF02543 and its 38k sequences:",
      model.predict(np.array([38]).reshape(-1, 1)))

# Re-plot
x = np.linspace(0, 500, 1000)
plt.figure()
plt.scatter(summary.n_chains, summary.comp_time)
plt.xlabel("Number of sequences (x1000)")
plt.ylabel("Computation time (h)")
plt.plot(x, b0 + b1*x, "-r", label="linear fit")
plt.ylim(0, 10)
plt.legend()
plt.show()
