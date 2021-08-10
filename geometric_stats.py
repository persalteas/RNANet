#!/usr/bin/python3

# RNANet statistics
# Developed by Aglaé Tabot & Louis Becquey, 2021 

# This file computes additional geometric measures over the produced dataset,
# and estimates their distribtuions through Gaussian mixture models.
# THIS FILE IS NOT SUPPOSED TO BE RUN DIRECTLY.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import Bio, glob, json, os, random, sqlite3, warnings
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.vectors import Vector, calc_angle, calc_dihedral
from multiprocessing import Pool, Value
from pandas.core.common import SettingWithCopyWarning
from setproctitle import setproctitle
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from RNAnet import init_with_tqdm, trace_unhandled_exceptions, warn, notify

runDir = os.getcwd()

# This dic stores the number laws to use in the GMM to estimate each parameter's distribution.
# If you do not want to trust this data, you can use the --rescan-nmodes option.
# GMMs will be trained between 1 and 8 modes and the best model will be kept.
modes_data = {  
    # bonded distances, all-atom, common to all. Some are also used for HiRE-RNA.
    "C1'-C2'":3, "C2'-C3'":2, "C2'-O2'":2, "C3'-O3'":2, "C4'-C3'":2, "C4'-O4'":2, "C5'-C4'":2, "O3'-P":3, "O4'-C1'":3, "O5'-C5'":3, "P-O5'":3, "P-OP1":2, "P-OP2":2,
    
    # bonded distances, all-atom, purines
    "C4-C5":3, "C4-N9":2, "N3-C4":2, "C2-N3":2, "C2-N2":5, "N1-C2":3, "C6-N1":3, "C6-N6":3, "C6-O6":3, "C5-C6":2, "N7-C5":3, "C8-N7":2, "N9-C8":4, "C1'-N9":2,

    # bonded distances, all-atom, pyrimidines
    "C4-O4":2, "C4-N4":2, "C2-N1":1, "C2-O2":3, "N3-C2":4, "C4-N3":4, "C5-C4":2, "C6-C5":3, "N1-C6":2, "C1'-N1":2,

    # torsions, all atom
    "Alpha":3, "Beta":2, "Delta":2, "Epsilon":2, "Gamma":3, "Xhi":3, "Zeta":3,

    # Pyle, distances
    "C1'-P":3, "C4'-P":3, "P-C1'":3, "P-C4'":3,

    # Pyle, angles
    "C1'-P°-C1'°":3, "P-C1'-P°":2,

    # Pyle, torsions
    "Eta":1, "Theta":1, "Eta'":1, "Theta'":1, "Eta''":4, "Theta''":3,

    # HiRE-RNA, distances
    "C4'-P":3, "C4'-C1'":3, "C1'-B1":3, "B1-B2":2,

    # HiRE-RNA, angles
    "P-O5'-C5'":2, "O5'-C5'-C4'":1, "C5'-C4'-P":2, "C5'-C4'-C1'":2, "C4'-P-O5'":2, "C4'-C1'-B1":2, "C1'-C4'-P":2, "C1'-B1-B2":2,

    # HiRE-RNA, torsions
    "P-O5'-C5'-C4'":1, "O5'-C5'-C4'-P°":3, "O5'-C5'-C4'-C1'":3, "C5'-C4'-P°-O5'°":3, "C5'-C4'-C1'-B1":2, "C4'-P°-O5'°-C5'°":3, "C4'-C1'-B1-B2":3, "C1'-C4'-P°-O5'°":3,

    # HiRE-RNA, basepairs
    "cWW_AA_tips_distance":3, "cWW_AA_C1'-B1-B1pair":1, "cWW_AA_B1-B1pair-C1'pair":1, "cWW_AA_C4'-C1'-B1-B1pair":2, "cWW_AA_B1-B1pair-C1'pair-C4'pair":3, "cWW_AA_alpha_1":2, "cWW_AA_alpha_2":3, "cWW_AA_dB1":3, "cWW_AA_dB2":3, 
    "tWW_AA_tips_distance":1, "tWW_AA_C1'-B1-B1pair":1, "tWW_AA_B1-B1pair-C1'pair":1, "tWW_AA_C4'-C1'-B1-B1pair":2, "tWW_AA_B1-B1pair-C1'pair-C4'pair":3, "tWW_AA_alpha_1":2, "tWW_AA_alpha_2":1, "tWW_AA_dB1":1, "tWW_AA_dB2":2, 
    "cWH_AA_tips_distance":3, "cWH_AA_C1'-B1-B1pair":2, "cWH_AA_B1-B1pair-C1'pair":2, "cWH_AA_C4'-C1'-B1-B1pair":2, "cWH_AA_B1-B1pair-C1'pair-C4'pair":2, "cWH_AA_alpha_1":1, "cWH_AA_alpha_2":2, "cWH_AA_dB1":3, "cWH_AA_dB2":2, 
    "tWH_AA_tips_distance":3, "tWH_AA_C1'-B1-B1pair":1, "tWH_AA_B1-B1pair-C1'pair":3, "tWH_AA_C4'-C1'-B1-B1pair":2, "tWH_AA_B1-B1pair-C1'pair-C4'pair":2, "tWH_AA_alpha_1":1, "tWH_AA_alpha_2":3, "tWH_AA_dB1":2, "tWH_AA_dB2":1, 
    "cHW_AA_tips_distance":1, "cHW_AA_C1'-B1-B1pair":2, "cHW_AA_B1-B1pair-C1'pair":2, "cHW_AA_C4'-C1'-B1-B1pair":3, "cHW_AA_B1-B1pair-C1'pair-C4'pair":2, "cHW_AA_alpha_1":2, "cHW_AA_alpha_2":2, "cHW_AA_dB1":3, "cHW_AA_dB2":2, 
    "tHW_AA_tips_distance":4, "tHW_AA_C1'-B1-B1pair":2, "tHW_AA_B1-B1pair-C1'pair":2, "tHW_AA_C4'-C1'-B1-B1pair":2, "tHW_AA_B1-B1pair-C1'pair-C4'pair":2, "tHW_AA_alpha_1":2, "tHW_AA_alpha_2":1, "tHW_AA_dB1":2, "tHW_AA_dB2":1, 
    "cWS_AA_tips_distance":2, "cWS_AA_C1'-B1-B1pair":2, "cWS_AA_B1-B1pair-C1'pair":2, "cWS_AA_C4'-C1'-B1-B1pair":2, "cWS_AA_B1-B1pair-C1'pair-C4'pair":1, "cWS_AA_alpha_1":2, "cWS_AA_alpha_2":2, "cWS_AA_dB1":2, "cWS_AA_dB2":1, 
    "tWS_AA_tips_distance":2, "tWS_AA_C1'-B1-B1pair":2, "tWS_AA_B1-B1pair-C1'pair":2, "tWS_AA_C4'-C1'-B1-B1pair":3, "tWS_AA_B1-B1pair-C1'pair-C4'pair":1, "tWS_AA_alpha_1":2, "tWS_AA_alpha_2":2, "tWS_AA_dB1":2, "tWS_AA_dB2":3, 
    "cSW_AA_tips_distance":3, "cSW_AA_C1'-B1-B1pair":3, "cSW_AA_B1-B1pair-C1'pair":2, "cSW_AA_C4'-C1'-B1-B1pair":1, "cSW_AA_B1-B1pair-C1'pair-C4'pair":2, "cSW_AA_alpha_1":2, "cSW_AA_alpha_2":2, "cSW_AA_dB1":1, "cSW_AA_dB2":1, 
    "tSW_AA_tips_distance":3, "tSW_AA_C1'-B1-B1pair":3, "tSW_AA_B1-B1pair-C1'pair":3, "tSW_AA_C4'-C1'-B1-B1pair":2, "tSW_AA_B1-B1pair-C1'pair-C4'pair":2, "tSW_AA_alpha_1":2, "tSW_AA_alpha_2":2, "tSW_AA_dB1":2, "tSW_AA_dB2":2, 
    "cHH_AA_tips_distance":4, "cHH_AA_C1'-B1-B1pair":2, "cHH_AA_B1-B1pair-C1'pair":3, "cHH_AA_C4'-C1'-B1-B1pair":3, "cHH_AA_B1-B1pair-C1'pair-C4'pair":3, "cHH_AA_alpha_1":2, "cHH_AA_alpha_2":3, "cHH_AA_dB1":3, "cHH_AA_dB2":1, 
    "tHH_AA_tips_distance":2, "tHH_AA_C1'-B1-B1pair":2, "tHH_AA_B1-B1pair-C1'pair":2, "tHH_AA_C4'-C1'-B1-B1pair":3, "tHH_AA_B1-B1pair-C1'pair-C4'pair":1, "tHH_AA_alpha_1":2, "tHH_AA_alpha_2":2, "tHH_AA_dB1":2, "tHH_AA_dB2":2, 
    "cSH_AA_tips_distance":2, "cSH_AA_C1'-B1-B1pair":2, "cSH_AA_B1-B1pair-C1'pair":1, "cSH_AA_C4'-C1'-B1-B1pair":3, "cSH_AA_B1-B1pair-C1'pair-C4'pair":2, "cSH_AA_alpha_1":2, "cSH_AA_alpha_2":2, "cSH_AA_dB1":4, "cSH_AA_dB2":1, 
    "tSH_AA_tips_distance":2, "tSH_AA_C1'-B1-B1pair":1, "tSH_AA_B1-B1pair-C1'pair":2, "tSH_AA_C4'-C1'-B1-B1pair":2, "tSH_AA_B1-B1pair-C1'pair-C4'pair":2, "tSH_AA_alpha_1":2, "tSH_AA_alpha_2":3, "tSH_AA_dB1":2, "tSH_AA_dB2":2, 
    "cHS_AA_tips_distance":3, "cHS_AA_C1'-B1-B1pair":2, "cHS_AA_B1-B1pair-C1'pair":2, "cHS_AA_C4'-C1'-B1-B1pair":2, "cHS_AA_B1-B1pair-C1'pair-C4'pair":1, "cHS_AA_alpha_1":2, "cHS_AA_alpha_2":2, "cHS_AA_dB1":1, "cHS_AA_dB2":4, 
    "tHS_AA_tips_distance":4, "tHS_AA_C1'-B1-B1pair":2, "tHS_AA_B1-B1pair-C1'pair":2, "tHS_AA_C4'-C1'-B1-B1pair":2, "tHS_AA_B1-B1pair-C1'pair-C4'pair":1, "tHS_AA_alpha_1":2, "tHS_AA_alpha_2":1, "tHS_AA_dB1":2, "tHS_AA_dB2":1, 
    "cSS_AA_tips_distance":6, "cSS_AA_C1'-B1-B1pair":3, "cSS_AA_B1-B1pair-C1'pair":3, "cSS_AA_C4'-C1'-B1-B1pair":2, "cSS_AA_B1-B1pair-C1'pair-C4'pair":2, "cSS_AA_alpha_1":3, "cSS_AA_alpha_2":3, "cSS_AA_dB1":3, "cSS_AA_dB2":5, 
    "tSS_AA_tips_distance":5, "tSS_AA_C1'-B1-B1pair":1, "tSS_AA_B1-B1pair-C1'pair":1, "tSS_AA_C4'-C1'-B1-B1pair":2, "tSS_AA_B1-B1pair-C1'pair-C4'pair":1, "tSS_AA_alpha_1":3, "tSS_AA_alpha_2":1, "tSS_AA_dB1":4, "tSS_AA_dB2":2, 
    "cWW_AC_tips_distance":2, "cWW_AC_C1'-B1-B1pair":1, "cWW_AC_B1-B1pair-C1'pair":2, "cWW_AC_C4'-C1'-B1-B1pair":2, "cWW_AC_B1-B1pair-C1'pair-C4'pair":2, "cWW_AC_alpha_1":1, "cWW_AC_alpha_2":2, "cWW_AC_dB1":3, "cWW_AC_dB2":3, 
    "tWW_AC_tips_distance":2, "tWW_AC_C1'-B1-B1pair":3, "tWW_AC_B1-B1pair-C1'pair":2, "tWW_AC_C4'-C1'-B1-B1pair":3, "tWW_AC_B1-B1pair-C1'pair-C4'pair":3, "tWW_AC_alpha_1":3, "tWW_AC_alpha_2":2, "tWW_AC_dB1":4, "tWW_AC_dB2":3, 
    "cWH_AC_tips_distance":5, "cWH_AC_C1'-B1-B1pair":2, "cWH_AC_B1-B1pair-C1'pair":2, "cWH_AC_C4'-C1'-B1-B1pair":1, "cWH_AC_B1-B1pair-C1'pair-C4'pair":2, "cWH_AC_alpha_1":2, "cWH_AC_alpha_2":2, "cWH_AC_dB1":4, "cWH_AC_dB2":4, 
    "tWH_AC_tips_distance":8, "tWH_AC_C1'-B1-B1pair":1, "tWH_AC_B1-B1pair-C1'pair":2, "tWH_AC_C4'-C1'-B1-B1pair":2, "tWH_AC_B1-B1pair-C1'pair-C4'pair":3, "tWH_AC_alpha_1":2, "tWH_AC_alpha_2":2, "tWH_AC_dB1":3, "tWH_AC_dB2":3, 
    "cHW_AC_tips_distance":2, "cHW_AC_C1'-B1-B1pair":2, "cHW_AC_B1-B1pair-C1'pair":2, "cHW_AC_C4'-C1'-B1-B1pair":3, "cHW_AC_B1-B1pair-C1'pair-C4'pair":2, "cHW_AC_alpha_1":2, "cHW_AC_alpha_2":3, "cHW_AC_dB1":2, "cHW_AC_dB2":5, 
    "tHW_AC_tips_distance":3, "tHW_AC_C1'-B1-B1pair":2, "tHW_AC_B1-B1pair-C1'pair":3, "tHW_AC_C4'-C1'-B1-B1pair":3, "tHW_AC_B1-B1pair-C1'pair-C4'pair":2, "tHW_AC_alpha_1":2, "tHW_AC_alpha_2":2, "tHW_AC_dB1":3, "tHW_AC_dB2":3, 
    "cWS_AC_tips_distance":3, "cWS_AC_C1'-B1-B1pair":2, "cWS_AC_B1-B1pair-C1'pair":1, "cWS_AC_C4'-C1'-B1-B1pair":2, "cWS_AC_B1-B1pair-C1'pair-C4'pair":1, "cWS_AC_alpha_1":2, "cWS_AC_alpha_2":1, "cWS_AC_dB1":1, "cWS_AC_dB2":1, 
    "tWS_AC_tips_distance":4, "tWS_AC_C1'-B1-B1pair":2, "tWS_AC_B1-B1pair-C1'pair":1, "tWS_AC_C4'-C1'-B1-B1pair":2, "tWS_AC_B1-B1pair-C1'pair-C4'pair":2, "tWS_AC_alpha_1":3, "tWS_AC_alpha_2":1, "tWS_AC_dB1":3, "tWS_AC_dB2":2, 
    "cSW_AC_tips_distance":6, "cSW_AC_C1'-B1-B1pair":2, "cSW_AC_B1-B1pair-C1'pair":2, "cSW_AC_C4'-C1'-B1-B1pair":2, "cSW_AC_B1-B1pair-C1'pair-C4'pair":2, "cSW_AC_alpha_1":3, "cSW_AC_alpha_2":2, "cSW_AC_dB1":2, "cSW_AC_dB2":3, 
    "tSW_AC_tips_distance":5, "tSW_AC_C1'-B1-B1pair":1, "tSW_AC_B1-B1pair-C1'pair":2, "tSW_AC_C4'-C1'-B1-B1pair":1, "tSW_AC_B1-B1pair-C1'pair-C4'pair":2, "tSW_AC_alpha_1":1, "tSW_AC_alpha_2":2, "tSW_AC_dB1":2, "tSW_AC_dB2":3, 
    "cHH_AC_tips_distance":5, "cHH_AC_C1'-B1-B1pair":2, "cHH_AC_B1-B1pair-C1'pair":2, "cHH_AC_C4'-C1'-B1-B1pair":2, "cHH_AC_B1-B1pair-C1'pair-C4'pair":1, "cHH_AC_alpha_1":3, "cHH_AC_alpha_2":3, "cHH_AC_dB1":3, "cHH_AC_dB2":4, 
    "tHH_AC_tips_distance":4, "tHH_AC_C1'-B1-B1pair":1, "tHH_AC_B1-B1pair-C1'pair":2, "tHH_AC_C4'-C1'-B1-B1pair":2, "tHH_AC_B1-B1pair-C1'pair-C4'pair":3, "tHH_AC_alpha_1":2, "tHH_AC_alpha_2":2, "tHH_AC_dB1":4, "tHH_AC_dB2":3, 
    "cSH_AC_tips_distance":3, "cSH_AC_C1'-B1-B1pair":1, "cSH_AC_B1-B1pair-C1'pair":3, "cSH_AC_C4'-C1'-B1-B1pair":1, "cSH_AC_B1-B1pair-C1'pair-C4'pair":2, "cSH_AC_alpha_1":1, "cSH_AC_alpha_2":1, "cSH_AC_dB1":2, "cSH_AC_dB2":6, 
    "tSH_AC_tips_distance":8, "tSH_AC_C1'-B1-B1pair":3, "tSH_AC_B1-B1pair-C1'pair":2, "tSH_AC_C4'-C1'-B1-B1pair":1, "tSH_AC_B1-B1pair-C1'pair-C4'pair":2, "tSH_AC_alpha_1":2, "tSH_AC_alpha_2":3, "tSH_AC_dB1":1, "tSH_AC_dB2":2, 
    "cHS_AC_tips_distance":4, "cHS_AC_C1'-B1-B1pair":1, "cHS_AC_B1-B1pair-C1'pair":1, "cHS_AC_C4'-C1'-B1-B1pair":2, "cHS_AC_B1-B1pair-C1'pair-C4'pair":1, "cHS_AC_alpha_1":1, "cHS_AC_alpha_2":1, "cHS_AC_dB1":3, "cHS_AC_dB2":2, 
    "tHS_AC_tips_distance":8, "tHS_AC_C1'-B1-B1pair":1, "tHS_AC_B1-B1pair-C1'pair":2, "tHS_AC_C4'-C1'-B1-B1pair":2, "tHS_AC_B1-B1pair-C1'pair-C4'pair":2, "tHS_AC_alpha_1":1, "tHS_AC_alpha_2":1, "tHS_AC_dB1":1, "tHS_AC_dB2":1, 
    "cSS_AC_tips_distance":2, "cSS_AC_C1'-B1-B1pair":2, "cSS_AC_B1-B1pair-C1'pair":2, "cSS_AC_C4'-C1'-B1-B1pair":1, "cSS_AC_B1-B1pair-C1'pair-C4'pair":1, "cSS_AC_alpha_1":2, "cSS_AC_alpha_2":1, "cSS_AC_dB1":1, "cSS_AC_dB2":5, 
    "tSS_AC_tips_distance":5, "tSS_AC_C1'-B1-B1pair":2, "tSS_AC_B1-B1pair-C1'pair":2, "tSS_AC_C4'-C1'-B1-B1pair":1, "tSS_AC_B1-B1pair-C1'pair-C4'pair":2, "tSS_AC_alpha_1":2, "tSS_AC_alpha_2":2, "tSS_AC_dB1":3, "tSS_AC_dB2":5, 
    "cWW_AG_tips_distance":3, "cWW_AG_C1'-B1-B1pair":1, "cWW_AG_B1-B1pair-C1'pair":1, "cWW_AG_C4'-C1'-B1-B1pair":2, "cWW_AG_B1-B1pair-C1'pair-C4'pair":2, "cWW_AG_alpha_1":1, "cWW_AG_alpha_2":1, "cWW_AG_dB1":1, "cWW_AG_dB2":1, 
    "tWW_AG_tips_distance":5, "tWW_AG_C1'-B1-B1pair":1, "tWW_AG_B1-B1pair-C1'pair":1, "tWW_AG_C4'-C1'-B1-B1pair":2, "tWW_AG_B1-B1pair-C1'pair-C4'pair":2, "tWW_AG_alpha_1":2, "tWW_AG_alpha_2":2, "tWW_AG_dB1":2, "tWW_AG_dB2":3, 
    "cWH_AG_tips_distance":4, "cWH_AG_C1'-B1-B1pair":1, "cWH_AG_B1-B1pair-C1'pair":1, "cWH_AG_C4'-C1'-B1-B1pair":2, "cWH_AG_B1-B1pair-C1'pair-C4'pair":2, "cWH_AG_alpha_1":3, "cWH_AG_alpha_2":1, "cWH_AG_dB1":2, "cWH_AG_dB2":1, 
    "tWH_AG_tips_distance":3, "tWH_AG_C1'-B1-B1pair":1, "tWH_AG_B1-B1pair-C1'pair":1, "tWH_AG_C4'-C1'-B1-B1pair":2, "tWH_AG_B1-B1pair-C1'pair-C4'pair":2, "tWH_AG_alpha_1":2, "tWH_AG_alpha_2":1, "tWH_AG_dB1":2, "tWH_AG_dB2":1, 
    "cHW_AG_tips_distance":2, "cHW_AG_C1'-B1-B1pair":2, "cHW_AG_B1-B1pair-C1'pair":1, "cHW_AG_C4'-C1'-B1-B1pair":2, "cHW_AG_B1-B1pair-C1'pair-C4'pair":1, "cHW_AG_alpha_1":1, "cHW_AG_alpha_2":2, "cHW_AG_dB1":2, "cHW_AG_dB2":2, 
    "tHW_AG_tips_distance":3, "tHW_AG_C1'-B1-B1pair":2, "tHW_AG_B1-B1pair-C1'pair":2, "tHW_AG_C4'-C1'-B1-B1pair":2, "tHW_AG_B1-B1pair-C1'pair-C4'pair":2, "tHW_AG_alpha_1":2, "tHW_AG_alpha_2":2, "tHW_AG_dB1":2, "tHW_AG_dB2":2, 
    "cWS_AG_tips_distance":1, "cWS_AG_C1'-B1-B1pair":3, "cWS_AG_B1-B1pair-C1'pair":1, "cWS_AG_C4'-C1'-B1-B1pair":1, "cWS_AG_B1-B1pair-C1'pair-C4'pair":1, "cWS_AG_alpha_1":2, "cWS_AG_alpha_2":2, "cWS_AG_dB1":2, "cWS_AG_dB2":1, 
    "tWS_AG_tips_distance":6, "tWS_AG_C1'-B1-B1pair":1, "tWS_AG_B1-B1pair-C1'pair":2, "tWS_AG_C4'-C1'-B1-B1pair":2, "tWS_AG_B1-B1pair-C1'pair-C4'pair":1, "tWS_AG_alpha_1":2, "tWS_AG_alpha_2":2, "tWS_AG_dB1":1, "tWS_AG_dB2":3, 
    "cSW_AG_tips_distance":4, "cSW_AG_C1'-B1-B1pair":1, "cSW_AG_B1-B1pair-C1'pair":2, "cSW_AG_C4'-C1'-B1-B1pair":1, "cSW_AG_B1-B1pair-C1'pair-C4'pair":2, "cSW_AG_alpha_1":1, "cSW_AG_alpha_2":2, "cSW_AG_dB1":3, "cSW_AG_dB2":1, 
    "tSW_AG_tips_distance":7, "tSW_AG_C1'-B1-B1pair":3, "tSW_AG_B1-B1pair-C1'pair":2, "tSW_AG_C4'-C1'-B1-B1pair":2, "tSW_AG_B1-B1pair-C1'pair-C4'pair":2, "tSW_AG_alpha_1":2, "tSW_AG_alpha_2":2, "tSW_AG_dB1":3, "tSW_AG_dB2":3, 
    "cHH_AG_tips_distance":2, "cHH_AG_C1'-B1-B1pair":2, "cHH_AG_B1-B1pair-C1'pair":4, "cHH_AG_C4'-C1'-B1-B1pair":3, "cHH_AG_B1-B1pair-C1'pair-C4'pair":2, "cHH_AG_alpha_1":2, "cHH_AG_alpha_2":3, "cHH_AG_dB1":1, "cHH_AG_dB2":2, 
    "tHH_AG_tips_distance":8, "tHH_AG_C1'-B1-B1pair":3, "tHH_AG_B1-B1pair-C1'pair":3, "tHH_AG_C4'-C1'-B1-B1pair":3, "tHH_AG_B1-B1pair-C1'pair-C4'pair":2, "tHH_AG_alpha_1":3, "tHH_AG_alpha_2":3, "tHH_AG_dB1":1, "tHH_AG_dB2":2, 
    "cSH_AG_tips_distance":5, "cSH_AG_C1'-B1-B1pair":2, "cSH_AG_B1-B1pair-C1'pair":2, "cSH_AG_C4'-C1'-B1-B1pair":2, "cSH_AG_B1-B1pair-C1'pair-C4'pair":2, "cSH_AG_alpha_1":3, "cSH_AG_alpha_2":1, "cSH_AG_dB1":1, "cSH_AG_dB2":3, 
    "tSH_AG_tips_distance":5, "tSH_AG_C1'-B1-B1pair":2, "tSH_AG_B1-B1pair-C1'pair":2, "tSH_AG_C4'-C1'-B1-B1pair":2, "tSH_AG_B1-B1pair-C1'pair-C4'pair":3, "tSH_AG_alpha_1":2, "tSH_AG_alpha_2":4, "tSH_AG_dB1":3, "tSH_AG_dB2":2, 
    "cHS_AG_tips_distance":1, "cHS_AG_C1'-B1-B1pair":3, "cHS_AG_B1-B1pair-C1'pair":1, "cHS_AG_C4'-C1'-B1-B1pair":3, "cHS_AG_B1-B1pair-C1'pair-C4'pair":1, "cHS_AG_alpha_1":2, "cHS_AG_alpha_2":3, "cHS_AG_dB1":1, "cHS_AG_dB2":2, 
    "tHS_AG_tips_distance":6, "tHS_AG_C1'-B1-B1pair":1, "tHS_AG_B1-B1pair-C1'pair":2, "tHS_AG_C4'-C1'-B1-B1pair":2, "tHS_AG_B1-B1pair-C1'pair-C4'pair":2, "tHS_AG_alpha_1":1, "tHS_AG_alpha_2":2, "tHS_AG_dB1":2, "tHS_AG_dB2":1, 
    "cSS_AG_tips_distance":2, "cSS_AG_C1'-B1-B1pair":2, "cSS_AG_B1-B1pair-C1'pair":2, "cSS_AG_C4'-C1'-B1-B1pair":2, "cSS_AG_B1-B1pair-C1'pair-C4'pair":1, "cSS_AG_alpha_1":2, "cSS_AG_alpha_2":1, "cSS_AG_dB1":2, "cSS_AG_dB2":4, 
    "tSS_AG_tips_distance":4, "tSS_AG_C1'-B1-B1pair":3, "tSS_AG_B1-B1pair-C1'pair":1, "tSS_AG_C4'-C1'-B1-B1pair":2, "tSS_AG_B1-B1pair-C1'pair-C4'pair":1, "tSS_AG_alpha_1":2, "tSS_AG_alpha_2":1, "tSS_AG_dB1":2, "tSS_AG_dB2":4, 
    "cWW_AU_tips_distance":3, "cWW_AU_C1'-B1-B1pair":1, "cWW_AU_B1-B1pair-C1'pair":2, "cWW_AU_C4'-C1'-B1-B1pair":3, "cWW_AU_B1-B1pair-C1'pair-C4'pair":2, "cWW_AU_alpha_1":3, "cWW_AU_alpha_2":1, "cWW_AU_dB1":4, "cWW_AU_dB2":2, 
    "tWW_AU_tips_distance":3, "tWW_AU_C1'-B1-B1pair":3, "tWW_AU_B1-B1pair-C1'pair":3, "tWW_AU_C4'-C1'-B1-B1pair":2, "tWW_AU_B1-B1pair-C1'pair-C4'pair":2, "tWW_AU_alpha_1":3, "tWW_AU_alpha_2":2, "tWW_AU_dB1":3, "tWW_AU_dB2":2, 
    "cWH_AU_tips_distance":5, "cWH_AU_C1'-B1-B1pair":2, "cWH_AU_B1-B1pair-C1'pair":2, "cWH_AU_C4'-C1'-B1-B1pair":2, "cWH_AU_B1-B1pair-C1'pair-C4'pair":2, "cWH_AU_alpha_1":1, "cWH_AU_alpha_2":3, "cWH_AU_dB1":3, "cWH_AU_dB2":3, 
    "tWH_AU_tips_distance":6, "tWH_AU_C1'-B1-B1pair":1, "tWH_AU_B1-B1pair-C1'pair":3, "tWH_AU_C4'-C1'-B1-B1pair":2, "tWH_AU_B1-B1pair-C1'pair-C4'pair":2, "tWH_AU_alpha_1":2, "tWH_AU_alpha_2":2, "tWH_AU_dB1":1, "tWH_AU_dB2":3, 
    "cHW_AU_tips_distance":3, "cHW_AU_C1'-B1-B1pair":3, "cHW_AU_B1-B1pair-C1'pair":3, "cHW_AU_C4'-C1'-B1-B1pair":2, "cHW_AU_B1-B1pair-C1'pair-C4'pair":2, "cHW_AU_alpha_1":1, "cHW_AU_alpha_2":2, "cHW_AU_dB1":2, "cHW_AU_dB2":2, 
    "tHW_AU_tips_distance":3, "tHW_AU_C1'-B1-B1pair":2, "tHW_AU_B1-B1pair-C1'pair":2, "tHW_AU_C4'-C1'-B1-B1pair":2, "tHW_AU_B1-B1pair-C1'pair-C4'pair":2, "tHW_AU_alpha_1":2, "tHW_AU_alpha_2":1, "tHW_AU_dB1":1, "tHW_AU_dB2":4, 
    "cWS_AU_tips_distance":2, "cWS_AU_C1'-B1-B1pair":1, "cWS_AU_B1-B1pair-C1'pair":1, "cWS_AU_C4'-C1'-B1-B1pair":2, "cWS_AU_B1-B1pair-C1'pair-C4'pair":1, "cWS_AU_alpha_1":2, "cWS_AU_alpha_2":2, "cWS_AU_dB1":2, "cWS_AU_dB2":5, 
    "tWS_AU_tips_distance":2, "tWS_AU_C1'-B1-B1pair":2, "tWS_AU_B1-B1pair-C1'pair":2, "tWS_AU_C4'-C1'-B1-B1pair":2, "tWS_AU_B1-B1pair-C1'pair-C4'pair":1, "tWS_AU_alpha_1":2, "tWS_AU_alpha_2":2, "tWS_AU_dB1":3, "tWS_AU_dB2":4, 
    "cSW_AU_tips_distance":2, "cSW_AU_C1'-B1-B1pair":3, "cSW_AU_B1-B1pair-C1'pair":2, "cSW_AU_C4'-C1'-B1-B1pair":2, "cSW_AU_B1-B1pair-C1'pair-C4'pair":2, "cSW_AU_alpha_1":3, "cSW_AU_alpha_2":2, "cSW_AU_dB1":2, "cSW_AU_dB2":3, 
    "tSW_AU_tips_distance":3, "tSW_AU_C1'-B1-B1pair":2, "tSW_AU_B1-B1pair-C1'pair":3, "tSW_AU_C4'-C1'-B1-B1pair":3, "tSW_AU_B1-B1pair-C1'pair-C4'pair":2, "tSW_AU_alpha_1":2, "tSW_AU_alpha_2":1, "tSW_AU_dB1":3, "tSW_AU_dB2":4, 
    "cHH_AU_tips_distance":6, "cHH_AU_C1'-B1-B1pair":2, "cHH_AU_B1-B1pair-C1'pair":1, "cHH_AU_C4'-C1'-B1-B1pair":2, "cHH_AU_B1-B1pair-C1'pair-C4'pair":1, "cHH_AU_alpha_1":2, "cHH_AU_alpha_2":2, "cHH_AU_dB1":1, "cHH_AU_dB2":2, 
    "tHH_AU_tips_distance":8, "tHH_AU_C1'-B1-B1pair":3, "tHH_AU_B1-B1pair-C1'pair":3, "tHH_AU_C4'-C1'-B1-B1pair":3, "tHH_AU_B1-B1pair-C1'pair-C4'pair":2, "tHH_AU_alpha_1":3, "tHH_AU_alpha_2":3, "tHH_AU_dB1":1, "tHH_AU_dB2":3, 
    "cSH_AU_tips_distance":5, "cSH_AU_C1'-B1-B1pair":1, "cSH_AU_B1-B1pair-C1'pair":3, "cSH_AU_C4'-C1'-B1-B1pair":3, "cSH_AU_B1-B1pair-C1'pair-C4'pair":2, "cSH_AU_alpha_1":2, "cSH_AU_alpha_2":1, "cSH_AU_dB1":4, "cSH_AU_dB2":4, 
    "tSH_AU_tips_distance":5, "tSH_AU_C1'-B1-B1pair":3, "tSH_AU_B1-B1pair-C1'pair":1, "tSH_AU_C4'-C1'-B1-B1pair":1, "tSH_AU_B1-B1pair-C1'pair-C4'pair":2, "tSH_AU_alpha_1":3, "tSH_AU_alpha_2":3, "tSH_AU_dB1":3, "tSH_AU_dB2":4, 
    "cHS_AU_tips_distance":2, "cHS_AU_C1'-B1-B1pair":3, "cHS_AU_B1-B1pair-C1'pair":1, "cHS_AU_C4'-C1'-B1-B1pair":2, "cHS_AU_B1-B1pair-C1'pair-C4'pair":2, "cHS_AU_alpha_1":2, "cHS_AU_alpha_2":2, "cHS_AU_dB1":1, "cHS_AU_dB2":3, 
    "tHS_AU_tips_distance":2, "tHS_AU_C1'-B1-B1pair":2, "tHS_AU_B1-B1pair-C1'pair":2, "tHS_AU_C4'-C1'-B1-B1pair":2, "tHS_AU_B1-B1pair-C1'pair-C4'pair":3, "tHS_AU_alpha_1":3, "tHS_AU_alpha_2":2, "tHS_AU_dB1":3, "tHS_AU_dB2":3, 
    "cSS_AU_tips_distance":3, "cSS_AU_C1'-B1-B1pair":2, "cSS_AU_B1-B1pair-C1'pair":2, "cSS_AU_C4'-C1'-B1-B1pair":1, "cSS_AU_B1-B1pair-C1'pair-C4'pair":2, "cSS_AU_alpha_1":3, "cSS_AU_alpha_2":2, "cSS_AU_dB1":1, "cSS_AU_dB2":4, 
    "tSS_AU_tips_distance":5, "tSS_AU_C1'-B1-B1pair":2, "tSS_AU_B1-B1pair-C1'pair":1, "tSS_AU_C4'-C1'-B1-B1pair":3, "tSS_AU_B1-B1pair-C1'pair-C4'pair":2, "tSS_AU_alpha_1":2, "tSS_AU_alpha_2":3, "tSS_AU_dB1":3, "tSS_AU_dB2":8, 
    "cWW_CA_tips_distance":2, "cWW_CA_C1'-B1-B1pair":2, "cWW_CA_B1-B1pair-C1'pair":1, "cWW_CA_C4'-C1'-B1-B1pair":2, "cWW_CA_B1-B1pair-C1'pair-C4'pair":2, "cWW_CA_alpha_1":1, "cWW_CA_alpha_2":2, "cWW_CA_dB1":1, "cWW_CA_dB2":1, 
    "tWW_CA_tips_distance":4, "tWW_CA_C1'-B1-B1pair":2, "tWW_CA_B1-B1pair-C1'pair":2, "tWW_CA_C4'-C1'-B1-B1pair":3, "tWW_CA_B1-B1pair-C1'pair-C4'pair":2, "tWW_CA_alpha_1":2, "tWW_CA_alpha_2":1, "tWW_CA_dB1":4, "tWW_CA_dB2":2, 
    "cWH_CA_tips_distance":3, "cWH_CA_C1'-B1-B1pair":3, "cWH_CA_B1-B1pair-C1'pair":2, "cWH_CA_C4'-C1'-B1-B1pair":2, "cWH_CA_B1-B1pair-C1'pair-C4'pair":3, "cWH_CA_alpha_1":3, "cWH_CA_alpha_2":2, "cWH_CA_dB1":5, "cWH_CA_dB2":2, 
    "tWH_CA_tips_distance":5, "tWH_CA_C1'-B1-B1pair":1, "tWH_CA_B1-B1pair-C1'pair":1, "tWH_CA_C4'-C1'-B1-B1pair":2, "tWH_CA_B1-B1pair-C1'pair-C4'pair":2, "tWH_CA_alpha_1":3, "tWH_CA_alpha_2":1, "tWH_CA_dB1":3, "tWH_CA_dB2":2, 
    "cHW_CA_tips_distance":2, "cHW_CA_C1'-B1-B1pair":2, "cHW_CA_B1-B1pair-C1'pair":2, "cHW_CA_C4'-C1'-B1-B1pair":2, "cHW_CA_B1-B1pair-C1'pair-C4'pair":2, "cHW_CA_alpha_1":2, "cHW_CA_alpha_2":2, "cHW_CA_dB1":4, "cHW_CA_dB2":2, 
    "tHW_CA_tips_distance":2, "tHW_CA_C1'-B1-B1pair":2, "tHW_CA_B1-B1pair-C1'pair":2, "tHW_CA_C4'-C1'-B1-B1pair":2, "tHW_CA_B1-B1pair-C1'pair-C4'pair":2, "tHW_CA_alpha_1":2, "tHW_CA_alpha_2":2, "tHW_CA_dB1":6, "tHW_CA_dB2":2, 
    "cWS_CA_tips_distance":2, "cWS_CA_C1'-B1-B1pair":2, "cWS_CA_B1-B1pair-C1'pair":2, "cWS_CA_C4'-C1'-B1-B1pair":2, "cWS_CA_B1-B1pair-C1'pair-C4'pair":1, "cWS_CA_alpha_1":2, "cWS_CA_alpha_2":2, "cWS_CA_dB1":4, "cWS_CA_dB2":2, 
    "tWS_CA_tips_distance":5, "tWS_CA_C1'-B1-B1pair":3, "tWS_CA_B1-B1pair-C1'pair":1, "tWS_CA_C4'-C1'-B1-B1pair":3, "tWS_CA_B1-B1pair-C1'pair-C4'pair":2, "tWS_CA_alpha_1":3, "tWS_CA_alpha_2":1, "tWS_CA_dB1":1, "tWS_CA_dB2":1, 
    "cSW_CA_tips_distance":1, "cSW_CA_C1'-B1-B1pair":1, "cSW_CA_B1-B1pair-C1'pair":1, "cSW_CA_C4'-C1'-B1-B1pair":1, "cSW_CA_B1-B1pair-C1'pair-C4'pair":2, "cSW_CA_alpha_1":1, "cSW_CA_alpha_2":3, "cSW_CA_dB1":1, "cSW_CA_dB2":1, 
    "tSW_CA_tips_distance":3, "tSW_CA_C1'-B1-B1pair":2, "tSW_CA_B1-B1pair-C1'pair":2, "tSW_CA_C4'-C1'-B1-B1pair":1, "tSW_CA_B1-B1pair-C1'pair-C4'pair":1, "tSW_CA_alpha_1":2, "tSW_CA_alpha_2":3, "tSW_CA_dB1":3, "tSW_CA_dB2":1, 
    "cHH_CA_tips_distance":5, "cHH_CA_C1'-B1-B1pair":2, "cHH_CA_B1-B1pair-C1'pair":1, "cHH_CA_C4'-C1'-B1-B1pair":3, "cHH_CA_B1-B1pair-C1'pair-C4'pair":1, "cHH_CA_alpha_1":2, "cHH_CA_alpha_2":1, "cHH_CA_dB1":1, "cHH_CA_dB2":2, 
    "tHH_CA_tips_distance":1, "tHH_CA_C1'-B1-B1pair":2, "tHH_CA_B1-B1pair-C1'pair":2, "tHH_CA_C4'-C1'-B1-B1pair":3, "tHH_CA_B1-B1pair-C1'pair-C4'pair":3, "tHH_CA_alpha_1":2, "tHH_CA_alpha_2":1, "tHH_CA_dB1":3, "tHH_CA_dB2":5, 
    "cSH_CA_tips_distance":3, "cSH_CA_C1'-B1-B1pair":1, "cSH_CA_B1-B1pair-C1'pair":3, "cSH_CA_C4'-C1'-B1-B1pair":2, "cSH_CA_B1-B1pair-C1'pair-C4'pair":1, "cSH_CA_alpha_1":1, "cSH_CA_alpha_2":1, "cSH_CA_dB1":2, "cSH_CA_dB2":3, 
    "tSH_CA_tips_distance":2, "tSH_CA_C1'-B1-B1pair":1, "tSH_CA_B1-B1pair-C1'pair":2, "tSH_CA_C4'-C1'-B1-B1pair":2, "tSH_CA_B1-B1pair-C1'pair-C4'pair":2, "tSH_CA_alpha_1":3, "tSH_CA_alpha_2":2, "tSH_CA_dB1":6, "tSH_CA_dB2":4, 
    "cHS_CA_tips_distance":2, "cHS_CA_C1'-B1-B1pair":2, "cHS_CA_B1-B1pair-C1'pair":2, "cHS_CA_C4'-C1'-B1-B1pair":1, "cHS_CA_B1-B1pair-C1'pair-C4'pair":1, "cHS_CA_alpha_1":1, "cHS_CA_alpha_2":2, "cHS_CA_dB1":2, "cHS_CA_dB2":2, 
    "tHS_CA_tips_distance":3, "tHS_CA_C1'-B1-B1pair":2, "tHS_CA_B1-B1pair-C1'pair":1, "tHS_CA_C4'-C1'-B1-B1pair":2, "tHS_CA_B1-B1pair-C1'pair-C4'pair":2, "tHS_CA_alpha_1":3, "tHS_CA_alpha_2":3, "tHS_CA_dB1":2, "tHS_CA_dB2":1, 
    "cSS_CA_tips_distance":7, "cSS_CA_C1'-B1-B1pair":2, "cSS_CA_B1-B1pair-C1'pair":2, "cSS_CA_C4'-C1'-B1-B1pair":1, "cSS_CA_B1-B1pair-C1'pair-C4'pair":1, "cSS_CA_alpha_1":3, "cSS_CA_alpha_2":3, "cSS_CA_dB1":3, "cSS_CA_dB2":1, 
    "tSS_CA_tips_distance":5, "tSS_CA_C1'-B1-B1pair":2, "tSS_CA_B1-B1pair-C1'pair":2, "tSS_CA_C4'-C1'-B1-B1pair":2, "tSS_CA_B1-B1pair-C1'pair-C4'pair":1, "tSS_CA_alpha_1":2, "tSS_CA_alpha_2":2, "tSS_CA_dB1":4, "tSS_CA_dB2":2, 
    "cWW_CC_tips_distance":3, "cWW_CC_C1'-B1-B1pair":1, "cWW_CC_B1-B1pair-C1'pair":1, "cWW_CC_C4'-C1'-B1-B1pair":2, "cWW_CC_B1-B1pair-C1'pair-C4'pair":2, "cWW_CC_alpha_1":1, "cWW_CC_alpha_2":2, "cWW_CC_dB1":2, "cWW_CC_dB2":2, 
    "tWW_CC_tips_distance":6, "tWW_CC_C1'-B1-B1pair":3, "tWW_CC_B1-B1pair-C1'pair":3, "tWW_CC_C4'-C1'-B1-B1pair":3, "tWW_CC_B1-B1pair-C1'pair-C4'pair":3, "tWW_CC_alpha_1":2, "tWW_CC_alpha_2":2, "tWW_CC_dB1":6, "tWW_CC_dB2":3, 
    "cWH_CC_tips_distance":4, "cWH_CC_C1'-B1-B1pair":2, "cWH_CC_B1-B1pair-C1'pair":2, "cWH_CC_C4'-C1'-B1-B1pair":2, "cWH_CC_B1-B1pair-C1'pair-C4'pair":1, "cWH_CC_alpha_1":1, "cWH_CC_alpha_2":3, "cWH_CC_dB1":3, "cWH_CC_dB2":2, 
    "tWH_CC_tips_distance":1, "tWH_CC_C1'-B1-B1pair":1, "tWH_CC_B1-B1pair-C1'pair":3, "tWH_CC_C4'-C1'-B1-B1pair":2, "tWH_CC_B1-B1pair-C1'pair-C4'pair":1, "tWH_CC_alpha_1":3, "tWH_CC_alpha_2":1, "tWH_CC_dB1":3, "tWH_CC_dB2":3, 
    "cHW_CC_tips_distance":4, "cHW_CC_C1'-B1-B1pair":3, "cHW_CC_B1-B1pair-C1'pair":2, "cHW_CC_C4'-C1'-B1-B1pair":1, "cHW_CC_B1-B1pair-C1'pair-C4'pair":2, "cHW_CC_alpha_1":2, "cHW_CC_alpha_2":2, "cHW_CC_dB1":2, "cHW_CC_dB2":3, 
    "tHW_CC_tips_distance":2, "tHW_CC_C1'-B1-B1pair":1, "tHW_CC_B1-B1pair-C1'pair":3, "tHW_CC_C4'-C1'-B1-B1pair":3, "tHW_CC_B1-B1pair-C1'pair-C4'pair":2, "tHW_CC_alpha_1":2, "tHW_CC_alpha_2":2, "tHW_CC_dB1":3, "tHW_CC_dB2":3, 
    "cWS_CC_tips_distance":3, "cWS_CC_C1'-B1-B1pair":2, "cWS_CC_B1-B1pair-C1'pair":2, "cWS_CC_C4'-C1'-B1-B1pair":1, "cWS_CC_B1-B1pair-C1'pair-C4'pair":1, "cWS_CC_alpha_1":2, "cWS_CC_alpha_2":3, "cWS_CC_dB1":2, "cWS_CC_dB2":1, 
    "tWS_CC_tips_distance":5, "tWS_CC_C1'-B1-B1pair":2, "tWS_CC_B1-B1pair-C1'pair":2, "tWS_CC_C4'-C1'-B1-B1pair":2, "tWS_CC_B1-B1pair-C1'pair-C4'pair":1, "tWS_CC_alpha_1":2, "tWS_CC_alpha_2":2, "tWS_CC_dB1":2, "tWS_CC_dB2":2, 
    "cSW_CC_tips_distance":3, "cSW_CC_C1'-B1-B1pair":2, "cSW_CC_B1-B1pair-C1'pair":2, "cSW_CC_C4'-C1'-B1-B1pair":2, "cSW_CC_B1-B1pair-C1'pair-C4'pair":1, "cSW_CC_alpha_1":3, "cSW_CC_alpha_2":2, "cSW_CC_dB1":2, "cSW_CC_dB2":2, 
    "tSW_CC_tips_distance":5, "tSW_CC_C1'-B1-B1pair":1, "tSW_CC_B1-B1pair-C1'pair":2, "tSW_CC_C4'-C1'-B1-B1pair":1, "tSW_CC_B1-B1pair-C1'pair-C4'pair":2, "tSW_CC_alpha_1":1, "tSW_CC_alpha_2":2, "tSW_CC_dB1":3, "tSW_CC_dB2":2, 
    "cHH_CC_tips_distance":5, "cHH_CC_C1'-B1-B1pair":1, "cHH_CC_B1-B1pair-C1'pair":1, "cHH_CC_C4'-C1'-B1-B1pair":1, "cHH_CC_B1-B1pair-C1'pair-C4'pair":1, "cHH_CC_alpha_1":2, "cHH_CC_alpha_2":1, "cHH_CC_dB1":7, "cHH_CC_dB2":7, 
    "tHH_CC_tips_distance":5, "tHH_CC_C1'-B1-B1pair":3, "tHH_CC_B1-B1pair-C1'pair":2, "tHH_CC_C4'-C1'-B1-B1pair":3, "tHH_CC_B1-B1pair-C1'pair-C4'pair":2, "tHH_CC_alpha_1":1, "tHH_CC_alpha_2":3, "tHH_CC_dB1":5, "tHH_CC_dB2":5, 
    "cSH_CC_tips_distance":3, "cSH_CC_C1'-B1-B1pair":2, "cSH_CC_B1-B1pair-C1'pair":2, "cSH_CC_C4'-C1'-B1-B1pair":2, "cSH_CC_B1-B1pair-C1'pair-C4'pair":2, "cSH_CC_alpha_1":3, "cSH_CC_alpha_2":2, "cSH_CC_dB1":5, "cSH_CC_dB2":2, 
    "tSH_CC_tips_distance":5, "tSH_CC_C1'-B1-B1pair":2, "tSH_CC_B1-B1pair-C1'pair":1, "tSH_CC_C4'-C1'-B1-B1pair":2, "tSH_CC_B1-B1pair-C1'pair-C4'pair":2, "tSH_CC_alpha_1":3, "tSH_CC_alpha_2":1, "tSH_CC_dB1":4, "tSH_CC_dB2":2, 
    "cHS_CC_tips_distance":3, "cHS_CC_C1'-B1-B1pair":2, "cHS_CC_B1-B1pair-C1'pair":2, "cHS_CC_C4'-C1'-B1-B1pair":2, "cHS_CC_B1-B1pair-C1'pair-C4'pair":2, "cHS_CC_alpha_1":3, "cHS_CC_alpha_2":2, "cHS_CC_dB1":2, "cHS_CC_dB2":2, 
    "tHS_CC_tips_distance":5, "tHS_CC_C1'-B1-B1pair":3, "tHS_CC_B1-B1pair-C1'pair":1, "tHS_CC_C4'-C1'-B1-B1pair":2, "tHS_CC_B1-B1pair-C1'pair-C4'pair":3, "tHS_CC_alpha_1":1, "tHS_CC_alpha_2":2, "tHS_CC_dB1":4, "tHS_CC_dB2":4, 
    "cSS_CC_tips_distance":5, "cSS_CC_C1'-B1-B1pair":2, "cSS_CC_B1-B1pair-C1'pair":2, "cSS_CC_C4'-C1'-B1-B1pair":2, "cSS_CC_B1-B1pair-C1'pair-C4'pair":1, "cSS_CC_alpha_1":1, "cSS_CC_alpha_2":3, "cSS_CC_dB1":1, "cSS_CC_dB2":3, 
    "tSS_CC_tips_distance":5, "tSS_CC_C1'-B1-B1pair":2, "tSS_CC_B1-B1pair-C1'pair":2, "tSS_CC_C4'-C1'-B1-B1pair":3, "tSS_CC_B1-B1pair-C1'pair-C4'pair":2, "tSS_CC_alpha_1":3, "tSS_CC_alpha_2":2, "tSS_CC_dB1":2, "tSS_CC_dB2":1, 
    "cWW_CG_tips_distance":5, "cWW_CG_C1'-B1-B1pair":2, "cWW_CG_B1-B1pair-C1'pair":1, "cWW_CG_C4'-C1'-B1-B1pair":2, "cWW_CG_B1-B1pair-C1'pair-C4'pair":2, "cWW_CG_alpha_1":2, "cWW_CG_alpha_2":3, "cWW_CG_dB1":2, "cWW_CG_dB2":2, 
    "tWW_CG_tips_distance":3, "tWW_CG_C1'-B1-B1pair":1, "tWW_CG_B1-B1pair-C1'pair":2, "tWW_CG_C4'-C1'-B1-B1pair":2, "tWW_CG_B1-B1pair-C1'pair-C4'pair":2, "tWW_CG_alpha_1":2, "tWW_CG_alpha_2":1, "tWW_CG_dB1":1, "tWW_CG_dB2":4, 
    "cWH_CG_tips_distance":3, "cWH_CG_C1'-B1-B1pair":1, "cWH_CG_B1-B1pair-C1'pair":1, "cWH_CG_C4'-C1'-B1-B1pair":2, "cWH_CG_B1-B1pair-C1'pair-C4'pair":2, "cWH_CG_alpha_1":2, "cWH_CG_alpha_2":1, "cWH_CG_dB1":4, "cWH_CG_dB2":2, 
    "tWH_CG_tips_distance":4, "tWH_CG_C1'-B1-B1pair":2, "tWH_CG_B1-B1pair-C1'pair":1, "tWH_CG_C4'-C1'-B1-B1pair":2, "tWH_CG_B1-B1pair-C1'pair-C4'pair":3, "tWH_CG_alpha_1":2, "tWH_CG_alpha_2":1, "tWH_CG_dB1":3, "tWH_CG_dB2":2, 
    "cHW_CG_tips_distance":3, "cHW_CG_C1'-B1-B1pair":2, "cHW_CG_B1-B1pair-C1'pair":2, "cHW_CG_C4'-C1'-B1-B1pair":1, "cHW_CG_B1-B1pair-C1'pair-C4'pair":2, "cHW_CG_alpha_1":1, "cHW_CG_alpha_2":2, "cHW_CG_dB1":2, "cHW_CG_dB2":2, 
    "tHW_CG_tips_distance":5, "tHW_CG_C1'-B1-B1pair":1, "tHW_CG_B1-B1pair-C1'pair":2, "tHW_CG_C4'-C1'-B1-B1pair":1, "tHW_CG_B1-B1pair-C1'pair-C4'pair":2, "tHW_CG_alpha_1":3, "tHW_CG_alpha_2":2, "tHW_CG_dB1":4, "tHW_CG_dB2":3, 
    "cWS_CG_tips_distance":2, "cWS_CG_C1'-B1-B1pair":1, "cWS_CG_B1-B1pair-C1'pair":1, "cWS_CG_C4'-C1'-B1-B1pair":1, "cWS_CG_B1-B1pair-C1'pair-C4'pair":1, "cWS_CG_alpha_1":1, "cWS_CG_alpha_2":2, "cWS_CG_dB1":2, "cWS_CG_dB2":3, 
    "tWS_CG_tips_distance":2, "tWS_CG_C1'-B1-B1pair":3, "tWS_CG_B1-B1pair-C1'pair":1, "tWS_CG_C4'-C1'-B1-B1pair":2, "tWS_CG_B1-B1pair-C1'pair-C4'pair":1, "tWS_CG_alpha_1":2, "tWS_CG_alpha_2":1, "tWS_CG_dB1":2, "tWS_CG_dB2":4, 
    "cSW_CG_tips_distance":7, "cSW_CG_C1'-B1-B1pair":1, "cSW_CG_B1-B1pair-C1'pair":2, "cSW_CG_C4'-C1'-B1-B1pair":2, "cSW_CG_B1-B1pair-C1'pair-C4'pair":3, "cSW_CG_alpha_1":1, "cSW_CG_alpha_2":2, "cSW_CG_dB1":1, "cSW_CG_dB2":3, 
    "tSW_CG_tips_distance":4, "tSW_CG_C1'-B1-B1pair":1, "tSW_CG_B1-B1pair-C1'pair":2, "tSW_CG_C4'-C1'-B1-B1pair":3, "tSW_CG_B1-B1pair-C1'pair-C4'pair":2, "tSW_CG_alpha_1":1, "tSW_CG_alpha_2":2, "tSW_CG_dB1":7, "tSW_CG_dB2":2, 
    "cHH_CG_tips_distance":1, "cHH_CG_C1'-B1-B1pair":1, "cHH_CG_B1-B1pair-C1'pair":2, "cHH_CG_C4'-C1'-B1-B1pair":3, "cHH_CG_B1-B1pair-C1'pair-C4'pair":2, "cHH_CG_alpha_1":1, "cHH_CG_alpha_2":2, "cHH_CG_dB1":4, "cHH_CG_dB2":1, 
    "tHH_CG_tips_distance":8, "tHH_CG_C1'-B1-B1pair":2, "tHH_CG_B1-B1pair-C1'pair":2, "tHH_CG_C4'-C1'-B1-B1pair":3, "tHH_CG_B1-B1pair-C1'pair-C4'pair":2, "tHH_CG_alpha_1":2, "tHH_CG_alpha_2":3, "tHH_CG_dB1":3, "tHH_CG_dB2":4, 
    "cSH_CG_tips_distance":5, "cSH_CG_C1'-B1-B1pair":1, "cSH_CG_B1-B1pair-C1'pair":2, "cSH_CG_C4'-C1'-B1-B1pair":2, "cSH_CG_B1-B1pair-C1'pair-C4'pair":2, "cSH_CG_alpha_1":1, "cSH_CG_alpha_2":2, "cSH_CG_dB1":6, "cSH_CG_dB2":4, 
    "tSH_CG_tips_distance":5, "tSH_CG_C1'-B1-B1pair":1, "tSH_CG_B1-B1pair-C1'pair":2, "tSH_CG_C4'-C1'-B1-B1pair":2, "tSH_CG_B1-B1pair-C1'pair-C4'pair":1, "tSH_CG_alpha_1":1, "tSH_CG_alpha_2":3, "tSH_CG_dB1":2, "tSH_CG_dB2":3, 
    "cHS_CG_tips_distance":4, "cHS_CG_C1'-B1-B1pair":2, "cHS_CG_B1-B1pair-C1'pair":2, "cHS_CG_C4'-C1'-B1-B1pair":3, "cHS_CG_B1-B1pair-C1'pair-C4'pair":2, "cHS_CG_alpha_1":2, "cHS_CG_alpha_2":3, "cHS_CG_dB1":5, "cHS_CG_dB2":2, 
    "tHS_CG_tips_distance":4, "tHS_CG_C1'-B1-B1pair":1, "tHS_CG_B1-B1pair-C1'pair":2, "tHS_CG_C4'-C1'-B1-B1pair":3, "tHS_CG_B1-B1pair-C1'pair-C4'pair":1, "tHS_CG_alpha_1":1, "tHS_CG_alpha_2":1, "tHS_CG_dB1":3, "tHS_CG_dB2":2, 
    "cSS_CG_tips_distance":1, "cSS_CG_C1'-B1-B1pair":2, "cSS_CG_B1-B1pair-C1'pair":1, "cSS_CG_C4'-C1'-B1-B1pair":2, "cSS_CG_B1-B1pair-C1'pair-C4'pair":1, "cSS_CG_alpha_1":1, "cSS_CG_alpha_2":2, "cSS_CG_dB1":3, "cSS_CG_dB2":3, 
    "tSS_CG_tips_distance":5, "tSS_CG_C1'-B1-B1pair":2, "tSS_CG_B1-B1pair-C1'pair":2, "tSS_CG_C4'-C1'-B1-B1pair":1, "tSS_CG_B1-B1pair-C1'pair-C4'pair":2, "tSS_CG_alpha_1":1, "tSS_CG_alpha_2":2, "tSS_CG_dB1":1, "tSS_CG_dB2":2, 
    "cWW_CU_tips_distance":4, "cWW_CU_C1'-B1-B1pair":1, "cWW_CU_B1-B1pair-C1'pair":1, "cWW_CU_C4'-C1'-B1-B1pair":2, "cWW_CU_B1-B1pair-C1'pair-C4'pair":2, "cWW_CU_alpha_1":1, "cWW_CU_alpha_2":1, "cWW_CU_dB1":1, "cWW_CU_dB2":1, 
    "tWW_CU_tips_distance":1, "tWW_CU_C1'-B1-B1pair":2, "tWW_CU_B1-B1pair-C1'pair":2, "tWW_CU_C4'-C1'-B1-B1pair":2, "tWW_CU_B1-B1pair-C1'pair-C4'pair":2, "tWW_CU_alpha_1":1, "tWW_CU_alpha_2":2, "tWW_CU_dB1":2, "tWW_CU_dB2":1, 
    "cWH_CU_tips_distance":5, "cWH_CU_C1'-B1-B1pair":2, "cWH_CU_B1-B1pair-C1'pair":2, "cWH_CU_C4'-C1'-B1-B1pair":2, "cWH_CU_B1-B1pair-C1'pair-C4'pair":2, "cWH_CU_alpha_1":3, "cWH_CU_alpha_2":2, "cWH_CU_dB1":3, "cWH_CU_dB2":1, 
    "tWH_CU_tips_distance":1, "tWH_CU_C1'-B1-B1pair":2, "tWH_CU_B1-B1pair-C1'pair":2, "tWH_CU_C4'-C1'-B1-B1pair":3, "tWH_CU_B1-B1pair-C1'pair-C4'pair":2, "tWH_CU_alpha_1":3, "tWH_CU_alpha_2":3, "tWH_CU_dB1":5, "tWH_CU_dB2":2, 
    "cHW_CU_tips_distance":3, "cHW_CU_C1'-B1-B1pair":2, "cHW_CU_B1-B1pair-C1'pair":2, "cHW_CU_C4'-C1'-B1-B1pair":1, "cHW_CU_B1-B1pair-C1'pair-C4'pair":3, "cHW_CU_alpha_1":2, "cHW_CU_alpha_2":2, "cHW_CU_dB1":1, "cHW_CU_dB2":3, 
    "tHW_CU_tips_distance":8, "tHW_CU_C1'-B1-B1pair":1, "tHW_CU_B1-B1pair-C1'pair":1, "tHW_CU_C4'-C1'-B1-B1pair":3, "tHW_CU_B1-B1pair-C1'pair-C4'pair":2, "tHW_CU_alpha_1":1, "tHW_CU_alpha_2":2, "tHW_CU_dB1":3, "tHW_CU_dB2":3, 
    "cWS_CU_tips_distance":4, "cWS_CU_C1'-B1-B1pair":1, "cWS_CU_B1-B1pair-C1'pair":2, "cWS_CU_C4'-C1'-B1-B1pair":2, "cWS_CU_B1-B1pair-C1'pair-C4'pair":2, "cWS_CU_alpha_1":3, "cWS_CU_alpha_2":2, "cWS_CU_dB1":4, "cWS_CU_dB2":2, 
    "tWS_CU_tips_distance":5, "tWS_CU_C1'-B1-B1pair":3, "tWS_CU_B1-B1pair-C1'pair":1, "tWS_CU_C4'-C1'-B1-B1pair":2, "tWS_CU_B1-B1pair-C1'pair-C4'pair":2, "tWS_CU_alpha_1":2, "tWS_CU_alpha_2":1, "tWS_CU_dB1":3, "tWS_CU_dB2":5, 
    "cSW_CU_tips_distance":3, "cSW_CU_C1'-B1-B1pair":2, "cSW_CU_B1-B1pair-C1'pair":2, "cSW_CU_C4'-C1'-B1-B1pair":2, "cSW_CU_B1-B1pair-C1'pair-C4'pair":3, "cSW_CU_alpha_1":3, "cSW_CU_alpha_2":3, "cSW_CU_dB1":2, "cSW_CU_dB2":4, 
    "tSW_CU_tips_distance":7, "tSW_CU_C1'-B1-B1pair":2, "tSW_CU_B1-B1pair-C1'pair":2, "tSW_CU_C4'-C1'-B1-B1pair":2, "tSW_CU_B1-B1pair-C1'pair-C4'pair":2, "tSW_CU_alpha_1":2, "tSW_CU_alpha_2":2, "tSW_CU_dB1":2, "tSW_CU_dB2":2, 
    "cHH_CU_tips_distance":6, "cHH_CU_C1'-B1-B1pair":2, "cHH_CU_B1-B1pair-C1'pair":1, "cHH_CU_C4'-C1'-B1-B1pair":2, "cHH_CU_B1-B1pair-C1'pair-C4'pair":3, "cHH_CU_alpha_1":1, "cHH_CU_alpha_2":1, "cHH_CU_dB1":2, "cHH_CU_dB2":4, 
    "tHH_CU_tips_distance":5, "tHH_CU_C1'-B1-B1pair":3, "tHH_CU_B1-B1pair-C1'pair":2, "tHH_CU_C4'-C1'-B1-B1pair":2, "tHH_CU_B1-B1pair-C1'pair-C4'pair":1, "tHH_CU_alpha_1":2, "tHH_CU_alpha_2":2, "tHH_CU_dB1":2, "tHH_CU_dB2":2, 
    "cSH_CU_tips_distance":5, "cSH_CU_C1'-B1-B1pair":2, "cSH_CU_B1-B1pair-C1'pair":2, "cSH_CU_C4'-C1'-B1-B1pair":2, "cSH_CU_B1-B1pair-C1'pair-C4'pair":1, "cSH_CU_alpha_1":1, "cSH_CU_alpha_2":1, "cSH_CU_dB1":4, "cSH_CU_dB2":2, 
    "tSH_CU_tips_distance":5, "tSH_CU_C1'-B1-B1pair":2, "tSH_CU_B1-B1pair-C1'pair":3, "tSH_CU_C4'-C1'-B1-B1pair":2, "tSH_CU_B1-B1pair-C1'pair-C4'pair":2, "tSH_CU_alpha_1":3, "tSH_CU_alpha_2":3, "tSH_CU_dB1":4, "tSH_CU_dB2":2, 
    "cHS_CU_tips_distance":2, "cHS_CU_C1'-B1-B1pair":1, "cHS_CU_B1-B1pair-C1'pair":2, "cHS_CU_C4'-C1'-B1-B1pair":2, "cHS_CU_B1-B1pair-C1'pair-C4'pair":2, "cHS_CU_alpha_1":1, "cHS_CU_alpha_2":2, "cHS_CU_dB1":2, "cHS_CU_dB2":4, 
    "tHS_CU_tips_distance":8, "tHS_CU_C1'-B1-B1pair":2, "tHS_CU_B1-B1pair-C1'pair":1, "tHS_CU_C4'-C1'-B1-B1pair":2, "tHS_CU_B1-B1pair-C1'pair-C4'pair":2, "tHS_CU_alpha_1":2, "tHS_CU_alpha_2":2, "tHS_CU_dB1":3, "tHS_CU_dB2":4, 
    "cSS_CU_tips_distance":5, "cSS_CU_C1'-B1-B1pair":2, "cSS_CU_B1-B1pair-C1'pair":2, "cSS_CU_C4'-C1'-B1-B1pair":1, "cSS_CU_B1-B1pair-C1'pair-C4'pair":1, "cSS_CU_alpha_1":2, "cSS_CU_alpha_2":3, "cSS_CU_dB1":6, "cSS_CU_dB2":1, 
    "tSS_CU_tips_distance":5, "tSS_CU_C1'-B1-B1pair":2, "tSS_CU_B1-B1pair-C1'pair":3, "tSS_CU_C4'-C1'-B1-B1pair":2, "tSS_CU_B1-B1pair-C1'pair-C4'pair":2, "tSS_CU_alpha_1":3, "tSS_CU_alpha_2":3, "tSS_CU_dB1":7, "tSS_CU_dB2":2, 
    "cWW_GA_tips_distance":5, "cWW_GA_C1'-B1-B1pair":1, "cWW_GA_B1-B1pair-C1'pair":1, "cWW_GA_C4'-C1'-B1-B1pair":2, "cWW_GA_B1-B1pair-C1'pair-C4'pair":2, "cWW_GA_alpha_1":1, "cWW_GA_alpha_2":1, "cWW_GA_dB1":2, "cWW_GA_dB2":1, 
    "tWW_GA_tips_distance":6, "tWW_GA_C1'-B1-B1pair":1, "tWW_GA_B1-B1pair-C1'pair":1, "tWW_GA_C4'-C1'-B1-B1pair":1, "tWW_GA_B1-B1pair-C1'pair-C4'pair":2, "tWW_GA_alpha_1":1, "tWW_GA_alpha_2":2, "tWW_GA_dB1":1, "tWW_GA_dB2":2, 
    "cWH_GA_tips_distance":2, "cWH_GA_C1'-B1-B1pair":1, "cWH_GA_B1-B1pair-C1'pair":1, "cWH_GA_C4'-C1'-B1-B1pair":3, "cWH_GA_B1-B1pair-C1'pair-C4'pair":2, "cWH_GA_alpha_1":2, "cWH_GA_alpha_2":1, "cWH_GA_dB1":2, "cWH_GA_dB2":2, 
    "tWH_GA_tips_distance":7, "tWH_GA_C1'-B1-B1pair":1, "tWH_GA_B1-B1pair-C1'pair":2, "tWH_GA_C4'-C1'-B1-B1pair":1, "tWH_GA_B1-B1pair-C1'pair-C4'pair":2, "tWH_GA_alpha_1":2, "tWH_GA_alpha_2":2, "tWH_GA_dB1":1, "tWH_GA_dB2":6, 
    "cHW_GA_tips_distance":4, "cHW_GA_C1'-B1-B1pair":2, "cHW_GA_B1-B1pair-C1'pair":2, "cHW_GA_C4'-C1'-B1-B1pair":2, "cHW_GA_B1-B1pair-C1'pair-C4'pair":3, "cHW_GA_alpha_1":1, "cHW_GA_alpha_2":2, "cHW_GA_dB1":1, "cHW_GA_dB2":4, 
    "tHW_GA_tips_distance":3, "tHW_GA_C1'-B1-B1pair":2, "tHW_GA_B1-B1pair-C1'pair":1, "tHW_GA_C4'-C1'-B1-B1pair":2, "tHW_GA_B1-B1pair-C1'pair-C4'pair":2, "tHW_GA_alpha_1":1, "tHW_GA_alpha_2":2, "tHW_GA_dB1":3, "tHW_GA_dB2":1, 
    "cWS_GA_tips_distance":6, "cWS_GA_C1'-B1-B1pair":3, "cWS_GA_B1-B1pair-C1'pair":2, "cWS_GA_C4'-C1'-B1-B1pair":2, "cWS_GA_B1-B1pair-C1'pair-C4'pair":1, "cWS_GA_alpha_1":2, "cWS_GA_alpha_2":3, "cWS_GA_dB1":3, "cWS_GA_dB2":4, 
    "tWS_GA_tips_distance":5, "tWS_GA_C1'-B1-B1pair":3, "tWS_GA_B1-B1pair-C1'pair":2, "tWS_GA_C4'-C1'-B1-B1pair":1, "tWS_GA_B1-B1pair-C1'pair-C4'pair":1, "tWS_GA_alpha_1":2, "tWS_GA_alpha_2":2, "tWS_GA_dB1":2, "tWS_GA_dB2":5, 
    "cSW_GA_tips_distance":4, "cSW_GA_C1'-B1-B1pair":1, "cSW_GA_B1-B1pair-C1'pair":1, "cSW_GA_C4'-C1'-B1-B1pair":1, "cSW_GA_B1-B1pair-C1'pair-C4'pair":1, "cSW_GA_alpha_1":1, "cSW_GA_alpha_2":2, "cSW_GA_dB1":1, "cSW_GA_dB2":2, 
    "tSW_GA_tips_distance":2, "tSW_GA_C1'-B1-B1pair":1, "tSW_GA_B1-B1pair-C1'pair":2, "tSW_GA_C4'-C1'-B1-B1pair":1, "tSW_GA_B1-B1pair-C1'pair-C4'pair":2, "tSW_GA_alpha_1":1, "tSW_GA_alpha_2":3, "tSW_GA_dB1":2, "tSW_GA_dB2":2, 
    "cHH_GA_tips_distance":3, "cHH_GA_C1'-B1-B1pair":2, "cHH_GA_B1-B1pair-C1'pair":2, "cHH_GA_C4'-C1'-B1-B1pair":2, "cHH_GA_B1-B1pair-C1'pair-C4'pair":2, "cHH_GA_alpha_1":2, "cHH_GA_alpha_2":3, "cHH_GA_dB1":2, "cHH_GA_dB2":3, 
    "tHH_GA_tips_distance":3, "tHH_GA_C1'-B1-B1pair":3, "tHH_GA_B1-B1pair-C1'pair":2, "tHH_GA_C4'-C1'-B1-B1pair":2, "tHH_GA_B1-B1pair-C1'pair-C4'pair":2, "tHH_GA_alpha_1":1, "tHH_GA_alpha_2":2, "tHH_GA_dB1":3, "tHH_GA_dB2":2, 
    "cSH_GA_tips_distance":1, "cSH_GA_C1'-B1-B1pair":2, "cSH_GA_B1-B1pair-C1'pair":2, "cSH_GA_C4'-C1'-B1-B1pair":2, "cSH_GA_B1-B1pair-C1'pair-C4'pair":2, "cSH_GA_alpha_1":1, "cSH_GA_alpha_2":2, "cSH_GA_dB1":2, "cSH_GA_dB2":1, 
    "tSH_GA_tips_distance":3, "tSH_GA_C1'-B1-B1pair":1, "tSH_GA_B1-B1pair-C1'pair":1, "tSH_GA_C4'-C1'-B1-B1pair":2, "tSH_GA_B1-B1pair-C1'pair-C4'pair":2, "tSH_GA_alpha_1":2, "tSH_GA_alpha_2":2, "tSH_GA_dB1":2, "tSH_GA_dB2":7, 
    "cHS_GA_tips_distance":5, "cHS_GA_C1'-B1-B1pair":3, "cHS_GA_B1-B1pair-C1'pair":3, "cHS_GA_C4'-C1'-B1-B1pair":3, "cHS_GA_B1-B1pair-C1'pair-C4'pair":2, "cHS_GA_alpha_1":2, "cHS_GA_alpha_2":2, "cHS_GA_dB1":3, "cHS_GA_dB2":4, 
    "tHS_GA_tips_distance":5, "tHS_GA_C1'-B1-B1pair":3, "tHS_GA_B1-B1pair-C1'pair":1, "tHS_GA_C4'-C1'-B1-B1pair":3, "tHS_GA_B1-B1pair-C1'pair-C4'pair":2, "tHS_GA_alpha_1":2, "tHS_GA_alpha_2":1, "tHS_GA_dB1":1, "tHS_GA_dB2":2, 
    "cSS_GA_tips_distance":4, "cSS_GA_C1'-B1-B1pair":3, "cSS_GA_B1-B1pair-C1'pair":2, "cSS_GA_C4'-C1'-B1-B1pair":1, "cSS_GA_B1-B1pair-C1'pair-C4'pair":1, "cSS_GA_alpha_1":2, "cSS_GA_alpha_2":1, "cSS_GA_dB1":1, "cSS_GA_dB2":1, 
    "tSS_GA_tips_distance":4, "tSS_GA_C1'-B1-B1pair":1, "tSS_GA_B1-B1pair-C1'pair":1, "tSS_GA_C4'-C1'-B1-B1pair":1, "tSS_GA_B1-B1pair-C1'pair-C4'pair":1, "tSS_GA_alpha_1":1, "tSS_GA_alpha_2":2, "tSS_GA_dB1":5, "tSS_GA_dB2":2, 
    "cWW_GC_tips_distance":5, "cWW_GC_C1'-B1-B1pair":1, "cWW_GC_B1-B1pair-C1'pair":2, "cWW_GC_C4'-C1'-B1-B1pair":2, "cWW_GC_B1-B1pair-C1'pair-C4'pair":2, "cWW_GC_alpha_1":2, "cWW_GC_alpha_2":1, "cWW_GC_dB1":2, "cWW_GC_dB2":3, 
    "tWW_GC_tips_distance":3, "tWW_GC_C1'-B1-B1pair":1, "tWW_GC_B1-B1pair-C1'pair":2, "tWW_GC_C4'-C1'-B1-B1pair":2, "tWW_GC_B1-B1pair-C1'pair-C4'pair":2, "tWW_GC_alpha_1":1, "tWW_GC_alpha_2":2, "tWW_GC_dB1":3, "tWW_GC_dB2":4, 
    "cWH_GC_tips_distance":7, "cWH_GC_C1'-B1-B1pair":2, "cWH_GC_B1-B1pair-C1'pair":2, "cWH_GC_C4'-C1'-B1-B1pair":2, "cWH_GC_B1-B1pair-C1'pair-C4'pair":1, "cWH_GC_alpha_1":2, "cWH_GC_alpha_2":2, "cWH_GC_dB1":2, "cWH_GC_dB2":3, 
    "tWH_GC_tips_distance":5, "tWH_GC_C1'-B1-B1pair":1, "tWH_GC_B1-B1pair-C1'pair":1, "tWH_GC_C4'-C1'-B1-B1pair":2, "tWH_GC_B1-B1pair-C1'pair-C4'pair":2, "tWH_GC_alpha_1":3, "tWH_GC_alpha_2":3, "tWH_GC_dB1":2, "tWH_GC_dB2":2, 
    "cHW_GC_tips_distance":4, "cHW_GC_C1'-B1-B1pair":1, "cHW_GC_B1-B1pair-C1'pair":1, "cHW_GC_C4'-C1'-B1-B1pair":2, "cHW_GC_B1-B1pair-C1'pair-C4'pair":2, "cHW_GC_alpha_1":1, "cHW_GC_alpha_2":1, "cHW_GC_dB1":3, "cHW_GC_dB2":4, 
    "tHW_GC_tips_distance":5, "tHW_GC_C1'-B1-B1pair":2, "tHW_GC_B1-B1pair-C1'pair":2, "tHW_GC_C4'-C1'-B1-B1pair":2, "tHW_GC_B1-B1pair-C1'pair-C4'pair":2, "tHW_GC_alpha_1":2, "tHW_GC_alpha_2":2, "tHW_GC_dB1":2, "tHW_GC_dB2":4, 
    "cWS_GC_tips_distance":8, "cWS_GC_C1'-B1-B1pair":1, "cWS_GC_B1-B1pair-C1'pair":1, "cWS_GC_C4'-C1'-B1-B1pair":2, "cWS_GC_B1-B1pair-C1'pair-C4'pair":2, "cWS_GC_alpha_1":2, "cWS_GC_alpha_2":1, "cWS_GC_dB1":2, "cWS_GC_dB2":1, 
    "tWS_GC_tips_distance":2, "tWS_GC_C1'-B1-B1pair":1, "tWS_GC_B1-B1pair-C1'pair":1, "tWS_GC_C4'-C1'-B1-B1pair":3, "tWS_GC_B1-B1pair-C1'pair-C4'pair":2, "tWS_GC_alpha_1":1, "tWS_GC_alpha_2":1, "tWS_GC_dB1":4, "tWS_GC_dB2":5, 
    "cSW_GC_tips_distance":4, "cSW_GC_C1'-B1-B1pair":2, "cSW_GC_B1-B1pair-C1'pair":3, "cSW_GC_C4'-C1'-B1-B1pair":1, "cSW_GC_B1-B1pair-C1'pair-C4'pair":2, "cSW_GC_alpha_1":3, "cSW_GC_alpha_2":2, "cSW_GC_dB1":3, "cSW_GC_dB2":2, 
    "tSW_GC_tips_distance":2, "tSW_GC_C1'-B1-B1pair":1, "tSW_GC_B1-B1pair-C1'pair":3, "tSW_GC_C4'-C1'-B1-B1pair":1, "tSW_GC_B1-B1pair-C1'pair-C4'pair":2, "tSW_GC_alpha_1":2, "tSW_GC_alpha_2":2, "tSW_GC_dB1":4, "tSW_GC_dB2":2, 
    "cHH_GC_tips_distance":1, "cHH_GC_C1'-B1-B1pair":3, "cHH_GC_B1-B1pair-C1'pair":1, "cHH_GC_C4'-C1'-B1-B1pair":2, "cHH_GC_B1-B1pair-C1'pair-C4'pair":1, "cHH_GC_alpha_1":2, "cHH_GC_alpha_2":2, "cHH_GC_dB1":3, "cHH_GC_dB2":3, 
    "tHH_GC_tips_distance":8, "tHH_GC_C1'-B1-B1pair":2, "tHH_GC_B1-B1pair-C1'pair":1, "tHH_GC_C4'-C1'-B1-B1pair":2, "tHH_GC_B1-B1pair-C1'pair-C4'pair":2, "tHH_GC_alpha_1":3, "tHH_GC_alpha_2":1, "tHH_GC_dB1":6, "tHH_GC_dB2":3, 
    "cSH_GC_tips_distance":8, "cSH_GC_C1'-B1-B1pair":2, "cSH_GC_B1-B1pair-C1'pair":3, "cSH_GC_C4'-C1'-B1-B1pair":1, "cSH_GC_B1-B1pair-C1'pair-C4'pair":3, "cSH_GC_alpha_1":2, "cSH_GC_alpha_2":2, "cSH_GC_dB1":5, "cSH_GC_dB2":4, 
    "tSH_GC_tips_distance":4, "tSH_GC_C1'-B1-B1pair":1, "tSH_GC_B1-B1pair-C1'pair":2, "tSH_GC_C4'-C1'-B1-B1pair":1, "tSH_GC_B1-B1pair-C1'pair-C4'pair":4, "tSH_GC_alpha_1":1, "tSH_GC_alpha_2":2, "tSH_GC_dB1":2, "tSH_GC_dB2":3, 
    "cHS_GC_tips_distance":5, "cHS_GC_C1'-B1-B1pair":2, "cHS_GC_B1-B1pair-C1'pair":2, "cHS_GC_C4'-C1'-B1-B1pair":2, "cHS_GC_B1-B1pair-C1'pair-C4'pair":2, "cHS_GC_alpha_1":3, "cHS_GC_alpha_2":1, "cHS_GC_dB1":2, "cHS_GC_dB2":5, 
    "tHS_GC_tips_distance":5, "tHS_GC_C1'-B1-B1pair":2, "tHS_GC_B1-B1pair-C1'pair":2, "tHS_GC_C4'-C1'-B1-B1pair":2, "tHS_GC_B1-B1pair-C1'pair-C4'pair":3, "tHS_GC_alpha_1":2, "tHS_GC_alpha_2":2, "tHS_GC_dB1":2, "tHS_GC_dB2":2, 
    "cSS_GC_tips_distance":2, "cSS_GC_C1'-B1-B1pair":2, "cSS_GC_B1-B1pair-C1'pair":2, "cSS_GC_C4'-C1'-B1-B1pair":1, "cSS_GC_B1-B1pair-C1'pair-C4'pair":1, "cSS_GC_alpha_1":2, "cSS_GC_alpha_2":3, "cSS_GC_dB1":3, "cSS_GC_dB2":3, 
    "tSS_GC_tips_distance":5, "tSS_GC_C1'-B1-B1pair":2, "tSS_GC_B1-B1pair-C1'pair":2, "tSS_GC_C4'-C1'-B1-B1pair":1, "tSS_GC_B1-B1pair-C1'pair-C4'pair":2, "tSS_GC_alpha_1":2, "tSS_GC_alpha_2":3, "tSS_GC_dB1":2, "tSS_GC_dB2":1, 
    "cWW_GG_tips_distance":3, "cWW_GG_C1'-B1-B1pair":1, "cWW_GG_B1-B1pair-C1'pair":1, "cWW_GG_C4'-C1'-B1-B1pair":2, "cWW_GG_B1-B1pair-C1'pair-C4'pair":1, "cWW_GG_alpha_1":1, "cWW_GG_alpha_2":2, "cWW_GG_dB1":2, "cWW_GG_dB2":2, 
    "tWW_GG_tips_distance":4, "tWW_GG_C1'-B1-B1pair":1, "tWW_GG_B1-B1pair-C1'pair":1, "tWW_GG_C4'-C1'-B1-B1pair":2, "tWW_GG_B1-B1pair-C1'pair-C4'pair":2, "tWW_GG_alpha_1":2, "tWW_GG_alpha_2":2, "tWW_GG_dB1":1, "tWW_GG_dB2":2, 
    "cWH_GG_tips_distance":2, "cWH_GG_C1'-B1-B1pair":2, "cWH_GG_B1-B1pair-C1'pair":2, "cWH_GG_C4'-C1'-B1-B1pair":2, "cWH_GG_B1-B1pair-C1'pair-C4'pair":2, "cWH_GG_alpha_1":2, "cWH_GG_alpha_2":2, "cWH_GG_dB1":4, "cWH_GG_dB2":3, 
    "tWH_GG_tips_distance":2, "tWH_GG_C1'-B1-B1pair":1, "tWH_GG_B1-B1pair-C1'pair":2, "tWH_GG_C4'-C1'-B1-B1pair":2, "tWH_GG_B1-B1pair-C1'pair-C4'pair":2, "tWH_GG_alpha_1":2, "tWH_GG_alpha_2":2, "tWH_GG_dB1":2, "tWH_GG_dB2":3, 
    "cHW_GG_tips_distance":3, "cHW_GG_C1'-B1-B1pair":2, "cHW_GG_B1-B1pair-C1'pair":2, "cHW_GG_C4'-C1'-B1-B1pair":2, "cHW_GG_B1-B1pair-C1'pair-C4'pair":2, "cHW_GG_alpha_1":1, "cHW_GG_alpha_2":1, "cHW_GG_dB1":2, "cHW_GG_dB2":2, 
    "tHW_GG_tips_distance":4, "tHW_GG_C1'-B1-B1pair":2, "tHW_GG_B1-B1pair-C1'pair":2, "tHW_GG_C4'-C1'-B1-B1pair":1, "tHW_GG_B1-B1pair-C1'pair-C4'pair":2, "tHW_GG_alpha_1":2, "tHW_GG_alpha_2":2, "tHW_GG_dB1":1, "tHW_GG_dB2":4, 
    "cWS_GG_tips_distance":2, "cWS_GG_C1'-B1-B1pair":1, "cWS_GG_B1-B1pair-C1'pair":1, "cWS_GG_C4'-C1'-B1-B1pair":2, "cWS_GG_B1-B1pair-C1'pair-C4'pair":1, "cWS_GG_alpha_1":2, "cWS_GG_alpha_2":2, "cWS_GG_dB1":4, "cWS_GG_dB2":3, 
    "tWS_GG_tips_distance":8, "tWS_GG_C1'-B1-B1pair":3, "tWS_GG_B1-B1pair-C1'pair":2, "tWS_GG_C4'-C1'-B1-B1pair":3, "tWS_GG_B1-B1pair-C1'pair-C4'pair":2, "tWS_GG_alpha_1":1, "tWS_GG_alpha_2":1, "tWS_GG_dB1":1, "tWS_GG_dB2":3, 
    "cSW_GG_tips_distance":1, "cSW_GG_C1'-B1-B1pair":1, "cSW_GG_B1-B1pair-C1'pair":1, "cSW_GG_C4'-C1'-B1-B1pair":1, "cSW_GG_B1-B1pair-C1'pair-C4'pair":2, "cSW_GG_alpha_1":2, "cSW_GG_alpha_2":2, "cSW_GG_dB1":2, "cSW_GG_dB2":2, 
    "tSW_GG_tips_distance":5, "tSW_GG_C1'-B1-B1pair":3, "tSW_GG_B1-B1pair-C1'pair":2, "tSW_GG_C4'-C1'-B1-B1pair":3, "tSW_GG_B1-B1pair-C1'pair-C4'pair":2, "tSW_GG_alpha_1":1, "tSW_GG_alpha_2":3, "tSW_GG_dB1":2, "tSW_GG_dB2":1, 
    "cHH_GG_tips_distance":4, "cHH_GG_C1'-B1-B1pair":1, "cHH_GG_B1-B1pair-C1'pair":1, "cHH_GG_C4'-C1'-B1-B1pair":2, "cHH_GG_B1-B1pair-C1'pair-C4'pair":3, "cHH_GG_alpha_1":1, "cHH_GG_alpha_2":2, "cHH_GG_dB1":2, "cHH_GG_dB2":3, 
    "tHH_GG_tips_distance":8, "tHH_GG_C1'-B1-B1pair":2, "tHH_GG_B1-B1pair-C1'pair":2, "tHH_GG_C4'-C1'-B1-B1pair":2, "tHH_GG_B1-B1pair-C1'pair-C4'pair":3, "tHH_GG_alpha_1":2, "tHH_GG_alpha_2":2, "tHH_GG_dB1":2, "tHH_GG_dB2":3, 
    "cSH_GG_tips_distance":2, "cSH_GG_C1'-B1-B1pair":2, "cSH_GG_B1-B1pair-C1'pair":1, "cSH_GG_C4'-C1'-B1-B1pair":1, "cSH_GG_B1-B1pair-C1'pair-C4'pair":2, "cSH_GG_alpha_1":2, "cSH_GG_alpha_2":1, "cSH_GG_dB1":1, "cSH_GG_dB2":1, 
    "tSH_GG_tips_distance":2, "tSH_GG_C1'-B1-B1pair":2, "tSH_GG_B1-B1pair-C1'pair":2, "tSH_GG_C4'-C1'-B1-B1pair":2, "tSH_GG_B1-B1pair-C1'pair-C4'pair":2, "tSH_GG_alpha_1":2, "tSH_GG_alpha_2":2, "tSH_GG_dB1":1, "tSH_GG_dB2":2, 
    "cHS_GG_tips_distance":2, "cHS_GG_C1'-B1-B1pair":1, "cHS_GG_B1-B1pair-C1'pair":2, "cHS_GG_C4'-C1'-B1-B1pair":2, "cHS_GG_B1-B1pair-C1'pair-C4'pair":1, "cHS_GG_alpha_1":1, "cHS_GG_alpha_2":2, "cHS_GG_dB1":1, "cHS_GG_dB2":2, 
    "tHS_GG_tips_distance":2, "tHS_GG_C1'-B1-B1pair":2, "tHS_GG_B1-B1pair-C1'pair":2, "tHS_GG_C4'-C1'-B1-B1pair":2, "tHS_GG_B1-B1pair-C1'pair-C4'pair":1, "tHS_GG_alpha_1":2, "tHS_GG_alpha_2":3, "tHS_GG_dB1":2, "tHS_GG_dB2":1, 
    "cSS_GG_tips_distance":2, "cSS_GG_C1'-B1-B1pair":2, "cSS_GG_B1-B1pair-C1'pair":2, "cSS_GG_C4'-C1'-B1-B1pair":1, "cSS_GG_B1-B1pair-C1'pair-C4'pair":1, "cSS_GG_alpha_1":2, "cSS_GG_alpha_2":3, "cSS_GG_dB1":3, "cSS_GG_dB2":5, 
    "tSS_GG_tips_distance":2, "tSS_GG_C1'-B1-B1pair":3, "tSS_GG_B1-B1pair-C1'pair":2, "tSS_GG_C4'-C1'-B1-B1pair":2, "tSS_GG_B1-B1pair-C1'pair-C4'pair":1, "tSS_GG_alpha_1":1, "tSS_GG_alpha_2":3, "tSS_GG_dB1":3, "tSS_GG_dB2":2, 
    "cWW_GU_tips_distance":2, "cWW_GU_C1'-B1-B1pair":2, "cWW_GU_B1-B1pair-C1'pair":2, "cWW_GU_C4'-C1'-B1-B1pair":2, "cWW_GU_B1-B1pair-C1'pair-C4'pair":1, "cWW_GU_alpha_1":3, "cWW_GU_alpha_2":2, "cWW_GU_dB1":4, "cWW_GU_dB2":3, 
    "tWW_GU_tips_distance":2, "tWW_GU_C1'-B1-B1pair":3, "tWW_GU_B1-B1pair-C1'pair":2, "tWW_GU_C4'-C1'-B1-B1pair":2, "tWW_GU_B1-B1pair-C1'pair-C4'pair":3, "tWW_GU_alpha_1":2, "tWW_GU_alpha_2":2, "tWW_GU_dB1":3, "tWW_GU_dB2":3, 
    "cWH_GU_tips_distance":2, "cWH_GU_C1'-B1-B1pair":1, "cWH_GU_B1-B1pair-C1'pair":2, "cWH_GU_C4'-C1'-B1-B1pair":1, "cWH_GU_B1-B1pair-C1'pair-C4'pair":2, "cWH_GU_alpha_1":2, "cWH_GU_alpha_2":4, "cWH_GU_dB1":3, "cWH_GU_dB2":1, 
    "tWH_GU_tips_distance":8, "tWH_GU_C1'-B1-B1pair":1, "tWH_GU_B1-B1pair-C1'pair":2, "tWH_GU_C4'-C1'-B1-B1pair":2, "tWH_GU_B1-B1pair-C1'pair-C4'pair":2, "tWH_GU_alpha_1":2, "tWH_GU_alpha_2":2, "tWH_GU_dB1":3, "tWH_GU_dB2":1, 
    "cHW_GU_tips_distance":4, "cHW_GU_C1'-B1-B1pair":2, "cHW_GU_B1-B1pair-C1'pair":1, "cHW_GU_C4'-C1'-B1-B1pair":2, "cHW_GU_B1-B1pair-C1'pair-C4'pair":2, "cHW_GU_alpha_1":2, "cHW_GU_alpha_2":2, "cHW_GU_dB1":3, "cHW_GU_dB2":3, 
    "tHW_GU_tips_distance":1, "tHW_GU_C1'-B1-B1pair":3, "tHW_GU_B1-B1pair-C1'pair":1, "tHW_GU_C4'-C1'-B1-B1pair":2, "tHW_GU_B1-B1pair-C1'pair-C4'pair":3, "tHW_GU_alpha_1":3, "tHW_GU_alpha_2":1, "tHW_GU_dB1":2, "tHW_GU_dB2":5, 
    "cWS_GU_tips_distance":2, "cWS_GU_C1'-B1-B1pair":1, "cWS_GU_B1-B1pair-C1'pair":1, "cWS_GU_C4'-C1'-B1-B1pair":1, "cWS_GU_B1-B1pair-C1'pair-C4'pair":2, "cWS_GU_alpha_1":3, "cWS_GU_alpha_2":3, "cWS_GU_dB1":2, "cWS_GU_dB2":3, 
    "tWS_GU_tips_distance":4, "tWS_GU_C1'-B1-B1pair":3, "tWS_GU_B1-B1pair-C1'pair":1, "tWS_GU_C4'-C1'-B1-B1pair":3, "tWS_GU_B1-B1pair-C1'pair-C4'pair":2, "tWS_GU_alpha_1":1, "tWS_GU_alpha_2":2, "tWS_GU_dB1":3, "tWS_GU_dB2":3, 
    "cSW_GU_tips_distance":2, "cSW_GU_C1'-B1-B1pair":2, "cSW_GU_B1-B1pair-C1'pair":2, "cSW_GU_C4'-C1'-B1-B1pair":2, "cSW_GU_B1-B1pair-C1'pair-C4'pair":2, "cSW_GU_alpha_1":1, "cSW_GU_alpha_2":2, "cSW_GU_dB1":3, "cSW_GU_dB2":2, 
    "tSW_GU_tips_distance":3, "tSW_GU_C1'-B1-B1pair":1, "tSW_GU_B1-B1pair-C1'pair":2, "tSW_GU_C4'-C1'-B1-B1pair":2, "tSW_GU_B1-B1pair-C1'pair-C4'pair":2, "tSW_GU_alpha_1":1, "tSW_GU_alpha_2":2, "tSW_GU_dB1":5, "tSW_GU_dB2":1, 
    "cHH_GU_tips_distance":5, "cHH_GU_C1'-B1-B1pair":2, "cHH_GU_B1-B1pair-C1'pair":3, "cHH_GU_C4'-C1'-B1-B1pair":2, "cHH_GU_B1-B1pair-C1'pair-C4'pair":2, "cHH_GU_alpha_1":2, "cHH_GU_alpha_2":2, "cHH_GU_dB1":5, "cHH_GU_dB2":3, 
    "tHH_GU_tips_distance":5, "tHH_GU_C1'-B1-B1pair":2, "tHH_GU_B1-B1pair-C1'pair":1, "tHH_GU_C4'-C1'-B1-B1pair":1, "tHH_GU_B1-B1pair-C1'pair-C4'pair":2, "tHH_GU_alpha_1":2, "tHH_GU_alpha_2":1, "tHH_GU_dB1":8, "tHH_GU_dB2":2, 
    "cSH_GU_tips_distance":3, "cSH_GU_C1'-B1-B1pair":1, "cSH_GU_B1-B1pair-C1'pair":2, "cSH_GU_C4'-C1'-B1-B1pair":3, "cSH_GU_B1-B1pair-C1'pair-C4'pair":2, "cSH_GU_alpha_1":2, "cSH_GU_alpha_2":1, "cSH_GU_dB1":2, "cSH_GU_dB2":2, 
    "tSH_GU_tips_distance":2, "tSH_GU_C1'-B1-B1pair":2, "tSH_GU_B1-B1pair-C1'pair":2, "tSH_GU_C4'-C1'-B1-B1pair":1, "tSH_GU_B1-B1pair-C1'pair-C4'pair":1, "tSH_GU_alpha_1":2, "tSH_GU_alpha_2":3, "tSH_GU_dB1":3, "tSH_GU_dB2":3, 
    "cHS_GU_tips_distance":8, "cHS_GU_C1'-B1-B1pair":1, "cHS_GU_B1-B1pair-C1'pair":1, "cHS_GU_C4'-C1'-B1-B1pair":2, "cHS_GU_B1-B1pair-C1'pair-C4'pair":2, "cHS_GU_alpha_1":1, "cHS_GU_alpha_2":1, "cHS_GU_dB1":4, "cHS_GU_dB2":3, 
    "tHS_GU_tips_distance":5, "tHS_GU_C1'-B1-B1pair":4, "tHS_GU_B1-B1pair-C1'pair":2, "tHS_GU_C4'-C1'-B1-B1pair":2, "tHS_GU_B1-B1pair-C1'pair-C4'pair":1, "tHS_GU_alpha_1":2, "tHS_GU_alpha_2":1, "tHS_GU_dB1":1, "tHS_GU_dB2":3, 
    "cSS_GU_tips_distance":2, "cSS_GU_C1'-B1-B1pair":3, "cSS_GU_B1-B1pair-C1'pair":2, "cSS_GU_C4'-C1'-B1-B1pair":2, "cSS_GU_B1-B1pair-C1'pair-C4'pair":2, "cSS_GU_alpha_1":2, "cSS_GU_alpha_2":1, "cSS_GU_dB1":3, "cSS_GU_dB2":4, 
    "tSS_GU_tips_distance":5, "tSS_GU_C1'-B1-B1pair":2, "tSS_GU_B1-B1pair-C1'pair":2, "tSS_GU_C4'-C1'-B1-B1pair":1, "tSS_GU_B1-B1pair-C1'pair-C4'pair":3, "tSS_GU_alpha_1":2, "tSS_GU_alpha_2":2, "tSS_GU_dB1":2, "tSS_GU_dB2":6, 
    "cWW_UA_tips_distance":4, "cWW_UA_C1'-B1-B1pair":2, "cWW_UA_B1-B1pair-C1'pair":2, "cWW_UA_C4'-C1'-B1-B1pair":1, "cWW_UA_B1-B1pair-C1'pair-C4'pair":2, "cWW_UA_alpha_1":2, "cWW_UA_alpha_2":2, "cWW_UA_dB1":2, "cWW_UA_dB2":7, 
    "tWW_UA_tips_distance":2, "tWW_UA_C1'-B1-B1pair":1, "tWW_UA_B1-B1pair-C1'pair":2, "tWW_UA_C4'-C1'-B1-B1pair":2, "tWW_UA_B1-B1pair-C1'pair-C4'pair":1, "tWW_UA_alpha_1":2, "tWW_UA_alpha_2":1, "tWW_UA_dB1":6, "tWW_UA_dB2":1, 
    "cWH_UA_tips_distance":3, "cWH_UA_C1'-B1-B1pair":3, "cWH_UA_B1-B1pair-C1'pair":3, "cWH_UA_C4'-C1'-B1-B1pair":3, "cWH_UA_B1-B1pair-C1'pair-C4'pair":2, "cWH_UA_alpha_1":2, "cWH_UA_alpha_2":3, "cWH_UA_dB1":4, "cWH_UA_dB2":3, 
    "tWH_UA_tips_distance":3, "tWH_UA_C1'-B1-B1pair":2, "tWH_UA_B1-B1pair-C1'pair":1, "tWH_UA_C4'-C1'-B1-B1pair":2, "tWH_UA_B1-B1pair-C1'pair-C4'pair":2, "tWH_UA_alpha_1":1, "tWH_UA_alpha_2":2, "tWH_UA_dB1":3, "tWH_UA_dB2":2, 
    "cHW_UA_tips_distance":5, "cHW_UA_C1'-B1-B1pair":1, "cHW_UA_B1-B1pair-C1'pair":1, "cHW_UA_C4'-C1'-B1-B1pair":3, "cHW_UA_B1-B1pair-C1'pair-C4'pair":1, "cHW_UA_alpha_1":1, "cHW_UA_alpha_2":1, "cHW_UA_dB1":3, "cHW_UA_dB2":1, 
    "tHW_UA_tips_distance":7, "tHW_UA_C1'-B1-B1pair":3, "tHW_UA_B1-B1pair-C1'pair":2, "tHW_UA_C4'-C1'-B1-B1pair":1, "tHW_UA_B1-B1pair-C1'pair-C4'pair":2, "tHW_UA_alpha_1":3, "tHW_UA_alpha_2":3, "tHW_UA_dB1":2, "tHW_UA_dB2":1, 
    "cWS_UA_tips_distance":1, "cWS_UA_C1'-B1-B1pair":2, "cWS_UA_B1-B1pair-C1'pair":3, "cWS_UA_C4'-C1'-B1-B1pair":2, "cWS_UA_B1-B1pair-C1'pair-C4'pair":1, "cWS_UA_alpha_1":2, "cWS_UA_alpha_2":2, "cWS_UA_dB1":3, "cWS_UA_dB2":4, 
    "tWS_UA_tips_distance":5, "tWS_UA_C1'-B1-B1pair":1, "tWS_UA_B1-B1pair-C1'pair":2, "tWS_UA_C4'-C1'-B1-B1pair":2, "tWS_UA_B1-B1pair-C1'pair-C4'pair":1, "tWS_UA_alpha_1":1, "tWS_UA_alpha_2":3, "tWS_UA_dB1":1, "tWS_UA_dB2":1, 
    "cSW_UA_tips_distance":2, "cSW_UA_C1'-B1-B1pair":1, "cSW_UA_B1-B1pair-C1'pair":1, "cSW_UA_C4'-C1'-B1-B1pair":2, "cSW_UA_B1-B1pair-C1'pair-C4'pair":2, "cSW_UA_alpha_1":2, "cSW_UA_alpha_2":3, "cSW_UA_dB1":3, "cSW_UA_dB2":3, 
    "tSW_UA_tips_distance":2, "tSW_UA_C1'-B1-B1pair":1, "tSW_UA_B1-B1pair-C1'pair":2, "tSW_UA_C4'-C1'-B1-B1pair":1, "tSW_UA_B1-B1pair-C1'pair-C4'pair":1, "tSW_UA_alpha_1":2, "tSW_UA_alpha_2":2, "tSW_UA_dB1":3, "tSW_UA_dB2":2, 
    "cHH_UA_tips_distance":4, "cHH_UA_C1'-B1-B1pair":1, "cHH_UA_B1-B1pair-C1'pair":1, "cHH_UA_C4'-C1'-B1-B1pair":1, "cHH_UA_B1-B1pair-C1'pair-C4'pair":2, "cHH_UA_alpha_1":2, "cHH_UA_alpha_2":2, "cHH_UA_dB1":5, "cHH_UA_dB2":2, 
    "tHH_UA_tips_distance":4, "tHH_UA_C1'-B1-B1pair":2, "tHH_UA_B1-B1pair-C1'pair":2, "tHH_UA_C4'-C1'-B1-B1pair":2, "tHH_UA_B1-B1pair-C1'pair-C4'pair":2, "tHH_UA_alpha_1":2, "tHH_UA_alpha_2":3, "tHH_UA_dB1":3, "tHH_UA_dB2":1, 
    "cSH_UA_tips_distance":4, "cSH_UA_C1'-B1-B1pair":1, "cSH_UA_B1-B1pair-C1'pair":1, "cSH_UA_C4'-C1'-B1-B1pair":2, "cSH_UA_B1-B1pair-C1'pair-C4'pair":2, "cSH_UA_alpha_1":2, "cSH_UA_alpha_2":2, "cSH_UA_dB1":3, "cSH_UA_dB2":2, 
    "tSH_UA_tips_distance":2, "tSH_UA_C1'-B1-B1pair":2, "tSH_UA_B1-B1pair-C1'pair":2, "tSH_UA_C4'-C1'-B1-B1pair":3, "tSH_UA_B1-B1pair-C1'pair-C4'pair":2, "tSH_UA_alpha_1":3, "tSH_UA_alpha_2":2, "tSH_UA_dB1":4, "tSH_UA_dB2":1, 
    "cHS_UA_tips_distance":5, "cHS_UA_C1'-B1-B1pair":2, "cHS_UA_B1-B1pair-C1'pair":2, "cHS_UA_C4'-C1'-B1-B1pair":2, "cHS_UA_B1-B1pair-C1'pair-C4'pair":2, "cHS_UA_alpha_1":2, "cHS_UA_alpha_2":2, "cHS_UA_dB1":1, "cHS_UA_dB2":3, 
    "tHS_UA_tips_distance":5, "tHS_UA_C1'-B1-B1pair":2, "tHS_UA_B1-B1pair-C1'pair":2, "tHS_UA_C4'-C1'-B1-B1pair":3, "tHS_UA_B1-B1pair-C1'pair-C4'pair":1, "tHS_UA_alpha_1":3, "tHS_UA_alpha_2":3, "tHS_UA_dB1":2, "tHS_UA_dB2":7, 
    "cSS_UA_tips_distance":2, "cSS_UA_C1'-B1-B1pair":2, "cSS_UA_B1-B1pair-C1'pair":2, "cSS_UA_C4'-C1'-B1-B1pair":2, "cSS_UA_B1-B1pair-C1'pair-C4'pair":1, "cSS_UA_alpha_1":1, "cSS_UA_alpha_2":1, "cSS_UA_dB1":2, "cSS_UA_dB2":1, 
    "tSS_UA_tips_distance":5, "tSS_UA_C1'-B1-B1pair":1, "tSS_UA_B1-B1pair-C1'pair":3, "tSS_UA_C4'-C1'-B1-B1pair":2, "tSS_UA_B1-B1pair-C1'pair-C4'pair":3, "tSS_UA_alpha_1":2, "tSS_UA_alpha_2":2, "tSS_UA_dB1":4, "tSS_UA_dB2":4, 
    "cWW_UC_tips_distance":3, "cWW_UC_C1'-B1-B1pair":1, "cWW_UC_B1-B1pair-C1'pair":2, "cWW_UC_C4'-C1'-B1-B1pair":2, "cWW_UC_B1-B1pair-C1'pair-C4'pair":2, "cWW_UC_alpha_1":2, "cWW_UC_alpha_2":1, "cWW_UC_dB1":1, "cWW_UC_dB2":2, 
    "tWW_UC_tips_distance":4, "tWW_UC_C1'-B1-B1pair":2, "tWW_UC_B1-B1pair-C1'pair":2, "tWW_UC_C4'-C1'-B1-B1pair":2, "tWW_UC_B1-B1pair-C1'pair-C4'pair":2, "tWW_UC_alpha_1":3, "tWW_UC_alpha_2":1, "tWW_UC_dB1":1, "tWW_UC_dB2":4, 
    "cWH_UC_tips_distance":2, "cWH_UC_C1'-B1-B1pair":2, "cWH_UC_B1-B1pair-C1'pair":2, "cWH_UC_C4'-C1'-B1-B1pair":2, "cWH_UC_B1-B1pair-C1'pair-C4'pair":4, "cWH_UC_alpha_1":2, "cWH_UC_alpha_2":3, "cWH_UC_dB1":3, "cWH_UC_dB2":3, 
    "tWH_UC_tips_distance":4, "tWH_UC_C1'-B1-B1pair":3, "tWH_UC_B1-B1pair-C1'pair":2, "tWH_UC_C4'-C1'-B1-B1pair":3, "tWH_UC_B1-B1pair-C1'pair-C4'pair":1, "tWH_UC_alpha_1":4, "tWH_UC_alpha_2":1, "tWH_UC_dB1":4, "tWH_UC_dB2":2, 
    "cHW_UC_tips_distance":5, "cHW_UC_C1'-B1-B1pair":2, "cHW_UC_B1-B1pair-C1'pair":2, "cHW_UC_C4'-C1'-B1-B1pair":1, "cHW_UC_B1-B1pair-C1'pair-C4'pair":2, "cHW_UC_alpha_1":2, "cHW_UC_alpha_2":2, "cHW_UC_dB1":2, "cHW_UC_dB2":6, 
    "tHW_UC_tips_distance":2, "tHW_UC_C1'-B1-B1pair":2, "tHW_UC_B1-B1pair-C1'pair":2, "tHW_UC_C4'-C1'-B1-B1pair":3, "tHW_UC_B1-B1pair-C1'pair-C4'pair":2, "tHW_UC_alpha_1":2, "tHW_UC_alpha_2":4, "tHW_UC_dB1":4, "tHW_UC_dB2":4, 
    "cWS_UC_tips_distance":4, "cWS_UC_C1'-B1-B1pair":2, "cWS_UC_B1-B1pair-C1'pair":2, "cWS_UC_C4'-C1'-B1-B1pair":2, "cWS_UC_B1-B1pair-C1'pair-C4'pair":2, "cWS_UC_alpha_1":3, "cWS_UC_alpha_2":2, "cWS_UC_dB1":3, "cWS_UC_dB2":2, 
    "tWS_UC_tips_distance":4, "tWS_UC_C1'-B1-B1pair":2, "tWS_UC_B1-B1pair-C1'pair":1, "tWS_UC_C4'-C1'-B1-B1pair":2, "tWS_UC_B1-B1pair-C1'pair-C4'pair":2, "tWS_UC_alpha_1":2, "tWS_UC_alpha_2":1, "tWS_UC_dB1":3, "tWS_UC_dB2":2, 
    "cSW_UC_tips_distance":4, "cSW_UC_C1'-B1-B1pair":1, "cSW_UC_B1-B1pair-C1'pair":2, "cSW_UC_C4'-C1'-B1-B1pair":2, "cSW_UC_B1-B1pair-C1'pair-C4'pair":2, "cSW_UC_alpha_1":2, "cSW_UC_alpha_2":3, "cSW_UC_dB1":3, "cSW_UC_dB2":6, 
    "tSW_UC_tips_distance":5, "tSW_UC_C1'-B1-B1pair":1, "tSW_UC_B1-B1pair-C1'pair":2, "tSW_UC_C4'-C1'-B1-B1pair":3, "tSW_UC_B1-B1pair-C1'pair-C4'pair":1, "tSW_UC_alpha_1":2, "tSW_UC_alpha_2":2, "tSW_UC_dB1":2, "tSW_UC_dB2":1, 
    "cHH_UC_tips_distance":5, "cHH_UC_C1'-B1-B1pair":2, "cHH_UC_B1-B1pair-C1'pair":1, "cHH_UC_C4'-C1'-B1-B1pair":2, "cHH_UC_B1-B1pair-C1'pair-C4'pair":2, "cHH_UC_alpha_1":1, "cHH_UC_alpha_2":3, "cHH_UC_dB1":7, "cHH_UC_dB2":3, 
    "tHH_UC_tips_distance":5, "tHH_UC_C1'-B1-B1pair":1, "tHH_UC_B1-B1pair-C1'pair":1, "tHH_UC_C4'-C1'-B1-B1pair":2, "tHH_UC_B1-B1pair-C1'pair-C4'pair":3, "tHH_UC_alpha_1":2, "tHH_UC_alpha_2":2, "tHH_UC_dB1":8, "tHH_UC_dB2":8, 
    "cSH_UC_tips_distance":5, "cSH_UC_C1'-B1-B1pair":2, "cSH_UC_B1-B1pair-C1'pair":2, "cSH_UC_C4'-C1'-B1-B1pair":2, "cSH_UC_B1-B1pair-C1'pair-C4'pair":1, "cSH_UC_alpha_1":2, "cSH_UC_alpha_2":3, "cSH_UC_dB1":5, "cSH_UC_dB2":3, 
    "tSH_UC_tips_distance":2, "tSH_UC_C1'-B1-B1pair":1, "tSH_UC_B1-B1pair-C1'pair":1, "tSH_UC_C4'-C1'-B1-B1pair":2, "tSH_UC_B1-B1pair-C1'pair-C4'pair":1, "tSH_UC_alpha_1":2, "tSH_UC_alpha_2":2, "tSH_UC_dB1":2, "tSH_UC_dB2":7, 
    "cHS_UC_tips_distance":5, "cHS_UC_C1'-B1-B1pair":2, "cHS_UC_B1-B1pair-C1'pair":2, "cHS_UC_C4'-C1'-B1-B1pair":1, "cHS_UC_B1-B1pair-C1'pair-C4'pair":3, "cHS_UC_alpha_1":3, "cHS_UC_alpha_2":2, "cHS_UC_dB1":6, "cHS_UC_dB2":7, 
    "tHS_UC_tips_distance":5, "tHS_UC_C1'-B1-B1pair":3, "tHS_UC_B1-B1pair-C1'pair":2, "tHS_UC_C4'-C1'-B1-B1pair":2, "tHS_UC_B1-B1pair-C1'pair-C4'pair":3, "tHS_UC_alpha_1":3, "tHS_UC_alpha_2":1, "tHS_UC_dB1":5, "tHS_UC_dB2":7, 
    "cSS_UC_tips_distance":5, "cSS_UC_C1'-B1-B1pair":2, "cSS_UC_B1-B1pair-C1'pair":1, "cSS_UC_C4'-C1'-B1-B1pair":3, "cSS_UC_B1-B1pair-C1'pair-C4'pair":1, "cSS_UC_alpha_1":3, "cSS_UC_alpha_2":3, "cSS_UC_dB1":8, "cSS_UC_dB2":5, 
    "tSS_UC_tips_distance":5, "tSS_UC_C1'-B1-B1pair":2, "tSS_UC_B1-B1pair-C1'pair":1, "tSS_UC_C4'-C1'-B1-B1pair":3, "tSS_UC_B1-B1pair-C1'pair-C4'pair":3, "tSS_UC_alpha_1":3, "tSS_UC_alpha_2":1, "tSS_UC_dB1":8, "tSS_UC_dB2":7, 
    "cWW_UG_tips_distance":3, "cWW_UG_C1'-B1-B1pair":2, "cWW_UG_B1-B1pair-C1'pair":3, "cWW_UG_C4'-C1'-B1-B1pair":2, "cWW_UG_B1-B1pair-C1'pair-C4'pair":2, "cWW_UG_alpha_1":2, "cWW_UG_alpha_2":3, "cWW_UG_dB1":4, "cWW_UG_dB2":3, 
    "tWW_UG_tips_distance":2, "tWW_UG_C1'-B1-B1pair":1, "tWW_UG_B1-B1pair-C1'pair":1, "tWW_UG_C4'-C1'-B1-B1pair":2, "tWW_UG_B1-B1pair-C1'pair-C4'pair":2, "tWW_UG_alpha_1":3, "tWW_UG_alpha_2":3, "tWW_UG_dB1":3, "tWW_UG_dB2":4, 
    "cWH_UG_tips_distance":2, "cWH_UG_C1'-B1-B1pair":1, "cWH_UG_B1-B1pair-C1'pair":2, "cWH_UG_C4'-C1'-B1-B1pair":2, "cWH_UG_B1-B1pair-C1'pair-C4'pair":2, "cWH_UG_alpha_1":2, "cWH_UG_alpha_2":2, "cWH_UG_dB1":2, "cWH_UG_dB2":2, 
    "tWH_UG_tips_distance":1, "tWH_UG_C1'-B1-B1pair":2, "tWH_UG_B1-B1pair-C1'pair":2, "tWH_UG_C4'-C1'-B1-B1pair":2, "tWH_UG_B1-B1pair-C1'pair-C4'pair":2, "tWH_UG_alpha_1":2, "tWH_UG_alpha_2":2, "tWH_UG_dB1":6, "tWH_UG_dB2":2, 
    "cHW_UG_tips_distance":2, "cHW_UG_C1'-B1-B1pair":2, "cHW_UG_B1-B1pair-C1'pair":2, "cHW_UG_C4'-C1'-B1-B1pair":1, "cHW_UG_B1-B1pair-C1'pair-C4'pair":2, "cHW_UG_alpha_1":1, "cHW_UG_alpha_2":2, "cHW_UG_dB1":4, "cHW_UG_dB2":4, 
    "tHW_UG_tips_distance":1, "tHW_UG_C1'-B1-B1pair":2, "tHW_UG_B1-B1pair-C1'pair":1, "tHW_UG_C4'-C1'-B1-B1pair":2, "tHW_UG_B1-B1pair-C1'pair-C4'pair":2, "tHW_UG_alpha_1":3, "tHW_UG_alpha_2":2, "tHW_UG_dB1":6, "tHW_UG_dB2":3, 
    "cWS_UG_tips_distance":2, "cWS_UG_C1'-B1-B1pair":4, "cWS_UG_B1-B1pair-C1'pair":2, "cWS_UG_C4'-C1'-B1-B1pair":3, "cWS_UG_B1-B1pair-C1'pair-C4'pair":2, "cWS_UG_alpha_1":2, "cWS_UG_alpha_2":2, "cWS_UG_dB1":2, "cWS_UG_dB2":2, 
    "tWS_UG_tips_distance":5, "tWS_UG_C1'-B1-B1pair":2, "tWS_UG_B1-B1pair-C1'pair":2, "tWS_UG_C4'-C1'-B1-B1pair":2, "tWS_UG_B1-B1pair-C1'pair-C4'pair":2, "tWS_UG_alpha_1":2, "tWS_UG_alpha_2":1, "tWS_UG_dB1":3, "tWS_UG_dB2":5, 
    "cSW_UG_tips_distance":2, "cSW_UG_C1'-B1-B1pair":2, "cSW_UG_B1-B1pair-C1'pair":3, "cSW_UG_C4'-C1'-B1-B1pair":2, "cSW_UG_B1-B1pair-C1'pair-C4'pair":1, "cSW_UG_alpha_1":2, "cSW_UG_alpha_2":2, "cSW_UG_dB1":3, "cSW_UG_dB2":2, 
    "tSW_UG_tips_distance":4, "tSW_UG_C1'-B1-B1pair":1, "tSW_UG_B1-B1pair-C1'pair":1, "tSW_UG_C4'-C1'-B1-B1pair":2, "tSW_UG_B1-B1pair-C1'pair-C4'pair":3, "tSW_UG_alpha_1":2, "tSW_UG_alpha_2":2, "tSW_UG_dB1":2, "tSW_UG_dB2":2, 
    "cHH_UG_tips_distance":5, "cHH_UG_C1'-B1-B1pair":3, "cHH_UG_B1-B1pair-C1'pair":2, "cHH_UG_C4'-C1'-B1-B1pair":2, "cHH_UG_B1-B1pair-C1'pair-C4'pair":2, "cHH_UG_alpha_1":2, "cHH_UG_alpha_2":3, "cHH_UG_dB1":4, "cHH_UG_dB2":5, 
    "tHH_UG_tips_distance":5, "tHH_UG_C1'-B1-B1pair":2, "tHH_UG_B1-B1pair-C1'pair":2, "tHH_UG_C4'-C1'-B1-B1pair":2, "tHH_UG_B1-B1pair-C1'pair-C4'pair":3, "tHH_UG_alpha_1":3, "tHH_UG_alpha_2":2, "tHH_UG_dB1":3, "tHH_UG_dB2":2, 
    "cSH_UG_tips_distance":5, "cSH_UG_C1'-B1-B1pair":1, "cSH_UG_B1-B1pair-C1'pair":2, "cSH_UG_C4'-C1'-B1-B1pair":2, "cSH_UG_B1-B1pair-C1'pair-C4'pair":2, "cSH_UG_alpha_1":2, "cSH_UG_alpha_2":2, "cSH_UG_dB1":3, "cSH_UG_dB2":4, 
    "tSH_UG_tips_distance":5, "tSH_UG_C1'-B1-B1pair":2, "tSH_UG_B1-B1pair-C1'pair":1, "tSH_UG_C4'-C1'-B1-B1pair":2, "tSH_UG_B1-B1pair-C1'pair-C4'pair":1, "tSH_UG_alpha_1":3, "tSH_UG_alpha_2":1, "tSH_UG_dB1":2, "tSH_UG_dB2":2, 
    "cHS_UG_tips_distance":3, "cHS_UG_C1'-B1-B1pair":2, "cHS_UG_B1-B1pair-C1'pair":3, "cHS_UG_C4'-C1'-B1-B1pair":2, "cHS_UG_B1-B1pair-C1'pair-C4'pair":4, "cHS_UG_alpha_1":2, "cHS_UG_alpha_2":3, "cHS_UG_dB1":3, "cHS_UG_dB2":4, 
    "tHS_UG_tips_distance":7, "tHS_UG_C1'-B1-B1pair":1, "tHS_UG_B1-B1pair-C1'pair":3, "tHS_UG_C4'-C1'-B1-B1pair":2, "tHS_UG_B1-B1pair-C1'pair-C4'pair":1, "tHS_UG_alpha_1":2, "tHS_UG_alpha_2":3, "tHS_UG_dB1":2, "tHS_UG_dB2":1, 
    "cSS_UG_tips_distance":2, "cSS_UG_C1'-B1-B1pair":2, "cSS_UG_B1-B1pair-C1'pair":2, "cSS_UG_C4'-C1'-B1-B1pair":2, "cSS_UG_B1-B1pair-C1'pair-C4'pair":2, "cSS_UG_alpha_1":1, "cSS_UG_alpha_2":2, "cSS_UG_dB1":2, "cSS_UG_dB2":3, 
    "tSS_UG_tips_distance":5, "tSS_UG_C1'-B1-B1pair":2, "tSS_UG_B1-B1pair-C1'pair":2, "tSS_UG_C4'-C1'-B1-B1pair":1, "tSS_UG_B1-B1pair-C1'pair-C4'pair":2, "tSS_UG_alpha_1":2, "tSS_UG_alpha_2":2, "tSS_UG_dB1":3, "tSS_UG_dB2":4, 
    "cWW_UU_tips_distance":1, "cWW_UU_C1'-B1-B1pair":2, "cWW_UU_B1-B1pair-C1'pair":3, "cWW_UU_C4'-C1'-B1-B1pair":3, "cWW_UU_B1-B1pair-C1'pair-C4'pair":2, "cWW_UU_alpha_1":2, "cWW_UU_alpha_2":2, "cWW_UU_dB1":2, "cWW_UU_dB2":1, 
    "tWW_UU_tips_distance":3, "tWW_UU_C1'-B1-B1pair":2, "tWW_UU_B1-B1pair-C1'pair":2, "tWW_UU_C4'-C1'-B1-B1pair":2, "tWW_UU_B1-B1pair-C1'pair-C4'pair":2, "tWW_UU_alpha_1":2, "tWW_UU_alpha_2":2, "tWW_UU_dB1":4, "tWW_UU_dB2":5, 
    "cWH_UU_tips_distance":2, "cWH_UU_C1'-B1-B1pair":2, "cWH_UU_B1-B1pair-C1'pair":2, "cWH_UU_C4'-C1'-B1-B1pair":3, "cWH_UU_B1-B1pair-C1'pair-C4'pair":3, "cWH_UU_alpha_1":2, "cWH_UU_alpha_2":3, "cWH_UU_dB1":3, "cWH_UU_dB2":5, 
    "tWH_UU_tips_distance":3, "tWH_UU_C1'-B1-B1pair":2, "tWH_UU_B1-B1pair-C1'pair":2, "tWH_UU_C4'-C1'-B1-B1pair":2, "tWH_UU_B1-B1pair-C1'pair-C4'pair":2, "tWH_UU_alpha_1":3, "tWH_UU_alpha_2":3, "tWH_UU_dB1":2, "tWH_UU_dB2":2, 
    "cHW_UU_tips_distance":1, "cHW_UU_C1'-B1-B1pair":2, "cHW_UU_B1-B1pair-C1'pair":3, "cHW_UU_C4'-C1'-B1-B1pair":1, "cHW_UU_B1-B1pair-C1'pair-C4'pair":3, "cHW_UU_alpha_1":2, "cHW_UU_alpha_2":2, "cHW_UU_dB1":3, "cHW_UU_dB2":4, 
    "tHW_UU_tips_distance":3, "tHW_UU_C1'-B1-B1pair":3, "tHW_UU_B1-B1pair-C1'pair":2, "tHW_UU_C4'-C1'-B1-B1pair":2, "tHW_UU_B1-B1pair-C1'pair-C4'pair":2, "tHW_UU_alpha_1":2, "tHW_UU_alpha_2":3, "tHW_UU_dB1":2, "tHW_UU_dB2":2, 
    "cWS_UU_tips_distance":5, "cWS_UU_C1'-B1-B1pair":1, "cWS_UU_B1-B1pair-C1'pair":1, "cWS_UU_C4'-C1'-B1-B1pair":2, "cWS_UU_B1-B1pair-C1'pair-C4'pair":3, "cWS_UU_alpha_1":2, "cWS_UU_alpha_2":1, "cWS_UU_dB1":2, "cWS_UU_dB2":1, 
    "tWS_UU_tips_distance":3, "tWS_UU_C1'-B1-B1pair":2, "tWS_UU_B1-B1pair-C1'pair":2, "tWS_UU_C4'-C1'-B1-B1pair":3, "tWS_UU_B1-B1pair-C1'pair-C4'pair":2, "tWS_UU_alpha_1":2, "tWS_UU_alpha_2":2, "tWS_UU_dB1":3, "tWS_UU_dB2":3, 
    "cSW_UU_tips_distance":5, "cSW_UU_C1'-B1-B1pair":1, "cSW_UU_B1-B1pair-C1'pair":3, "cSW_UU_C4'-C1'-B1-B1pair":2, "cSW_UU_B1-B1pair-C1'pair-C4'pair":3, "cSW_UU_alpha_1":2, "cSW_UU_alpha_2":3, "cSW_UU_dB1":1, "cSW_UU_dB2":4, 
    "tSW_UU_tips_distance":6, "tSW_UU_C1'-B1-B1pair":3, "tSW_UU_B1-B1pair-C1'pair":1, "tSW_UU_C4'-C1'-B1-B1pair":2, "tSW_UU_B1-B1pair-C1'pair-C4'pair":2, "tSW_UU_alpha_1":1, "tSW_UU_alpha_2":2, "tSW_UU_dB1":3, "tSW_UU_dB2":3, 
    "cHH_UU_tips_distance":5, "cHH_UU_C1'-B1-B1pair":1, "cHH_UU_B1-B1pair-C1'pair":1, "cHH_UU_C4'-C1'-B1-B1pair":3, "cHH_UU_B1-B1pair-C1'pair-C4'pair":2, "cHH_UU_alpha_1":2, "cHH_UU_alpha_2":2, "cHH_UU_dB1":1, "cHH_UU_dB2":5, 
    "tHH_UU_tips_distance":5, "tHH_UU_C1'-B1-B1pair":2, "tHH_UU_B1-B1pair-C1'pair":3, "tHH_UU_C4'-C1'-B1-B1pair":1, "tHH_UU_B1-B1pair-C1'pair-C4'pair":3, "tHH_UU_alpha_1":2, "tHH_UU_alpha_2":4, "tHH_UU_dB1":4, "tHH_UU_dB2":5, 
    "cSH_UU_tips_distance":5, "cSH_UU_C1'-B1-B1pair":1, "cSH_UU_B1-B1pair-C1'pair":3, "cSH_UU_C4'-C1'-B1-B1pair":2, "cSH_UU_B1-B1pair-C1'pair-C4'pair":2, "cSH_UU_alpha_1":3, "cSH_UU_alpha_2":2, "cSH_UU_dB1":2, "cSH_UU_dB2":5, 
    "tSH_UU_tips_distance":5, "tSH_UU_C1'-B1-B1pair":2, "tSH_UU_B1-B1pair-C1'pair":1, "tSH_UU_C4'-C1'-B1-B1pair":3, "tSH_UU_B1-B1pair-C1'pair-C4'pair":3, "tSH_UU_alpha_1":1, "tSH_UU_alpha_2":1, "tSH_UU_dB1":1, "tSH_UU_dB2":5, 
    "cHS_UU_tips_distance":7, "cHS_UU_C1'-B1-B1pair":2, "cHS_UU_B1-B1pair-C1'pair":2, "cHS_UU_C4'-C1'-B1-B1pair":2, "cHS_UU_B1-B1pair-C1'pair-C4'pair":2, "cHS_UU_alpha_1":2, "cHS_UU_alpha_2":2, "cHS_UU_dB1":3, "cHS_UU_dB2":2, 
    "tHS_UU_tips_distance":5, "tHS_UU_C1'-B1-B1pair":1, "tHS_UU_B1-B1pair-C1'pair":2, "tHS_UU_C4'-C1'-B1-B1pair":2, "tHS_UU_B1-B1pair-C1'pair-C4'pair":1, "tHS_UU_alpha_1":1, "tHS_UU_alpha_2":2, "tHS_UU_dB1":4, "tHS_UU_dB2":1, 
    "cSS_UU_tips_distance":5, "cSS_UU_C1'-B1-B1pair":2, "cSS_UU_B1-B1pair-C1'pair":2, "cSS_UU_C4'-C1'-B1-B1pair":2, "cSS_UU_B1-B1pair-C1'pair-C4'pair":3, "cSS_UU_alpha_1":2, "cSS_UU_alpha_2":2, "cSS_UU_dB1":6, "cSS_UU_dB2":4, 
    "tSS_UU_tips_distance":8, "tSS_UU_C1'-B1-B1pair":1, "tSS_UU_B1-B1pair-C1'pair":1, "tSS_UU_C4'-C1'-B1-B1pair":2, "tSS_UU_B1-B1pair-C1'pair-C4'pair":1, "tSS_UU_alpha_1":1, "tSS_UU_alpha_2":2, "tSS_UU_dB1":3, "tSS_UU_dB2":4, 
}  

@trace_unhandled_exceptions
def retrieve_angles(db, res): 
    """
    Retrieve torsion angles from RNANet.db and convert them to degrees
    """

    # Retrieve angle values
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        df = pd.read_sql(f"""SELECT chain_id, nt_name, alpha, beta, gamma, delta, epsilon, zeta, chi 
                            FROM (
                            SELECT chain_id FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
                            WHERE chain.rfam_acc = 'unmappd' AND structure.resolution <= {res} AND issue = 0
                            ) AS c NATURAL JOIN nucleotide
                            WHERE nt_name='A' OR nt_name='C' OR nt_name='G' OR nt_name='U';""", conn)

    # convert to degrees
    j = (180.0/np.pi)
    torsions = df.iloc[:, 0:2].merge(
        df.iloc[:, 2:9].applymap(lambda x: j*x if x <= np.pi else j*x-360.0, na_action='ignore'), 
        left_index=True, right_index=True
    )
    return torsions

def retrieve_eta_theta(db, res):
    """
    Retrieve pseudotorsions from RNANet.db and convert them to degrees
    """
    # Retrieve angle values
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        df = pd.read_sql(f"""SELECT chain_id, nt_name, eta, theta, eta_prime, theta_prime, eta_base, theta_base 
                            FROM (
                            SELECT chain_id FROM chain JOIN structure ON chain.structure_id = structure.pdb_id
                            WHERE chain.rfam_acc = 'unmappd' AND structure.resolution <= {res} AND issue = 0
                            ) AS c NATURAL JOIN nucleotide
                            WHERE nt_name='A' OR nt_name='C' OR nt_name='G' OR nt_name='U';""", conn)

    # convert to degrees
    j = (180.0/np.pi)
    pseudotorsions = df.iloc[:, 0:2].merge(
        df.iloc[:, 2:8].applymap(lambda x: j*x if x <= np.pi else j*x-360.0, na_action='ignore'), 
        left_index=True, right_index=True
    )
    return pseudotorsions

def get_euclidian_distance(L1, L2):
    """
    Returns the distance between two points (coordinates in lists)
    """

    if len(L1)*len(L2) == 0:
        return np.nan
    
    if len(L1) == 1:
        L1 = L1[0]
    if len(L2) == 1:
        L2 = L2[0]

    e = 0
    for i in range(len(L1)):
        try:
            e += float(L1[i] - L2[i])**2
        except TypeError:
            print("Terms: ", L1, L2)
        except IndexError:
            print("Terms: ", L1, L2)

    return np.sqrt(e)

def get_flat_angle(L1, L2, L3):
    """
    Returns the flat angles (in radians) defined by 3 points.
    L1, L2, L3 : lists of (x,y,z) coordinates
    Returns NaN if one of the lists is empty.
    """

    if len(L1)*len(L2)*len(L3) == 0:
        return np.nan

    return calc_angle(Vector(L1[0]), Vector(L2[0]), Vector(L3[0]))*(180/np.pi)

def get_torsion_angle(L1, L2, L3, L4):
    if len(L1)*len(L2)*len(L3)*len(L4) == 0:
        return np.nan
    
    return calc_dihedral(Vector(L1[0]), Vector(L2[0]), Vector(L3[0]), Vector(L4[0]))*(180/np.pi)

def pos_b1(res):
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

    if len(coordb1):
        return [coordb1]
    else:
        return []

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
    if len(coordb2):
        return [coordb2]
    else:
        return []

@trace_unhandled_exceptions
def measures_aa(name, s, thr_idx):
    """
    Measures the distance between atoms linked by covalent bonds
    """

    # do not recompute something already computed
    if os.path.isfile(runDir + "/results/geometry/all-atoms/distances/dist_atoms_" + name + ".csv"):
        return
    
    last_o3p = [] # o3 'of the previous nucleotide linked to the P of the current nucleotide
    l_common = []
    l_purines = []
    l_pyrimidines = []
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} measure_aa_dists({name})")

    chain = next(s[0].get_chains()) # 1 chain per file
    residues = list(chain.get_residues())
    pbar = tqdm(total=len(residues), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {name} measure_aa_dists", unit="res", leave=False)
    pbar.update(0)
    for res in chain :
        
        # for residues A, G, C, U
        op3_p = []
        p_op1 = []
        p_op2 = []
        p_o5p = []
        o5p_c5p = []
        c5p_c4p = []
        c4p_o4p = []
        o4p_c1p = []
        c1p_c2p = []
        c2p_o2p = []
        c2p_c3p = []
        c3p_o3p = []
        c4p_c3p = []
        
        # if res = A or G
        c1p_n9 = None
        n9_c8 = None
        c8_n7 = None
        n7_c5 = None
        c5_c6 = None
        c6_n1 = None
        n1_c2 = None
        c2_n3 = None
        n3_c4 = None
        c4_n9 = None
        c4_c5 = None
        # if res = G
        c6_o6 = None
        c2_n2 = None
        # if res = A
        c6_n6 = None
        # if res = C or U
        c1p_n1 = None
        n1_c6 = None
        c6_c5 = None
        c5_c4 = None
        c4_n3 = None
        n3_c2 = None
        c2_n1 = None
        c2_o2 = None
        # if res = C
        c4_n4 = None
        # if res = U
        c4_o4 = None
        last_o3p_p = None


        if res.get_resname()=='A' or res.get_resname()=='G' or res.get_resname()=='C' or res.get_resname()=='U' :

            # get the coordinates of the atoms
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_op3 = [ atom.get_coord() for atom in res if "OP3" in atom.get_fullname() ] # OP3 belongs to previous nucleotide !
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
            
            if len(atom_op3):
                last_o3p_p = get_euclidian_distance(atom_op3, atom_p) # This nucleotide has an OP3 atom (likely the begining of a chain)
            else:
                last_o3p_p = get_euclidian_distance(last_o3p, atom_p) # link with the previous nucleotide
            p_op1 = get_euclidian_distance(atom_op1, atom_p)
            p_op2 = get_euclidian_distance(atom_op2, atom_p)
            p_o5p = get_euclidian_distance(atom_o5p, atom_p)
            o5p_c5p = get_euclidian_distance(atom_o5p, atom_c5p)
            c5p_c4p = get_euclidian_distance(atom_c5p, atom_c4p)
            c4p_o4p = get_euclidian_distance(atom_c4p, atom_o4p)
            c4p_c3p = get_euclidian_distance(atom_c4p, atom_c3p)
            o4p_c1p = get_euclidian_distance(atom_o4p, atom_c1p)
            c1p_c2p = get_euclidian_distance(atom_c1p, atom_c2p)
            c2p_o2p = get_euclidian_distance(atom_c2p, atom_o2p)
            c2p_c3p = get_euclidian_distance(atom_c2p, atom_c3p)
            c3p_o3p = get_euclidian_distance(atom_c3p, atom_o3p)

            last_o3p = atom_o3p # o3' of this residue becomes the previous o3' of the following
            
            # different cases for the aromatic cycles
            if res.get_resname()=='A' or res.get_resname()=='G': 
                # compute the distances between atoms of aromatic cycles
                c1p_n9 = get_euclidian_distance(atom_c1p, atom_n9)
                n9_c8 = get_euclidian_distance(atom_n9, atom_c8)
                c8_n7 = get_euclidian_distance(atom_c8, atom_n7)
                n7_c5 = get_euclidian_distance(atom_n7, atom_c5)
                c5_c6 = get_euclidian_distance(atom_c5, atom_c6)
                c6_o6 = get_euclidian_distance(atom_c6, atom_o6)
                c6_n6 = get_euclidian_distance(atom_c6, atom_n6)
                c6_n1 = get_euclidian_distance(atom_c6, atom_n1)
                n1_c2 = get_euclidian_distance(atom_n1, atom_c2)
                c2_n2 = get_euclidian_distance(atom_c2, atom_n2)
                c2_n3 = get_euclidian_distance(atom_c2, atom_n3)
                n3_c4 = get_euclidian_distance(atom_n3, atom_c4)
                c4_n9 = get_euclidian_distance(atom_c4, atom_n9)
                c4_c5 = get_euclidian_distance(atom_c4, atom_c5)
            if res.get_resname()=='C' or res.get_resname()=='U' :
                c1p_n1 = get_euclidian_distance(atom_c1p, atom_n1)
                n1_c6 = get_euclidian_distance(atom_n1, atom_c6)
                c6_c5 = get_euclidian_distance(atom_c6, atom_c5)
                c5_c4 = get_euclidian_distance(atom_c5, atom_c4)
                c4_n3 = get_euclidian_distance(atom_c4, atom_n3)
                n3_c2 = get_euclidian_distance(atom_n3, atom_c2)
                c2_o2 = get_euclidian_distance(atom_c2, atom_o2)
                c2_n1 = get_euclidian_distance(atom_c2, atom_n1)
                c4_n4 = get_euclidian_distance(atom_c4, atom_n4)
                c4_o4 = get_euclidian_distance(atom_c4, atom_o4)

            l_common.append([res.get_resname(), last_o3p_p, p_op1, p_op2, p_o5p, o5p_c5p, c5p_c4p, c4p_o4p, c4p_c3p, o4p_c1p, c1p_c2p, c2p_o2p, c2p_c3p, c3p_o3p] )
            l_purines.append([c1p_n9, n9_c8, c8_n7, n7_c5, c5_c6, c6_o6, c6_n6, c6_n1, n1_c2, c2_n2, c2_n3, n3_c4, c4_n9, c4_c5])
            l_pyrimidines.append([c1p_n1, n1_c6, c6_c5, c5_c4, c4_n3, n3_c2, c2_o2, c2_n1, c4_n4, c4_o4])
            pbar.update(1)

    df_comm = pd.DataFrame(l_common, columns=["Residue", "O3'-P", "P-OP1", "P-OP2", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-O4'", "C4'-C3'", "O4'-C1'", "C1'-C2'", "C2'-O2'", "C2'-C3'", "C3'-O3'"])
    df_pur = pd.DataFrame(l_purines, columns=["C1'-N9", "N9-C8", "C8-N7", "N7-C5", "C5-C6", "C6-O6", "C6-N6", "C6-N1", "N1-C2", "C2-N2", "C2-N3", "N3-C4", "C4-N9", "C4-C5" ])
    df_pyr = pd.DataFrame(l_pyrimidines, columns=["C1'-N1", "N1-C6", "C6-C5", "C5-C4", "C4-N3", "N3-C2", "C2-O2", "C2-N1", "C4-N4", "C4-O4"])
    df = pd.concat([df_comm, df_pur, df_pyr], axis = 1)
    pbar.close()
    
    df.to_csv(runDir + "/results/geometry/all-atoms/distances/dist_atoms_" + name + ".csv")

@trace_unhandled_exceptions
def measures_pyle(name, s, thr_idx):
    """
    Measures the distances and plane angles involving C1' and P atoms 
    Saves the results in a dataframe
    """

    # do not recompute something already computed
    if (os.path.isfile(runDir + '/results/geometry/Pyle/angles/flat_angles_pyle_' + name + '.csv') and
        os.path.isfile(runDir + "/results/geometry/Pyle/distances/distances_pyle_" + name + ".csv")):
        return

    l_dist = []
    l_angl = []
    last_p = []
    last_c1p = []
    last_c4p = []

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} measures_pyle({name})")

    chain = next(s[0].get_chains())
    for res in tqdm(chain, position=thr_idx+1, desc=f"Worker {thr_idx+1}: {name} measures_pyle", unit="res", leave=False):
        p_c1p_psuiv = np.nan
        c1p_psuiv_c1psuiv = np.nan
        if res.get_resname() not in ['ATP', 'CCC', 'A3P', 'A23', 'GDP', 'RIA', "2BA"] :
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
            atom_c4p = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ]
            if len(atom_c1p) > 1:
                for atom in res:
                    if "C1'" in atom.get_fullname():
                        print("\n", atom.get_fullname(), "-", res.get_resname(), "\n")

            p_c1p_psuiv = get_flat_angle(last_p, last_c1p, atom_p)
            c1p_psuiv_c1psuiv = get_flat_angle(last_c1p, atom_p, atom_c1p)
            c1p_psuiv = get_euclidian_distance(last_c1p, atom_p)
            p_c1p = get_euclidian_distance(atom_p, atom_c1p)
            c4p_psuiv = get_euclidian_distance(last_c4p, atom_p)
            p_c4p = get_euclidian_distance(atom_p, atom_c4p)

            last_p = atom_p
            last_c1p = atom_c1p
            last_c4p = atom_c4p

            l_dist.append([res.get_resname(), c1p_psuiv, p_c1p, c4p_psuiv, p_c4p])
            l_angl.append([res.get_resname(), p_c1p_psuiv, c1p_psuiv_c1psuiv])

    df = pd.DataFrame(l_dist, columns=["Residue", "C1'-P", "P-C1'", "C4'-P", "P-C4'"])
    df.to_csv(runDir + "/results/geometry/Pyle/distances/distances_pyle_" + name + ".csv")
    df = pd.DataFrame(l_angl, columns=["Residue", "P-C1'-P°", "C1'-P°-C1'°"])
    df.to_csv(runDir + "/results/geometry/Pyle/angles/flat_angles_pyle_"+name+".csv")

@trace_unhandled_exceptions
def measures_hrna(name, s, thr_idx):
    """
    Measures the distance/angles between the atoms of the HiRE-RNA model linked by covalent bonds
    """
    
    # do not recompute something already computed
    if (os.path.isfile(runDir + '/results/geometry/HiRE-RNA/distances/distances_HiRERNA '+name+'.csv') and 
        os.path.isfile(runDir + '/results/geometry/HiRE-RNA/angles/angles_HiRERNA '+name+'.csv') and 
        os.path.isfile(runDir + '/results/geometry/HiRE-RNA/torsions/torsions_HiRERNA '+name+'.csv')):
        return

    l_dist = []
    l_angl = []
    l_tors = []
    last_c4p = []
    last_c5p = []
    last_c1p = []
    last_o5p = []

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} measures_hrna({name})")

    chain = next(s[0].get_chains())
    residues=list(chain.get_residues())
    for res in tqdm(chain, position=thr_idx+1, desc=f"Worker {thr_idx+1}: {name} measures_hrna", unit="res", leave=False):
        # distances
        p_o5p = None
        o5p_c5p = None
        c5p_c4p = None
        c4p_c1p = None
        c1p_b1 = None
        b1_b2 = None
        last_c4p_p = np.nan
        
        # angles
        p_o5p_c5p = None
        o5p_c5p_c4p = None
        c5p_c4p_c1p = None
        c4p_c1p_b1 = None
        c1p_b1_b2 = None
        lastc4p_p_o5p = None
        lastc5p_lastc4p_p = None
        lastc1p_lastc4p_p = None

        # torsions
        p_o5_c5_c4 = np.nan
        o5_c5_c4_c1 = np.nan
        c5_c4_c1_b1 = np.nan
        c4_c1_b1_b2 = np.nan
        o5_c5_c4_psuiv = np.nan
        c5_c4_psuiv_o5suiv = np.nan
        c4_psuiv_o5suiv_c5suiv = np.nan
        c1_c4_psuiv_o5suiv = np.nan

        if res.get_resname() not in ['ATP', 'CCC', 'A3P', 'A23', 'GDP', 'RIA', "2BA"] : # several phosphate groups, ignore
            atom_p   = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_o5p = [ atom.get_coord() for atom in res if "O5'" in atom.get_fullname() ]
            atom_c5p = [ atom.get_coord() for atom in res if "C5'" in atom.get_fullname() ]
            atom_c4p = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
            atom_b1 = pos_b1(res) # position b1 to be calculated, depending on the case
            atom_b2 = pos_b2(res) # position b2 to be calculated only for those with 2 cycles

            # Distances. If one of the atoms is empty, the euclidian distance returns NaN.
            last_c4p_p  = get_euclidian_distance(last_c4p, atom_p)
            p_o5p       = get_euclidian_distance(atom_p, atom_o5p)
            o5p_c5p     = get_euclidian_distance(atom_o5p, atom_c5p)
            c5p_c4p     = get_euclidian_distance(atom_c5p, atom_c4p)
            c4p_c1p     = get_euclidian_distance(atom_c4p, atom_c1p)
            c1p_b1      = get_euclidian_distance(atom_c1p, atom_b1)
            b1_b2       = get_euclidian_distance(atom_b1, atom_b2)

            # flat angles. Same.
            lastc4p_p_o5p       = get_flat_angle(last_c4p, atom_p, atom_o5p)
            lastc1p_lastc4p_p   = get_flat_angle(last_c1p, last_c4p, atom_p)
            lastc5p_lastc4p_p   = get_flat_angle(last_c5p, last_c4p, atom_p)
            p_o5p_c5p           = get_flat_angle(atom_p, atom_o5p, atom_c5p)
            o5p_c5p_c4p         = get_flat_angle(atom_o5p, atom_c5p, atom_c4p)
            c5p_c4p_c1p         = get_flat_angle(atom_c5p, atom_c4p, atom_c1p)
            c4p_c1p_b1          = get_flat_angle(atom_c4p, atom_c1p, atom_b1)
            c1p_b1_b2           = get_flat_angle(atom_c1p, atom_b1, atom_b2)

            # torsions. Idem.
            p_o5_c5_c4  = get_torsion_angle(atom_p, atom_o5p, atom_c5p, atom_c4p)
            o5_c5_c4_c1 = get_torsion_angle(atom_o5p, atom_c5p, atom_c4p, atom_c1p)
            c5_c4_c1_b1 = get_torsion_angle(atom_c5p, atom_c4p, atom_c1p, atom_b1)
            c4_c1_b1_b2 = get_torsion_angle(atom_c4p, atom_c1p, atom_b1, atom_b2)
            o5_c5_c4_psuiv = get_torsion_angle(last_o5p, last_c5p, last_c4p, atom_p)
            c5_c4_psuiv_o5suiv = get_torsion_angle(last_c5p, last_c4p, atom_p, atom_o5p)
            c4_psuiv_o5suiv_c5suiv = get_torsion_angle(last_c4p, atom_p, atom_o5p, atom_c5p)
            c1_c4_psuiv_o5suiv = get_torsion_angle(last_c1p, last_c4p, atom_p, atom_o5p)

            last_c4p = atom_c4p
            last_c5p = atom_c5p
            last_c1p = atom_c1p
            last_o5p = atom_o5p
            l_dist.append([res.get_resname(), last_c4p_p, p_o5p, o5p_c5p, c5p_c4p, c4p_c1p, c1p_b1, b1_b2])
            l_angl.append([res.get_resname(), lastc4p_p_o5p, lastc1p_lastc4p_p, lastc5p_lastc4p_p, p_o5p_c5p, o5p_c5p_c4p, c5p_c4p_c1p, c4p_c1p_b1, c1p_b1_b2])
            l_tors.append([res.get_resname(), p_o5_c5_c4, o5_c5_c4_c1, c5_c4_c1_b1, c4_c1_b1_b2, o5_c5_c4_psuiv, c5_c4_psuiv_o5suiv, c4_psuiv_o5suiv_c5suiv, c1_c4_psuiv_o5suiv])

    df = pd.DataFrame(l_dist, columns=["Residue", "C4'-P", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-C1'", "C1'-B1", "B1-B2"])
    df.to_csv(runDir + '/results/geometry/HiRE-RNA/distances/distances_HiRERNA '+name+'.csv')
    df = pd.DataFrame(l_angl, columns=["Residue", "C4'-P-O5'", "C1'-C4'-P", "C5'-C4'-P", "P-O5'-C5'", "O5'-C5'-C4'", "C5'-C4'-C1'", "C4'-C1'-B1", "C1'-B1-B2"])
    df.to_csv(runDir + '/results/geometry/HiRE-RNA/angles/angles_HiRERNA ' + name + ".csv")
    df=pd.DataFrame(l_tors, columns=["Residue", "P-O5'-C5'-C4'", "O5'-C5'-C4'-C1'", "C5'-C4'-C1'-B1", "C4'-C1'-B1-B2", "O5'-C5'-C4'-P°", "C5'-C4'-P°-O5'°", "C4'-P°-O5'°-C5'°", "C1'-C4'-P°-O5'°"])
    df.to_csv(runDir + '/results/geometry/HiRE-RNA/torsions/torsions_HiRERNA '+name+'.csv')

@trace_unhandled_exceptions
def measures_hrna_basepairs(name, s, path_to_3D_data, thr_idx):
    """
    Open a rna_only/ file, and run measures_hrna_basepairs_chain() on every chain
    """  

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} measures_hrna_basepairs({name})")
    
    l = []
    chain = next(s[0].get_chains())
            
    # do not recompute something already computed
    if os.path.isfile(runDir + "/results/geometry/HiRE-RNA/basepairs/basepairs_"+name+".csv"):
        return

    df = pd.read_csv(os.path.abspath(path_to_3D_data +"datapoints/" + name))

    # if df['index_chain'][0] == 1: # ignore files with numbering errors : TODO : remove when we get DSSR Pro, there should not be numbering errors anymore
    l = measures_hrna_basepairs_chain(name, chain, df, thr_idx)
    df_calc = pd.DataFrame(l, columns=["type_LW", "nt1_idx", "nt1_res", "nt2_idx", "nt2_res", "Distance", 
                                       "211_angle", "112_angle", "dB1", "dB2", "alpha1", "alpha2", "3211_torsion", "1123_torsion"])
    df_calc.to_csv(runDir + "/results/geometry/HiRE-RNA/basepairs/"+'basepairs_' + name + '.csv', float_format="%.3f")

@trace_unhandled_exceptions
def measures_hrna_basepairs_chain(name, chain, df, thr_idx):
    """
    Cleanup of the dataset
    measurements of distances and angles between paired nucleotides in the chain
    """

    results = []
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

    pairs = df[['index_chain', 'old_nt_resnum', 'paired', 'pair_type_LW']] # columns we keep
    for i in range(pairs.shape[0]): # we remove the lines where no pairing (NaN in paired)
        index_with_nan = pairs.index[pairs.iloc[:,2].isnull()]
        pairs.drop(index_with_nan, 0, inplace=True)

    paired_int = []
    for i in pairs.index:   # convert values ​​from paired to integers or lists of integers
        paired = pairs.at[i, 'paired']
        if type(paired) is np.int64 or type(paired) is np.float64:
            paired_int.append(int(paired))
        else :  #strings
            if len(paired) < 3: # a single pairing
                paired_int.append(int(paired))         
            else : # several pairings
                paired = paired.split(',')
                l = [ int(i) for i in paired ]
                paired_int.append(l)

    pair_type_LW_bis = []
    for j in pairs.index:
        pair_type_LW = pairs.at[j, 'pair_type_LW']
        if len(pair_type_LW) < 4 : # a single pairing
            pair_type_LW_bis.append(pair_type_LW)
        else : # several pairings
            pair_type_LW = pair_type_LW.split(',')
            l = [ i for i in pair_type_LW ]
            pair_type_LW_bis.append(pair_type_LW)

    # addition of these new columns
    pairs.insert(4, "paired_int", paired_int, True)
    pairs.insert(5, "pair_type_LW_bis", pair_type_LW_bis, True)
    
    indexNames = pairs[pairs['paired_int'] == 0].index
    pairs.drop(indexNames, inplace=True) # deletion of lines with a 0 in paired_int (matching to another RNA chain)

    for i in tqdm(pairs.index, position=thr_idx+1, desc=f"Worker {thr_idx+1}: {name} measures_hrna_basepairs_chain", unit="res", leave=False):
        # calculations for each row of the pairs dataset
        index = pairs.at[i, 'index_chain'] 
        res1 =  chain[(' ', index, ' ')].get_resname()
        if res1 not in ['A','C','G','U']:
            continue
        type_LW = pairs.at[i, 'pair_type_LW_bis'] # pairing type
        num_paired = pairs.at[i, 'paired_int'] # number (index_chain) of the paired nucleotide
        
        if type(num_paired) is int or type(num_paired) is np.int64:
            res2 =  chain[(' ', num_paired, ' ')].get_resname()
            if res2 not in ["A","C","G","U"]:
                continue
            measures = basepair_measures(chain[(' ', index, ' ')], chain[(' ', num_paired, ' ')])
            if measures is not None:
                results.append([type_LW, index, res1, num_paired, res2] + measures)
        else:
            for j in range(len(num_paired)): # if several pairings, process them one by one
                if num_paired[j] != 0:
                    res2 =  chain[(' ', num_paired[j], ' ')].get_resname()
                    if res2 not in ["A","C","G","U"]:
                        continue
                    measures = basepair_measures(chain[(' ', index, ' ')], chain[(' ', num_paired[j], ' ')])
                    if measures is not None:
                        results.append([type_LW[j], index, res1, num_paired[j], res2] + measures)

    return results

@trace_unhandled_exceptions
def basepair_measures(res, pair):
    """
    Measurement of the flat angles describing a basepair in the HiRE-RNA model
    """

    if res.get_resname()=='C' or res.get_resname()=='U' :
        atom_c4_res = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ] 
        atom_c1p_res = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
        atom_b1_res = pos_b1(res)
        if not len(atom_c4_res) or not len(atom_c1p_res) or not len(atom_b1_res):
            return
        a3_res = Vector(atom_c4_res[0])
        a2_res = Vector(atom_c1p_res[0])
        a1_res = Vector(atom_b1_res[0])
    if res.get_resname()=='A' or res.get_resname()=='G' :
        atom_c1p_res = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
        atom_b1_res = pos_b1(res)
        atom_b2_res = pos_b2(res)
        if not len(atom_c1p_res) or not len(atom_b1_res) or not len(atom_b2_res): 
            return
        a3_res = Vector(atom_c1p_res[0])
        a2_res = Vector(atom_b1_res[0])
        a1_res = Vector(atom_b2_res[0])
        
    if pair.get_resname()=='C' or pair.get_resname()=='U' :
        atom_c4_pair = [ atom.get_coord() for atom in pair if "C4'" in atom.get_fullname() ]
        atom_c1p_pair = [ atom.get_coord() for atom in pair if "C1'" in atom.get_fullname() ]
        atom_b1_pair = pos_b1(pair)
        if not len(atom_c4_pair) or not len(atom_c1p_pair) or not len(atom_b1_pair):
            return
        a3_pair = Vector(atom_c4_pair[0])
        a2_pair = Vector(atom_c1p_pair[0])
        a1_pair = Vector(atom_b1_pair[0])
    if pair.get_resname()=='A' or pair.get_resname()=='G' :
        atom_c1p_pair = [ atom.get_coord() for atom in pair if "C1'" in atom.get_fullname() ]
        atom_b1_pair = pos_b1(pair)
        atom_b2_pair = pos_b2(pair)
        if not len(atom_c1p_pair) or not len(atom_b1_pair) or not len(atom_b2_pair): # No C1' atom in the paired nucleotide, skip measures.
            return
        a3_pair = Vector(atom_c1p_pair[0])
        a2_pair = Vector(atom_b1_pair[0])
        a1_pair = Vector(atom_b2_pair[0])

    # Bond vectors
    res_32 = a3_res - a2_res
    res_12 = a1_res - a2_res
    pair_32 = a3_pair - a2_pair
    pair_12 = a1_pair - a2_pair
    rho = a1_res - a1_pair # from pair to res

    # dist
    dist = rho.norm()

    # we calculate the 2 plane angles
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        b = res_12.angle(rho)*(180/np.pi)   # equal to the previous implementation
        c = pair_12.angle(-rho)*(180/np.pi) #
    
    # Compute plane vectors
    n1 = (res_32**res_12).normalized() # ** between vectors, is the cross product
    n2 = (pair_32**pair_12).normalized()

    # Distances between base tip and the other base's plane (orthogonal projection)
    # if angle(rho, n) > pi/2 the distance is negative (signed following n)
    d1 = rho*n1 # projection of rho on axis n1
    d2 = rho*n2

    # Now the projection of rho in the planes. It's just a sum of the triangles' two other edges.
    p1 = (-rho+n1**d1).normalized() # between vector and scalar, ** is the multiplication by a scalar
    p2 = (rho-n2**d2).normalized()

    # Measure tau, the dihedral
    u = (res_12**rho).normalized()
    v = (rho**pair_12).normalized()
    cosTau1 = n1*u
    cosTau2 = v*n2 
    
    # cosTau is enough to compute alpha, but we can't distinguish
    # yet betwwen tau and -tau. If the full computation if required, then:
    tau1 = np.arccos(cosTau1)*(180/np.pi)
    tau2 = np.arccos(cosTau2)*(180/np.pi)
    w1 = u**n1
    w2 = v**n2
    if res_12*w1 < 0:
        tau1 = -tau1
    if pair_12*w2 < 0:
        tau2 = -tau2

    # And finally, the a1 and a2 angles between res_12 and p1 / pair_12 and p2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        a1 = (-res_12).angle(p1)*(180/np.pi)
        a2 = (-pair_12).angle(p2)*(180/np.pi)
    if cosTau1 > 0:
        # CosTau > 0 (Tau < 90 or Tau > 270) implies that alpha > 180.
        a1 = -a1
    if cosTau2 > 0:
        a2 = -a2

    return [dist, b, c, d1, d2, a1, a2, tau1, tau2]

@trace_unhandled_exceptions
def GMM_histo(data_ori, name_data, scan, toric=False, hist=True, col=None, save=True) :
    """
    Plot Gaussian-Mixture-Model (with or without histograms)
    """

    if len(data_ori) < 30:
        warn(f"We only have {len(data_ori)} observations of {name_data}, we cannot model it. Skipping.")
        return

    data_ori = np.array(data_ori)

    if toric:
        # Extend the data on the right and on the left (for angles)
        data = np.concatenate([data_ori, data_ori-360.0, data_ori+360.0])
    else:
        data = data_ori

    # chooses the number of components based on the maximum likelihood value (maxlogv)
    if scan:
        n_components_range = np.arange(8)+1
        # aic = []
        # bic = []
        maxlogv=[]
        md = np.array(data).reshape(-1,1)
        nb_components = 1
        nb_log_max = n_components_range[0]
        log_max = 0
        for n_comp in n_components_range:
            gmm = GaussianMixture(n_components=n_comp, random_state=1234).fit(md)
            # aic.append(abs(gmm.aic(md)))
            # bic.append(abs(gmm.bic(md)))
            maxlogv.append(gmm.lower_bound_)
            if gmm.lower_bound_== max(maxlogv) : # takes the maximum
                nb_components = n_comp
                # if there is convergence, keep the first maximum found
                if abs(gmm.lower_bound_-log_max) < 0.02 : # threshold=0.02
                    nb_components = nb_log_max
                    break
            log_max = max(maxlogv)
            nb_log_max = n_comp
    else:
        try:
            nb_components = modes_data[name_data]
        except KeyError:
            warn(f"Unexpected key {name_data} not known in geometric_stats.py mode_data. Skipping.")
            return # unexpected atom ? skip it...
        if toric:
            nb_components = nb_components * 2 + 1 # because we extend the xrange for toric computation. It will be restored later.
    
    # Now compute the final GMM
    obs = np.array(data).reshape(-1,1) # still on extended data
    g = GaussianMixture(n_components=nb_components, random_state=1234)
    g.fit(obs)

    if toric:
        # Now decide which to keep
        keep = []
        weights = []
        means = []
        covariances = []
        sum_weights = 0.0
        for m in g.means_:
            keep.append(m > -180 and m <= 180)
        for i, w in enumerate(g.weights_):
            if not keep[i]:
                continue
            sum_weights += w
        for i in range(nb_components):
            if not keep[i]:
                continue
            means.append(g.means_[i])
            covariances.append(g.covariances_[i])
            weights.append(g.weights_[i]/sum_weights)
        nb_components = len(means)
    else:
        weights = g.weights_
        means = g.means_
        covariances = g.covariances_

    if nb_components == 0:
        # Happens when the gaussians averages are outside [-180, 180]
        # an have been eliminated. Fix: increase the number of components
        # so that at least one is inside [-180,180]
        warn(f"Found 0 gaussians in interval [-180,180] for the {name_data} GMM. Please retry with a higher number of gaussians. Ignoring the measure for now.", error=True)
        return

    # plot histograms if asked, with the appropriate number of components
    if hist:
        plt.hist(data_ori, color="green", edgecolor='black', linewidth=1.2, bins=50, density=True)
    if toric:
        plt.xlabel("Angle (Degrees)")
    else:
        plt.xlabel("Distance (Angströms)")
    plt.ylabel("Density")

    # Prepare the GMM curve with some absciss points
    if toric:
        x = np.linspace(-360.0,360.0,721)
    else:
        D = obs.ravel()
        xmin = D.min()
        #xmax = min(10.0, D.max())
        xmax = D.max()
        x = np.linspace(xmin,xmax,1000)
    colors=['red', 'blue', 'gold', 'cyan', 'magenta', 'white', 'black', 'green']

    # prepare the dictionary to save the parameters
    summary_data = {}
    summary_data["measure"] = name_data
    summary_data["weights"] = []
    summary_data["means"] = []
    summary_data["std"] = []

    # plot
    curves = []
    newx = None # to be defined inside the loop
    for i in range(nb_components):

        # store the parameters
        mean = means[i]
        sigma = np.sqrt(covariances[i])
        weight = weights[i]
        summary_data["means"].append("{:.2f}".format(float(str(mean).strip("[]"))))
        summary_data["std"].append("{:.2f}".format(float(str(sigma).strip("[]"))))
        summary_data["weights"].append("{:.2f}".format(float(str(weight).strip("[]"))))

        # compute the right x and y data to plot
        y = weight*st.norm.pdf(x, mean, sigma)
        if toric:
            y_mod = (((y[0]+180.0)%360.0)-180.0)
            x_mod = (((x+180.0)%360.0)-180.0)
            s = sorted(zip(x_mod,y_mod))
            newx = []
            newy = []
            for k in range(0, len(s), 2):
                if k == 362.0:
                    continue # this value is dealt with when k = 360.0
                # print(k, "summing: ", s[k-int(k>360)], s[k+1-int(k>360)])
                newx.append(s[k-int(k>360)][0])
                if k == 360.0:
                    newy.append(s[k][1]+s[k+1][1]+s[k+2][1])
                else:
                    newy.append(s[k-int(k>360)][1]+s[k+1-int(k>360)][1])
        else:
            newx = x
            newy = y[0]

        if hist:
            # plot on top of the histograms
            plt.plot(newx, newy, c=colors[i])
        else:
            # store for later summation
            curves.append(np.array(newy))

    if hist:
        plt.title(f"Histogram of {name_data} with GMM of {nb_components} components (" + str(len(data_ori))+" values)")
        if save:
            plt.savefig(f"Histogram_{name_data}_{nb_components}_comps.png")
            plt.close()
    else:
        # Plot their sum, do not save figure yet
        plt.plot(newx, sum(curves), c=col, label=name_data)
        plt.legend()

        # Save the json
        with open(runDir + "/results/geometry/json/" + name_data + ".json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)
        notify("Saved " + name_data + ".json")

@trace_unhandled_exceptions
def gmm_aa_dists(scan):
    """
    Draw the figures representing the data on the measurements of distances between atoms
    """

    setproctitle("GMM (all atoms, distances)")

    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/all-atoms/distances/dist_atoms.csv"))

    last_o3p_p  = df["O3'-P"][~ np.isnan(df["O3'-P"])].values.tolist()
    p_op1       = df["P-OP1"][~ np.isnan(df["P-OP1"])].values.tolist()
    p_op2       = df["P-OP2"][~ np.isnan(df["P-OP2"])].values.tolist()
    p_o5p       = df["P-O5'"][~ np.isnan(df["P-O5'"])].values.tolist()
    o5p_c5p     = df["O5'-C5'"][~ np.isnan(df["O5'-C5'"])].values.tolist()
    c5p_c4p     = df["C5'-C4'"][~ np.isnan(df["C5'-C4'"])].values.tolist()
    c4p_o4p     = df["C4'-O4'"][~ np.isnan(df["C4'-O4'"])].values.tolist()
    o4p_c1p     = df["O4'-C1'"][~ np.isnan(df["O4'-C1'"])].values.tolist()
    c1p_c2p     = df["C1'-C2'"][~ np.isnan(df["C1'-C2'"])].values.tolist()
    c2p_o2p     = df["C2'-O2'"][~ np.isnan(df["C2'-O2'"])].values.tolist()
    c2p_c3p     = df["C2'-C3'"][~ np.isnan(df["C2'-C3'"])].values.tolist()
    c3p_o3p     = df["C3'-O3'"][~ np.isnan(df["C3'-O3'"])].values.tolist()
    c4p_c3p     = df["C4'-C3'"][~ np.isnan(df["C4'-C3'"])].values.tolist()
    
    #if res = A ou G
    c1p_n9 = df["C1'-N9"][~ np.isnan(df["C1'-N9"])].values.tolist()
    n9_c8 = df["N9-C8"][~ np.isnan(df["N9-C8"])].values.tolist()
    c8_n7 = df["C8-N7"][~ np.isnan(df["C8-N7"])].values.tolist()
    n7_c5 = df["N7-C5"][~ np.isnan(df["N7-C5"])].values.tolist()
    c5_c6 = df["C5-C6"][~ np.isnan(df["C5-C6"])].values.tolist()
    c6_n1 = df["C6-N1"][~ np.isnan(df["C6-N1"])].values.tolist()
    n1_c2 = df["N1-C2"][~ np.isnan(df["N1-C2"])].values.tolist()
    c2_n3 = df["C2-N3"][~ np.isnan(df["C2-N3"])].values.tolist()
    n3_c4 = df["N3-C4"][~ np.isnan(df["N3-C4"])].values.tolist()
    c4_n9 = df["C4-N9"][~ np.isnan(df["C4-N9"])].values.tolist()
    c4_c5 = df["C4-C5"][~ np.isnan(df["C4-C5"])].values.tolist()
    #if res = G
    c6_o6 = df["C6-O6"][~ np.isnan(df["C6-O6"])].values.tolist()
    c2_n2 = df["C2-N2"][~ np.isnan(df["C2-N2"])].values.tolist()
    #if res = A
    c6_n6 = df["C6-N6"][~ np.isnan(df["C6-N6"])].values.tolist()
    #if res = C ou U
    c1p_n1 = df["C1'-N1"][~ np.isnan(df["C1'-N1"])].values.tolist()
    n1_c6 = df["N1-C6"][~ np.isnan(df["N1-C6"])].values.tolist()
    c6_c5 = df["C6-C5"][~ np.isnan(df["C6-C5"])].values.tolist()
    c5_c4 = df["C5-C4"][~ np.isnan(df["C5-C4"])].values.tolist()
    c4_n3 = df["C4-N3"][~ np.isnan(df["C4-N3"])].values.tolist()
    n3_c2 = df["N3-C2"][~ np.isnan(df["N3-C2"])].values.tolist()
    c2_n1 = df["C2-N1"][~ np.isnan(df["C2-N1"])].values.tolist()
    c2_o2 = df["C2-O2"][~ np.isnan(df["C2-O2"])].values.tolist()
    #if res =C
    c4_n4 = df["C4-N4"][~ np.isnan(df["C4-N4"])].values.tolist()
    #if res=U
    c4_o4 = df["C4-O4"][~ np.isnan(df["C4-O4"])].values.tolist()

    os.makedirs(runDir+"/results/figures/GMM/all-atoms/distances/commun/", exist_ok=True)
    os.chdir(runDir+"/results/figures/GMM/all-atoms/distances/commun/")

    # draw figures for atoms common to all nucleotides
    GMM_histo(last_o3p_p, "O3'-P", scan)
    GMM_histo(p_op1, "P-OP1", scan)
    GMM_histo(p_op2, "P-OP2", scan)
    GMM_histo(p_o5p, "P-O5'", scan)
    GMM_histo(o5p_c5p, "O5'-C5'", scan)
    GMM_histo(c5p_c4p, "C5'-C4'", scan)
    GMM_histo(c4p_o4p, "C4'-O4'", scan)
    GMM_histo(c4p_c3p, "C4'-C3'", scan)
    GMM_histo(c3p_o3p, "C3'-O3'", scan)
    GMM_histo(o4p_c1p, "O4'-C1'", scan)
    GMM_histo(c1p_c2p, "C1'-C2'", scan)
    GMM_histo(c2p_c3p, "C2'-C3'", scan)
    GMM_histo(c2p_o2p, "C2'-O2'", scan)
    GMM_histo(last_o3p_p, "O3'-P", scan, toric=False, hist=False, col='saddlebrown')
    GMM_histo(p_op1, "P-OP1", scan, toric=False, hist=False, col='gold')
    GMM_histo(p_op2, "P-OP2", scan, toric=False, hist=False, col='lightseagreen')
    GMM_histo(p_o5p, "P-O5'", scan, toric=False, hist=False, col='darkturquoise')
    GMM_histo(o5p_c5p, "O5'-C5'", scan, toric=False, hist=False, col='darkkhaki')
    GMM_histo(c5p_c4p, "C5'-C4'", scan, toric=False, hist=False, col='indigo')
    GMM_histo(c4p_o4p, "C4'-O4'", scan, toric=False, hist=False, col='maroon')
    GMM_histo(c4p_c3p, "C4'-C3'", scan, toric=False, hist=False, col='burlywood')
    GMM_histo(c3p_o3p, "C3'-O3'", scan, toric=False, hist=False, col='steelblue')
    GMM_histo(o4p_c1p, "O4'-C1'", scan, toric=False, hist=False, col='tomato')
    GMM_histo(c1p_c2p, "C1'-C2'", scan, toric=False, hist=False, col='darkolivegreen')
    GMM_histo(c2p_c3p, "C2'-C3'", scan, toric=False, hist=False, col='orchid')
    GMM_histo(c2p_o2p, "C2'-O2'", scan, toric=False, hist=False, col='deeppink')
    axes = plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angströms)")
    plt.title("GMM of distances between common atoms ")
    plt.savefig(runDir + "/results/figures/GMM/all-atoms/distances/commun/" + "GMM_distances_common_atoms.png")
    plt.close()


    # purines
    os.makedirs(runDir+"/results/figures/GMM/all-atoms/distances/purines/", exist_ok=True)
    os.chdir(runDir+"/results/figures/GMM/all-atoms/distances/purines/")
    GMM_histo(c1p_n9, "C1'-N9", scan)
    GMM_histo(n9_c8, "N9-C8", scan)
    GMM_histo(c8_n7, "C8-N7", scan)
    GMM_histo(n7_c5, "N7-C5", scan)
    GMM_histo(c5_c6, "C5-C6", scan)
    GMM_histo(c6_o6, "C6-O6", scan)
    GMM_histo(c6_n6, "C6-N6", scan)
    GMM_histo(c6_n1, "C6-N1", scan)
    GMM_histo(n1_c2, "N1-C2", scan)
    GMM_histo(c2_n2, "C2-N2", scan)
    GMM_histo(c2_n3, "C2-N3", scan)
    GMM_histo(n3_c4, "N3-C4", scan)
    GMM_histo(c4_n9, "C4-N9", scan)
    GMM_histo(c4_c5, "C4-C5", scan)
    GMM_histo(c1p_n9, "C1'-N9", scan, hist=False, col='lightcoral')
    GMM_histo(n9_c8, "N9-C8", scan, hist=False, col='gold')
    GMM_histo(c8_n7, "C8-N7", scan, hist=False, col='lightseagreen')
    GMM_histo(n7_c5, "N7-C5", scan, hist=False, col='saddlebrown')
    GMM_histo(c5_c6, "C5-C6", scan, hist=False, col='darkturquoise')
    GMM_histo(c6_o6, "C6-O6", scan, hist=False, col='darkkhaki')
    GMM_histo(c6_n6, "C6-N6", scan, hist=False, col='indigo')
    GMM_histo(c6_n1, "C6-N1", scan, hist=False, col='maroon')
    GMM_histo(n1_c2, "N1-C2", scan, hist=False, col='burlywood')
    GMM_histo(c2_n2, "C2-N2", scan, hist=False, col='steelblue')
    GMM_histo(c2_n3, "C2-N3", scan, hist=False, col='tomato')
    GMM_histo(n3_c4, "N3-C4", scan, hist=False, col='darkolivegreen')
    GMM_histo(c4_n9, "C4-N9", scan, hist=False, col='orchid')
    GMM_histo(c4_c5, "C4-C5", scan, hist=False, col='deeppink')
    axes = plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angströms)")
    plt.title("GMM of distances between atoms of the purine cycles", fontsize=10)
    plt.savefig(runDir+ "/results/figures/GMM/all-atoms/distances/purines/" + "GMM_distances_purine_cycles.png")
    plt.close()

    # pyrimidines
    os.makedirs(runDir+"/results/figures/GMM/all-atoms/distances/pyrimidines/", exist_ok=True)
    os.chdir(runDir+"/results/figures/GMM/all-atoms/distances/pyrimidines/")
    GMM_histo(c1p_n1, "C1'-N1", scan)
    GMM_histo(n1_c6, "N1-C6", scan)
    GMM_histo(c6_c5, "C6-C5", scan)
    GMM_histo(c5_c4, "C5-C4", scan)
    GMM_histo(c4_n3, "C4-N3", scan)
    GMM_histo(n3_c2, "N3-C2", scan)
    GMM_histo(c2_o2, "C2-O2", scan)
    GMM_histo(c2_n1, "C2-N1", scan)
    GMM_histo(c4_n4, "C4-N4", scan)
    GMM_histo(c4_o4, "C4-O4", scan)
    GMM_histo(c1p_n1, "C1'-N1", scan, hist=False, col='lightcoral')
    GMM_histo(n1_c6, "N1-C6", scan, hist=False, col='gold')
    GMM_histo(c6_c5, "C6-C5", scan, hist=False, col='lightseagreen')
    GMM_histo(c5_c4, "C5-C4", scan, hist=False, col='deeppink')
    GMM_histo(c4_n3, "C4-N3", scan, hist=False, col='red')
    GMM_histo(n3_c2, "N3-C2", scan, hist=False, col='lime')
    GMM_histo(c2_o2, "C2-O2", scan, hist=False, col='indigo')
    GMM_histo(c2_n1, "C2-N1", scan, hist=False, col='maroon')
    GMM_histo(c4_n4, "C4-N4", scan, hist=False, col='burlywood')
    GMM_histo(c4_o4, "C4-O4", scan, hist=False, col='steelblue')
    axes = plt.gca()
    #axes.set_xlim(1, 2)
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angströms")
    plt.title("GMM of distances between atoms of the pyrimidine cycles", fontsize=10)
    plt.savefig(runDir + "/results/figures/GMM/all-atoms/distances/pyrimidines/" + "GMM_distances_pyrimidine_cycles.png")
    plt.close()

    os.chdir(runDir)
    setproctitle("GMM (all atoms, distances) finished")

@trace_unhandled_exceptions
def gmm_aa_torsions(scan, res):
    """
    Separates the torsion angle measurements by angle type and plots the figures representing the data
    """

    setproctitle("GMM (all atoms, torsions)")

    # we create lists to store the values ​​of each angle
    alpha = []
    beta = []
    gamma = []
    delta = []
    epsilon = []
    zeta = []
    chi = []
    angles_deg = retrieve_angles(runDir + "/results/RNANet.db", res)

    # we remove the null values
    alpha = angles_deg.alpha.values
    beta = angles_deg.beta.values
    gamma = angles_deg.gamma.values
    delta = angles_deg.delta.values
    epsilon = angles_deg.epsilon.values
    zeta = angles_deg.zeta.values
    chi = angles_deg.chi.values
    alpha = alpha[~np.isnan(alpha)]
    beta = beta[~np.isnan(beta)]
    gamma = gamma[~np.isnan(gamma)]
    delta = delta[~np.isnan(delta)]
    epsilon = epsilon[~np.isnan(epsilon)]
    zeta = zeta[~np.isnan(zeta)]
    chi  = chi[~np.isnan(chi)]

    os.makedirs(runDir + "/results/figures/GMM/all-atoms/torsions/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/all-atoms/torsions/")

    """
    We plot the GMMs with histogram for each angle
    We create the corresponding json with the means and standard deviations of each Gaussian
    We draw the figure grouping the GMMs of all angles without histogram to compare them with each other
    """

    GMM_histo(alpha, "Alpha", scan, toric=True)
    GMM_histo(beta, "Beta", scan, toric=True)
    GMM_histo(gamma, "Gamma", scan, toric=True)
    GMM_histo(delta, "Delta", scan, toric=True)
    GMM_histo(epsilon, "Epsilon", scan, toric=True)
    GMM_histo(zeta, "Zeta", scan, toric=True)
    GMM_histo(chi, "Xhi", scan, toric=True)
    GMM_histo(alpha, "Alpha", scan, toric=True, hist=False, col='red')
    GMM_histo(beta, "Beta", scan, toric=True, hist=False, col='firebrick')
    GMM_histo(gamma, "Gamma", scan, toric=True, hist=False, col='limegreen')
    GMM_histo(delta, "Delta", scan, toric=True, hist=False, col='darkslateblue')
    GMM_histo(epsilon, "Epsilon", scan, toric=True, hist=False, col='goldenrod')
    GMM_histo(zeta, "Zeta", scan, toric=True, hist=False, col='teal')
    GMM_histo(chi, "Xhi", scan, toric=True, hist=False, col='hotpink')
    plt.xlabel("Angle (Degrees)")
    plt.title("GMM of torsion angles")
    plt.savefig("GMM_torsions.png")
    plt.close()

    os.chdir(runDir)
    setproctitle("GMM (all atoms, torsions) finished")

@trace_unhandled_exceptions
def gmm_pyle(scan, res):

    setproctitle("GMM (Pyle model)")

    # Distances
    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/Pyle/distances/distances_pyle.csv"))  

    p_c1p = df["C1'-P"][~ np.isnan(df["C1'-P"])].values.tolist()
    c1p_p = df["P-C1'"][~ np.isnan(df["P-C1'"])].values.tolist()
    p_c4p = df["C4'-P"][~ np.isnan(df["C4'-P"])].values.tolist()
    c4p_p = df["P-C4'"][~ np.isnan(df["P-C4'"])].values.tolist()

    os.makedirs(runDir + "/results/figures/GMM/Pyle/distances/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/Pyle/distances/")

    GMM_histo(p_c1p, "P-C1'", scan)
    GMM_histo(c1p_p, "C1'-P", scan)
    GMM_histo(p_c4p, "P-C4'", scan)
    GMM_histo(c4p_p, "C4'-P", scan)
    GMM_histo(p_c4p, "P-C4'", scan, toric=False, hist=False, col='gold')
    GMM_histo(c4p_p, "C4'-P", scan, toric=False, hist=False, col='indigo')
    GMM_histo(p_c1p, "P-C1'", scan, toric=False, hist=False, col='firebrick')
    GMM_histo(c1p_p, "C1'-P", scan, toric=False, hist=False, col='seagreen')
    plt.xlabel("Distance (Angströms)")
    plt.title("GMM of distances (Pyle model)")
    plt.savefig("GMM_distances_pyle_model.png")
    plt.close()

    # Flat Angles
    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/Pyle/angles/flat_angles_pyle.csv"))  

    p_c1p_psuiv = list(df["P-C1'-P°"][~ np.isnan(df["P-C1'-P°"])])
    c1p_psuiv_c1psuiv = list(df["C1'-P°-C1'°"][~ np.isnan(df["C1'-P°-C1'°"])])

    os.makedirs(runDir + "/results/figures/GMM/Pyle/angles/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/Pyle/angles/")

    GMM_histo(p_c1p_psuiv, "P-C1'-P°", scan, toric=True)
    GMM_histo(c1p_psuiv_c1psuiv, "C1'-P°-C1'°", scan, toric=True)
    GMM_histo(p_c1p_psuiv, "P-C1'-P°", scan, toric=True, hist=False, col='firebrick')
    GMM_histo(c1p_psuiv_c1psuiv, "C1'-P°-C1'°", scan, toric=True, hist=False, col='seagreen')
    plt.xlabel("Angle (Degrees)")
    plt.title("GMM of flat angles (Pyle model)")
    plt.savefig("GMM_flat_angles_pyle_model.png")
    plt.close()

    # Torsion angles    
    eta=[]
    theta=[]
    eta_prime=[]
    theta_prime=[]
    eta_base=[]
    theta_base=[]

    angles_deg = retrieve_eta_theta(runDir + "/results/RNANet.db", res)

    eta = angles_deg.eta.values
    theta = angles_deg.theta.values
    eta_prime = angles_deg.eta_prime.values
    theta_prime = angles_deg.theta_prime.values
    eta_base = angles_deg.eta_base.values
    theta_base = angles_deg.theta_base.values
    eta = eta[~np.isnan(eta)]
    theta = theta[~np.isnan(theta)]
    eta_prime = eta_prime[~np.isnan(eta_prime)]
    theta_prime = theta_prime[~np.isnan(theta_prime)]
    eta_base = eta_base[~np.isnan(eta_base)]
    theta_base = theta_base[~np.isnan(theta_base)]

    os.makedirs(runDir + "/results/figures/GMM/Pyle/pseudotorsions/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/Pyle/pseudotorsions/")

    GMM_histo(eta, "Eta", scan, toric=True)
    GMM_histo(theta, "Theta", scan, toric=True)
    GMM_histo(eta_prime, "Eta'", scan, toric=True)
    GMM_histo(theta_prime, "Theta'", scan, toric=True)
    GMM_histo(eta_base, "Eta''", scan, toric=True)
    GMM_histo(theta_base, "Theta''", scan, toric=True)
    GMM_histo(eta, "Eta", scan, toric=True, hist=False, col='mediumaquamarine')
    GMM_histo(theta, "Theta", scan, toric=True, hist=False, col='darkorchid')
    GMM_histo(eta_prime, "Eta'", scan, toric=True, hist=False, col='cyan')
    GMM_histo(theta_prime, "Theta'", scan, toric=True, hist=False, col='crimson')
    GMM_histo(eta_base, "Eta''", scan, toric=True, hist=False, col='royalblue')
    GMM_histo(theta_base, "Theta''", scan, toric=True, hist=False, col='palevioletred')
    plt.xlabel("Angle (Degrees)")
    plt.title("GMM of pseudo-torsion angles (Pyle Model)")
    plt.savefig("GMM_pseudotorsion_angles_pyle_model.png")
    plt.close()
    
    os.chdir(runDir)
    setproctitle("GMM (Pyle model) finished")   

@trace_unhandled_exceptions
def gmm_hrna(scan):
    """
    Draw the figures representing the data on the measurements between atoms of the HiRE-RNA model
    """

    setproctitle("GMM (HiRE-RNA)")

    # Distances
    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/HiRE-RNA/distances/distances_HiRERNA.csv"))  

    last_c4p_p = list(df["C4'-P"][~ np.isnan(df["C4'-P"])])
    p_o5p = list(df["P-O5'"][~ np.isnan(df["P-O5'"])])
    o5p_c5p = list(df["O5'-C5'"][~ np.isnan(df["O5'-C5'"])])
    c5p_c4p = list(df["C5'-C4'"][~ np.isnan(df["C5'-C4'"])])
    c4p_c1p = list(df["C4'-C1'"][~ np.isnan(df["C4'-C1'"])])
    c1p_b1 = list(df["C1'-B1"][~ np.isnan(df["C1'-B1"])])
    b1_b2 = list(df["B1-B2"][~ np.isnan(df["B1-B2"])])

    os.makedirs(runDir + "/results/figures/GMM/HiRE-RNA/distances/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/HiRE-RNA/distances/")

    GMM_histo(o5p_c5p, "O5'-C5'", scan)
    GMM_histo(b1_b2, "B1-B2", scan)
    GMM_histo(c1p_b1, "C1'-B1", scan)
    GMM_histo(c5p_c4p, "C5'-C4'", scan)
    GMM_histo(c4p_c1p, "C4'-C1'", scan)
    GMM_histo(p_o5p, "P-O5'", scan)
    GMM_histo(last_c4p_p, "C4'-P", scan)
    
    GMM_histo(o5p_c5p, "O5'-C5'", scan, toric=False, hist=False, col='lightcoral')
    GMM_histo(b1_b2, "B1-B2", scan, toric=False, hist=False, col='limegreen')
    GMM_histo(c1p_b1, "C1'-B1", scan, toric=False, hist=False, col='tomato')
    GMM_histo(c5p_c4p, "C5'-C4'", scan, toric=False, hist=False, col='aquamarine')
    GMM_histo(c4p_c1p, "C4'-C1'", scan, toric=False, hist=False, col='goldenrod')
    GMM_histo(p_o5p, "P-O5'", scan, toric=False, hist=False, col='darkcyan')
    GMM_histo(last_c4p_p, "C4'-P", scan, toric=False, hist=False, col='deeppink')
    axes = plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Distance (Angströms)")
    plt.title("GMM of distances between HiRE-RNA beads")
    plt.savefig(runDir + "/results/figures/GMM/HiRE-RNA/distances/GMM_distances_HiRE_RNA.png")
    plt.close()

    # Angles
    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/HiRE-RNA/angles/angles_HiRERNA.csv"))  

    lastc4p_p_o5p = list(df["C4'-P-O5'"][~ np.isnan(df["C4'-P-O5'"])])
    lastc1p_lastc4p_p = list(df["C1'-C4'-P"][~ np.isnan(df["C1'-C4'-P"])])
    lastc5p_lastc4p_p = list(df["C5'-C4'-P"][~ np.isnan(df["C5'-C4'-P"])])
    p_o5p_c5p = list(df["P-O5'-C5'"][~ np.isnan(df["P-O5'-C5'"])])
    o5p_c5p_c4p = list(df["O5'-C5'-C4'"][~ np.isnan(df["O5'-C5'-C4'"])])
    c5p_c4p_c1p = list(df["C5'-C4'-C1'"][~ np.isnan(df["C5'-C4'-C1'"])])
    c4p_c1p_b1 = list(df["C4'-C1'-B1"][~ np.isnan(df["C4'-C1'-B1"])])
    c1p_b1_b2 = list(df["C1'-B1-B2"][~ np.isnan(df["C1'-B1-B2"])])

    os.makedirs(runDir + "/results/figures/GMM/HiRE-RNA/angles/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/HiRE-RNA/angles/")

    GMM_histo(lastc4p_p_o5p, "C4'-P-O5'", scan, toric=True)
    GMM_histo(lastc1p_lastc4p_p, "C1'-C4'-P", scan, toric=True)
    GMM_histo(lastc5p_lastc4p_p, "C5'-C4'-P", scan, toric=True)
    GMM_histo(p_o5p_c5p, "P-O5'-C5'", scan, toric=True)
    GMM_histo(o5p_c5p_c4p, "O5'-C5'-C4'", scan, toric=True)
    GMM_histo(c5p_c4p_c1p, "C5'-C4'-C1'", scan, toric=True)
    GMM_histo(c4p_c1p_b1, "C4'-C1'-B1", scan, toric=True)
    GMM_histo(c1p_b1_b2, "C1'-B1-B2", scan, toric=True)
    
    GMM_histo(lastc4p_p_o5p, "C4'-P-O5'", scan, toric=True, hist=False, col='lightcoral')
    GMM_histo(lastc1p_lastc4p_p, "C1'-C4'-P", scan, toric=True, hist=False, col='limegreen')
    GMM_histo(lastc5p_lastc4p_p, "C5'-C4'-P", scan, toric=True, hist=False, col='tomato')
    GMM_histo(p_o5p_c5p, "P-O5'-C5'", scan, toric=True, hist=False, col='aquamarine')
    GMM_histo(o5p_c5p_c4p, "O5'-C5'-C4'", scan, toric=True, hist=False, col='goldenrod')
    GMM_histo(c5p_c4p_c1p, "C5'-C4'-C1'", scan, toric=True, hist=False, col='darkcyan')
    GMM_histo(c4p_c1p_b1, "C4'-C1'-B1", scan, toric=True, hist=False, col='deeppink')
    GMM_histo(c1p_b1_b2, "C1'-B1-B2", scan, toric=True, hist=False, col='indigo')
    axes = plt.gca()
    axes.set_ylim(0, 100)
    plt.xlabel("Angle (Degres)")
    plt.title("GMM of angles between HiRE-RNA beads")
    plt.savefig(runDir + "/results/figures/GMM/HiRE-RNA/angles/GMM_angles_HiRE_RNA.png")
    plt.close()

    # Torsions    
    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/HiRE-RNA/torsions/torsions_HiRERNA.csv"))  

    p_o5_c5_c4 = list(df["P-O5'-C5'-C4'"][~ np.isnan(df["P-O5'-C5'-C4'"])])
    o5_c5_c4_c1 = list(df["O5'-C5'-C4'-C1'"][~ np.isnan(df["O5'-C5'-C4'-C1'"])])
    c5_c4_c1_b1 = list(df["C5'-C4'-C1'-B1"][~ np.isnan(df["C5'-C4'-C1'-B1"])])
    c4_c1_b1_b2 = list(df["C4'-C1'-B1-B2"][~ np.isnan(df["C4'-C1'-B1-B2"])])
    o5_c5_c4_psuiv = list(df["O5'-C5'-C4'-P°"][~ np.isnan(df["O5'-C5'-C4'-P°"])])
    c5_c4_psuiv_o5suiv = list(df["C5'-C4'-P°-O5'°"][~ np.isnan(df["C5'-C4'-P°-O5'°"])])
    c4_psuiv_o5suiv_c5suiv = list(df["C4'-P°-O5'°-C5'°"][~ np.isnan(df["C4'-P°-O5'°-C5'°"])])
    c1_c4_psuiv_o5suiv = list(df["C1'-C4'-P°-O5'°"][~ np.isnan(df["C1'-C4'-P°-O5'°"])])

    os.makedirs(runDir + "/results/figures/GMM/HiRE-RNA/torsions/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/HiRE-RNA/torsions/")

    GMM_histo(p_o5_c5_c4, "P-O5'-C5'-C4'", scan, toric=True)
    GMM_histo(o5_c5_c4_c1, "O5'-C5'-C4'-C1'", scan, toric=True)
    GMM_histo(c5_c4_c1_b1, "C5'-C4'-C1'-B1", scan, toric=True)
    GMM_histo(c4_c1_b1_b2, "C4'-C1'-B1-B2", scan, toric=True)
    GMM_histo(o5_c5_c4_psuiv, "O5'-C5'-C4'-P°", scan, toric=True)
    GMM_histo(c5_c4_psuiv_o5suiv, "C5'-C4'-P°-O5'°", scan, toric=True)
    GMM_histo(c4_psuiv_o5suiv_c5suiv, "C4'-P°-O5'°-C5'°", scan, toric=True)
    GMM_histo(c1_c4_psuiv_o5suiv, "C1'-C4'-P°-O5'°", scan, toric=True)

    GMM_histo(p_o5_c5_c4, "P-O5'-C5'-C4'", scan, toric=True, hist=False, col='darkred')
    GMM_histo(o5_c5_c4_c1, "O5'-C5'-C4'-C1'", scan, toric=True, hist=False, col='chocolate')
    GMM_histo(c5_c4_c1_b1, "C5'-C4'-C1'-B1", scan, toric=True, hist=False, col='mediumvioletred')
    GMM_histo(c4_c1_b1_b2, "C4'-C1'-B1-B2", scan, toric=True, hist=False, col='cadetblue')
    GMM_histo(o5_c5_c4_psuiv, "O5'-C5'-C4'-P°", scan, toric=True, hist=False, col='darkkhaki')
    GMM_histo(c5_c4_psuiv_o5suiv, "C5'-C4'-P°-O5'°", scan, toric=True, hist=False, col='springgreen')
    GMM_histo(c4_psuiv_o5suiv_c5suiv, "C4'-P°-O5'°-C5'°", scan, toric=True, hist=False, col='indigo')
    GMM_histo(c1_c4_psuiv_o5suiv, "C1'-C4'-P°-O5'°", scan, toric=True, hist=False, col='gold')
    plt.xlabel("Angle (Degrees)")
    plt.title("GMM of torsion angles between HiRE-RNA beads")
    plt.savefig("GMM_torsions_HiRE_RNA.png")
    plt.close()

    os.chdir(runDir)
    setproctitle("GMM (HiRE-RNA) finished")

@trace_unhandled_exceptions
def gmm_hrna_basepairs(scan):
    """
    Measures parameters of all kinds of non-canonical basepairs for the HiRE-RNA model.
    Please see Cragnolini & al 2015 to understand them.
    """

    setproctitle("GMM (HiRE-RNA basepairs)")

    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/HiRE-RNA/basepairs/basepairs_HiRERNA.csv"))

    lw = ["cWW", "tWW", "cWH", "tWH", "cHW", "tHW", "cWS", "tWS", "cSW", "tSW", "cHH", "tHH", "cSH", "tSH", "cHS", "tHS", "cSS", "tSS"]

    os.makedirs(runDir + "/results/figures/GMM/HiRE-RNA/basepairs/", exist_ok=True)
    os.chdir(runDir + "/results/figures/GMM/HiRE-RNA/basepairs/")

    for lw_type in lw:
        data = df[df['type_LW'] == lw_type ]
        if len(data):
            for b1 in ['A','C','G','U']:
                for b2 in ['A','C','G','U']:
                    thisbases = data[(data.nt1_res == b1)&(data.nt2_res == b2)]
                    if len(thisbases):
                        gmm_hrna_basepair_type(lw_type, b1+b2, thisbases, scan)

    os.chdir(runDir)
    setproctitle(f"GMM (HiRE-RNA basepairs) finished")

@trace_unhandled_exceptions
def gmm_hrna_basepair_type(type_LW, ntpair, data, scan):
    """
    function to plot the statistical figures you want
    By type of pairing:
    Superposition of GMMs of plane angles
    Superposition of the histogram and the GMM of the distances
    all in the same window
    """

    setproctitle(f"GMM (HiRE-RNA {type_LW} basepairs)")

    figure = plt.figure(figsize = (10, 10))
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0, hspace = 0.5)
   
    plt.subplot(2, 1, 1)
    GMM_histo(data["211_angle"], f"{type_LW}_{ntpair}_C1'-B1-B1pair", scan, toric=True, hist=False, col='cyan' )
    GMM_histo(data["112_angle"], f"{type_LW}_{ntpair}_B1-B1pair-C1'pair", scan, toric=True, hist=False, col='magenta')
    GMM_histo(data["3211_torsion"], f"{type_LW}_{ntpair}_C4'-C1'-B1-B1pair", scan, toric=True, hist=False, col='black' )
    GMM_histo(data["1123_torsion"], f"{type_LW}_{ntpair}_B1-B1pair-C1'pair-C4'pair", scan, toric=True, hist=False, col='maroon')
    GMM_histo(data["alpha1"], f"{type_LW}_{ntpair}_alpha_1", scan, toric=True, hist=False, col="yellow")
    GMM_histo(data["alpha2"], f"{type_LW}_{ntpair}_alpha_2", scan, toric=True, hist=False, col='olive')
    plt.xlabel("Angle (degree)")
    plt.title(f"GMM of plane angles for {type_LW} {ntpair} basepairs", fontsize=10)

    plt.subplot(2, 1, 2)
    GMM_histo(data["Distance"], f"{type_LW}_{ntpair}_tips_distance", scan, toric=False, hist=False, col="cyan")
    GMM_histo(data["dB1"], f"{type_LW}_{ntpair}_dB1", scan, toric=False, hist=False, col="tomato")
    GMM_histo(data["dB2"], f"{type_LW}_{ntpair}_dB2", scan, toric=False, hist=False, col="goldenrod")
    plt.xlabel("Distance (Angströms)")
    plt.title(f"GMM of distances for {type_LW} {ntpair} basepairs", fontsize=10)
    
    plt.savefig(f"{type_LW}_{ntpair}_basepairs.png" )
    plt.close()
    setproctitle(f"GMM (HiRE-RNA {type_LW} {ntpair} basepairs) finished")

@trace_unhandled_exceptions
def merge_jsons():
    """
    Reads the tons of JSON files produced by the geometric analyses, and compiles them into fewer files.
    It is simple concatenation of the JSONs.
    The original files are then deleted.
    """

    # All atom distances
    bonds = ["O3'-P", "P-OP1", "P-OP2", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-O4'", "C4'-C3'", "O4'-C1'", "C1'-C2'", "C2'-O2'", "C2'-C3'", "C3'-O3'", "C1'-N9",
             "N9-C8", "C8-N7", "N7-C5", "C5-C6", "C6-O6", "C6-N6", "C6-N1", "N1-C2", "C2-N2", "C2-N3", "N3-C4", "C4-N9", "C4-C5", 
             "C1'-N1", "N1-C6", "C6-C5", "C5-C4", "C4-N3", "N3-C2", "C2-O2", "C2-N1", "C4-N4", "C4-O4"]
    bonds = [ runDir + "/results/geometry/json/" + x + ".json" for x in bonds ]
    concat_jsons(bonds, runDir + "/results/geometry/json/all_atom_distances.json")
    

    # All atom torsions
    torsions = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Xhi", "Zeta"]
    torsions = [ runDir + "/results/geometry/json/" + x + ".json" for x in torsions ]
    concat_jsons(torsions, runDir + "/results/geometry/json/all_atom_torsions.json")
 
    # HiRE-RNA distances
    hrnabonds = [r"P-O5'", r"O5'-C5'", r"C5'-C4'", r"C4'-C1'", r"C1'-B1", r"B1-B2", r"C4'-P"]
    hrnabonds = [ runDir + "/results/geometry/json/" + x + ".json" for x in hrnabonds ]
    concat_jsons(hrnabonds, runDir + "/results/geometry/json/hirerna_distances.json")

    # HiRE-RNA angles
    hrnaangles = [r"P-O5'-C5'", r"O5'-C5'-C4'", r"C5'-C4'-C1'", r"C4'-C1'-B1", r"C1'-B1-B2", r"C4'-P-O5'", r"C5'-C4'-P", r"C1'-C4'-P"]
    hrnaangles = [ runDir + "/results/geometry/json/" + x + ".json" for x in hrnaangles ]
    concat_jsons(hrnaangles, runDir + "/results/geometry/json/hirerna_angles.json")

    # HiRE-RNA torsions
    hrnators = [r"P-O5'-C5'-C4'", r"O5'-C5'-C4'-C1'", r"C5'-C4'-C1'-B1", r"C4'-C1'-B1-B2", r"C4'-P°-O5'°-C5'°", r"C5'-C4'-P°-O5'°", r"C1'-C4'-P°-O5'°", r"O5'-C5'-C4'-P°"]
    hrnators = [ runDir + "/results/geometry/json/" + x + ".json" for x in hrnators ]
    concat_jsons(hrnators, runDir + "/results/geometry/json/hirerna_torsions.json")

    # HiRE-RNA basepairs
    for nt1 in ['A', 'C', 'G', 'U']:
        for nt2 in ['A', 'C', 'G', 'U']:
            bps = glob.glob(runDir + f"/results/geometry/json/*{nt1}{nt2}*.json")
            concat_jsons(bps, runDir + f"/results/geometry/json/hirerna_{nt1}{nt2}_basepairs.json")

    # Delete previous files
    for f in bonds + torsions + hrnabonds + hrnaangles + hrnators:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    for f in glob.glob(runDir + "/results/geometry/json/t*.json"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    for f in glob.glob(runDir + "/results/geometry/json/c*.json"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    for f in glob.glob(runDir + "/results/geometry/json/*tips_distance.json"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

@trace_unhandled_exceptions
def concat_worker(bunch):
    """
    Concatenates a bunch of CSV files and returns a Pandas DataFrame.
    bunch: List of strings (filepaths to CSV files)

    The function logs concatenations to a global TQDM progress bar.
    The function is expected to be used in parallel.
    """

    global sharedpbar
    global finished

    # initiate the dataframe with the first CSV file
    df_tot = pd.read_csv(bunch.pop(), engine="c")
    with finished.get_lock():
        finished.value += 1

    for f in range(len(bunch)):
        # Read and concatenate a new file
        df = pd.read_csv(bunch.pop(), engine='c')
        df_tot = pd.concat([df_tot, df], ignore_index=True)

        # Update the global progress bar
        with finished.get_lock():
            finished.value += 1
        with sharedpbar.get_lock():
            sharedpbar.n = finished.value
        sharedpbar.refresh()

    return df_tot

@trace_unhandled_exceptions
def concat_dataframes(fpath, outfilename, nworkers):
    """
    Concatenates the CSV files from fpath folder into a DataFrame gathering all.
    The function splits the file list into nworkers concatenation workers, and then merges the nworkers dataframes.
    """
    setproctitle(f"Concatenation of {fpath}")

    # Get the list of files
    flist = os.listdir(fpath)
    random.shuffle(flist)
    flist = [ os.path.abspath(fpath + x) for x in flist ]

    # Define a global progress bar to be shared between workers
    global sharedpbar
    global finished
    sharedpbar = tqdm(total=len(flist), position=0, desc="Preparing "+outfilename, leave=False)
    finished = Value('i', 0)

    # Divide the list into chunks
    start = 0
    end = int(len(flist)/nworkers)+1
    size = end
    chunks = []
    for i in range(nworkers):
        if i == nworkers-1:
            chunks.append(flist[start:])
        else:
            chunks.append(flist[start:end])
        start, end = end, end+size

    # Run parallel concatenations
    p = Pool(initializer=init_with_tqdm, initargs=(tqdm.get_lock(),), processes=nworkers)
    results = p.map(concat_worker, chunks, chunksize=1)
    p.close()
    p.join()
    sharedpbar.close()

    # Concatenate the results and save
    df_tot = pd.concat(results, ignore_index=True)
    df_tot.to_csv(fpath + outfilename)

@trace_unhandled_exceptions
def concat_jsons(flist, outfilename):
    """
    Reads JSON files computed by the geometry jobs and merge them into a smaller
    number of files
    """
    
    result = []
    for f in flist:
        # if not os.path.isfile(f):
        #     continue:
        with open(f, "rb") as infile:
            result.append(json.load(infile))

    # write the files
    with open(outfilename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    print("This file is not supposed to be run directly. Run statistics.py instead.")