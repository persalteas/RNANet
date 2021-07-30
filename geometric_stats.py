#!/usr/bin/python3

# RNANet statistics
# Developed by Aglaé Tabot & Louis Becquey, 2021 

# This file computes additional geometric measures over the produced dataset,
# and estimates their distribtuions through Gaussian mixture models.
# THIS FILE IS NOT SUPPOSED TO BE RUN DIRECTLY.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Bio, json, os, random, setproctitle, sqlite3
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.vectors import Vector, calc_angle, calc_dihedral
from multiprocessing import Pool, Value
from pandas.core.common import SettingWithCopyWarning
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# number of modes in the parameter distribution, used to know how many laws to use in the GMM. if you do not want to trust this data,
# you can use the --rescan-nmodes option. GMMs will be trained between 1 and 8 modes and the best model will be kept.
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
    "P-O5'-C5'":2, "O5'-C5'-C4'":1, "C5'-C4'-P":2, "C5'-C4'-C1'":2, "C4'-P-O5'":2, "C4'-CA'-B1":2, "C1'-C4'-P":2, "C1'-B1-B2":2,

    # HiRE-RNA, torsions
    "P-O5'-C5'-C4'":1, "O5'-C5'-C4'-P°":3, "O5'-C5'-C4'-C1'":3, "C5'-C4'-P°-O5'°":3, "C5'-C4'-C1'-B1":2, "C4'-P°-O5'°-C5'°":3, "C4'-C1'-B1-B2":3, "C1'-C4'-P°-O5'°":3,

    # HiRE-RNA, basepairs
    "cWW_AA_C1'-B1-B1pair":1, "cWW_AA_B1-B1pair-C1'pair":1, "cWW_AA_C4'-C1'-B1-B1pair":2, "cWW_AA_B1-B1pair-C1'pair-C4'pair":3, "cWW_AA_alpha_1":1, "cWW_AA_alpha_2":3, "cWW AA dB1":3, "cWW AA dB2":3, 
    "tWW_AA_C1'-B1-B1pair":1, "tWW_AA_B1-B1pair-C1'pair":1, "tWW_AA_C4'-C1'-B1-B1pair":1, "tWW_AA_B1-B1pair-C1'pair-C4'pair":3, "tWW_AA_alpha_1":2, "tWW_AA_alpha_2":1, "tWW AA dB1":1, "tWW AA dB2":2, 
    "cWH_AA_C1'-B1-B1pair":2, "cWH_AA_B1-B1pair-C1'pair":2, "cWH_AA_C4'-C1'-B1-B1pair":2, "cWH_AA_B1-B1pair-C1'pair-C4'pair":2, "cWH_AA_alpha_1":1, "cWH_AA_alpha_2":2, "cWH AA dB1":3, "cWH AA dB2":2, 
    "tWH_AA_C1'-B1-B1pair":1, "tWH_AA_B1-B1pair-C1'pair":3, "tWH_AA_C4'-C1'-B1-B1pair":2, "tWH_AA_B1-B1pair-C1'pair-C4'pair":1, "tWH_AA_alpha_1":1, "tWH_AA_alpha_2":3, "tWH AA dB1":2, "tWH AA dB2":1, 
    "cHW_AA_C1'-B1-B1pair":2, "cHW_AA_B1-B1pair-C1'pair":2, "cHW_AA_C4'-C1'-B1-B1pair":3, "cHW_AA_B1-B1pair-C1'pair-C4'pair":2, "cHW_AA_alpha_1":2, "cHW_AA_alpha_2":2, "cHW AA dB1":3, "cHW AA dB2":2, 
    "tHW_AA_C1'-B1-B1pair":2, "tHW_AA_B1-B1pair-C1'pair":2, "tHW_AA_C4'-C1'-B1-B1pair":2, "tHW_AA_B1-B1pair-C1'pair-C4'pair":2, "tHW_AA_alpha_1":2, "tHW_AA_alpha_2":1, "tHW AA dB1":2, "tHW AA dB2":1, 
    "cWS_AA_C1'-B1-B1pair":2, "cWS_AA_B1-B1pair-C1'pair":2, "cWS_AA_C4'-C1'-B1-B1pair":2, "cWS_AA_B1-B1pair-C1'pair-C4'pair":1, "cWS_AA_alpha_1":2, "cWS_AA_alpha_2":2, "cWS AA dB1":2, "cWS AA dB2":1, 
    "tWS_AA_C1'-B1-B1pair":2, "tWS_AA_B1-B1pair-C1'pair":2, "tWS_AA_C4'-C1'-B1-B1pair":3, "tWS_AA_B1-B1pair-C1'pair-C4'pair":1, "tWS_AA_alpha_1":2, "tWS_AA_alpha_2":2, "tWS AA dB1":2, "tWS AA dB2":3, 
    "cSW_AA_C1'-B1-B1pair":3, "cSW_AA_B1-B1pair-C1'pair":2, "cSW_AA_C4'-C1'-B1-B1pair":1, "cSW_AA_B1-B1pair-C1'pair-C4'pair":2, "cSW_AA_alpha_1":2, "cSW_AA_alpha_2":2, "cSW AA dB1":1, "cSW AA dB2":1, 
    "tSW_AA_C1'-B1-B1pair":3, "tSW_AA_B1-B1pair-C1'pair":3, "tSW_AA_C4'-C1'-B1-B1pair":2, "tSW_AA_B1-B1pair-C1'pair-C4'pair":2, "tSW_AA_alpha_1":2, "tSW_AA_alpha_2":2, "tSW AA dB1":2, "tSW AA dB2":2, 
    "cHH_AA_C1'-B1-B1pair":2, "cHH_AA_B1-B1pair-C1'pair":3, "cHH_AA_C4'-C1'-B1-B1pair":3, "cHH_AA_B1-B1pair-C1'pair-C4'pair":3, "cHH_AA_alpha_1":2, "cHH_AA_alpha_2":3, "cHH AA dB1":3, "cHH AA dB2":1, 
    "tHH_AA_C1'-B1-B1pair":2, "tHH_AA_B1-B1pair-C1'pair":2, "tHH_AA_C4'-C1'-B1-B1pair":3, "tHH_AA_B1-B1pair-C1'pair-C4'pair":1, "tHH_AA_alpha_1":2, "tHH_AA_alpha_2":2, "tHH AA dB1":2, "tHH AA dB2":2, 
    "cSH_AA_C1'-B1-B1pair":2, "cSH_AA_B1-B1pair-C1'pair":1, "cSH_AA_C4'-C1'-B1-B1pair":3, "cSH_AA_B1-B1pair-C1'pair-C4'pair":1, "cSH_AA_alpha_1":2, "cSH_AA_alpha_2":2, "cSH AA dB1":4, "cSH AA dB2":1, 
    "tSH_AA_C1'-B1-B1pair":1, "tSH_AA_B1-B1pair-C1'pair":2, "tSH_AA_C4'-C1'-B1-B1pair":2, "tSH_AA_B1-B1pair-C1'pair-C4'pair":2, "tSH_AA_alpha_1":2, "tSH_AA_alpha_2":3, "tSH AA dB1":2, "tSH AA dB2":2, 
    "cHS_AA_C1'-B1-B1pair":2, "cHS_AA_B1-B1pair-C1'pair":2, "cHS_AA_C4'-C1'-B1-B1pair":1, "cHS_AA_B1-B1pair-C1'pair-C4'pair":1, "cHS_AA_alpha_1":2, "cHS_AA_alpha_2":2, "cHS AA dB1":1, "cHS AA dB2":4, 
    "tHS_AA_C1'-B1-B1pair":2, "tHS_AA_B1-B1pair-C1'pair":2, "tHS_AA_C4'-C1'-B1-B1pair":1, "tHS_AA_B1-B1pair-C1'pair-C4'pair":1, "tHS_AA_alpha_1":2, "tHS_AA_alpha_2":1, "tHS AA dB1":2, "tHS AA dB2":1, 
    "cSS_AA_C1'-B1-B1pair":3, "cSS_AA_B1-B1pair-C1'pair":3, "cSS_AA_C4'-C1'-B1-B1pair":2, "cSS_AA_B1-B1pair-C1'pair-C4'pair":2, "cSS_AA_alpha_1":3, "cSS_AA_alpha_2":3, "cSS AA dB1":3, "cSS AA dB2":5, 
    "tSS_AA_C1'-B1-B1pair":1, "tSS_AA_B1-B1pair-C1'pair":1, "tSS_AA_C4'-C1'-B1-B1pair":2, "tSS_AA_B1-B1pair-C1'pair-C4'pair":1, "tSS_AA_alpha_1":3, "tSS_AA_alpha_2":1, "tSS AA dB1":4, "tSS AA dB2":2, 
    "cWW_AC_C1'-B1-B1pair":1, "cWW_AC_B1-B1pair-C1'pair":2, "cWW_AC_C4'-C1'-B1-B1pair":2, "cWW_AC_B1-B1pair-C1'pair-C4'pair":2, "cWW_AC_alpha_1":1, "cWW_AC_alpha_2":2, "cWW AC dB1":3, "cWW AC dB2":3, 
    "tWW_AC_C1'-B1-B1pair":3, "tWW_AC_B1-B1pair-C1'pair":2, "tWW_AC_C4'-C1'-B1-B1pair":2, "tWW_AC_B1-B1pair-C1'pair-C4'pair":3, "tWW_AC_alpha_1":3, "tWW_AC_alpha_2":2, "tWW AC dB1":4, "tWW AC dB2":3, 
    "cWH_AC_C1'-B1-B1pair":2, "cWH_AC_B1-B1pair-C1'pair":2, "cWH_AC_C4'-C1'-B1-B1pair":1, "cWH_AC_B1-B1pair-C1'pair-C4'pair":2, "cWH_AC_alpha_1":2, "cWH_AC_alpha_2":2, "cWH AC dB1":4, "cWH AC dB2":4, 
    "tWH_AC_C1'-B1-B1pair":1, "tWH_AC_B1-B1pair-C1'pair":2, "tWH_AC_C4'-C1'-B1-B1pair":2, "tWH_AC_B1-B1pair-C1'pair-C4'pair":3, "tWH_AC_alpha_1":2, "tWH_AC_alpha_2":2, "tWH AC dB1":3, "tWH AC dB2":3, 
    "cHW_AC_C1'-B1-B1pair":2, "cHW_AC_B1-B1pair-C1'pair":2, "cHW_AC_C4'-C1'-B1-B1pair":3, "cHW_AC_B1-B1pair-C1'pair-C4'pair":2, "cHW_AC_alpha_1":2, "cHW_AC_alpha_2":3, "cHW AC dB1":2, "cHW AC dB2":5, 
    "tHW_AC_C1'-B1-B1pair":2, "tHW_AC_B1-B1pair-C1'pair":3, "tHW_AC_C4'-C1'-B1-B1pair":3, "tHW_AC_B1-B1pair-C1'pair-C4'pair":1, "tHW_AC_alpha_1":2, "tHW_AC_alpha_2":2, "tHW AC dB1":3, "tHW AC dB2":3, 
    "cWS_AC_C1'-B1-B1pair":2, "cWS_AC_B1-B1pair-C1'pair":1, "cWS_AC_C4'-C1'-B1-B1pair":2, "cWS_AC_B1-B1pair-C1'pair-C4'pair":1, "cWS_AC_alpha_1":2, "cWS_AC_alpha_2":1, "cWS AC dB1":1, "cWS AC dB2":1, 
    "tWS_AC_C1'-B1-B1pair":2, "tWS_AC_B1-B1pair-C1'pair":1, "tWS_AC_C4'-C1'-B1-B1pair":2, "tWS_AC_B1-B1pair-C1'pair-C4'pair":2, "tWS_AC_alpha_1":3, "tWS_AC_alpha_2":1, "tWS AC dB1":3, "tWS AC dB2":2, 
    "cSW_AC_C1'-B1-B1pair":2, "cSW_AC_B1-B1pair-C1'pair":2, "cSW_AC_C4'-C1'-B1-B1pair":2, "cSW_AC_B1-B1pair-C1'pair-C4'pair":2, "cSW_AC_alpha_1":3, "cSW_AC_alpha_2":2, "cSW AC dB1":2, "cSW AC dB2":3, 
    "tSW_AC_C1'-B1-B1pair":1, "tSW_AC_B1-B1pair-C1'pair":2, "tSW_AC_C4'-C1'-B1-B1pair":1, "tSW_AC_B1-B1pair-C1'pair-C4'pair":2, "tSW_AC_alpha_1":1, "tSW_AC_alpha_2":2, "tSW AC dB1":2, "tSW AC dB2":3, 
    "cHH_AC_C1'-B1-B1pair":2, "cHH_AC_B1-B1pair-C1'pair":2, "cHH_AC_C4'-C1'-B1-B1pair":1, "cHH_AC_B1-B1pair-C1'pair-C4'pair":1, "cHH_AC_alpha_1":3, "cHH_AC_alpha_2":3, "cHH AC dB1":3, "cHH AC dB2":4, 
    "tHH_AC_C1'-B1-B1pair":1, "tHH_AC_B1-B1pair-C1'pair":2, "tHH_AC_C4'-C1'-B1-B1pair":2, "tHH_AC_B1-B1pair-C1'pair-C4'pair":3, "tHH_AC_alpha_1":2, "tHH_AC_alpha_2":2, "tHH AC dB1":4, "tHH AC dB2":3, 
    "cSH_AC_C1'-B1-B1pair":1, "cSH_AC_B1-B1pair-C1'pair":3, "cSH_AC_C4'-C1'-B1-B1pair":1, "cSH_AC_B1-B1pair-C1'pair-C4'pair":2, "cSH_AC_alpha_1":1, "cSH_AC_alpha_2":1, "cSH AC dB1":2, "cSH AC dB2":6, 
    "tSH_AC_C1'-B1-B1pair":3, "tSH_AC_B1-B1pair-C1'pair":2, "tSH_AC_C4'-C1'-B1-B1pair":1, "tSH_AC_B1-B1pair-C1'pair-C4'pair":2, "tSH_AC_alpha_1":2, "tSH_AC_alpha_2":3, "tSH AC dB1":1, "tSH AC dB2":2, 
    "cHS_AC_C1'-B1-B1pair":1, "cHS_AC_B1-B1pair-C1'pair":1, "cHS_AC_C4'-C1'-B1-B1pair":2, "cHS_AC_B1-B1pair-C1'pair-C4'pair":1, "cHS_AC_alpha_1":1, "cHS_AC_alpha_2":1, "cHS AC dB1":3, "cHS AC dB2":2, 
    "tHS_AC_C1'-B1-B1pair":1, "tHS_AC_B1-B1pair-C1'pair":2, "tHS_AC_C4'-C1'-B1-B1pair":2, "tHS_AC_B1-B1pair-C1'pair-C4'pair":2, "tHS_AC_alpha_1":1, "tHS_AC_alpha_2":1, "tHS AC dB1":1, "tHS AC dB2":1, 
    "cSS_AC_C1'-B1-B1pair":2, "cSS_AC_B1-B1pair-C1'pair":2, "cSS_AC_C4'-C1'-B1-B1pair":1, "cSS_AC_B1-B1pair-C1'pair-C4'pair":1, "cSS_AC_alpha_1":2, "cSS_AC_alpha_2":1, "cSS AC dB1":1, "cSS AC dB2":5, 
    "tSS_AC_C1'-B1-B1pair":2, "tSS_AC_B1-B1pair-C1'pair":2, "tSS_AC_C4'-C1'-B1-B1pair":1, "tSS_AC_B1-B1pair-C1'pair-C4'pair":2, "tSS_AC_alpha_1":2, "tSS_AC_alpha_2":2, "tSS AC dB1":3, "tSS AC dB2":5, 
    "cWW_AG_C1'-B1-B1pair":1, "cWW_AG_B1-B1pair-C1'pair":1, "cWW_AG_C4'-C1'-B1-B1pair":2, "cWW_AG_B1-B1pair-C1'pair-C4'pair":2, "cWW_AG_alpha_1":1, "cWW_AG_alpha_2":1, "cWW AG dB1":1, "cWW AG dB2":1, 
    "tWW_AG_C1'-B1-B1pair":1, "tWW_AG_B1-B1pair-C1'pair":1, "tWW_AG_C4'-C1'-B1-B1pair":2, "tWW_AG_B1-B1pair-C1'pair-C4'pair":2, "tWW_AG_alpha_1":1, "tWW_AG_alpha_2":2, "tWW AG dB1":2, "tWW AG dB2":3, 
    "cWH_AG_C1'-B1-B1pair":1, "cWH_AG_B1-B1pair-C1'pair":1, "cWH_AG_C4'-C1'-B1-B1pair":2, "cWH_AG_B1-B1pair-C1'pair-C4'pair":1, "cWH_AG_alpha_1":3, "cWH_AG_alpha_2":1, "cWH AG dB1":2, "cWH AG dB2":1, 
    "tWH_AG_C1'-B1-B1pair":1, "tWH_AG_B1-B1pair-C1'pair":1, "tWH_AG_C4'-C1'-B1-B1pair":2, "tWH_AG_B1-B1pair-C1'pair-C4'pair":2, "tWH_AG_alpha_1":1, "tWH_AG_alpha_2":1, "tWH AG dB1":2, "tWH AG dB2":1, 
    "cHW_AG_C1'-B1-B1pair":2, "cHW_AG_B1-B1pair-C1'pair":1, "cHW_AG_C4'-C1'-B1-B1pair":1, "cHW_AG_B1-B1pair-C1'pair-C4'pair":1, "cHW_AG_alpha_1":1, "cHW_AG_alpha_2":2, "cHW AG dB1":2, "cHW AG dB2":2, 
    "tHW_AG_C1'-B1-B1pair":2, "tHW_AG_B1-B1pair-C1'pair":2, "tHW_AG_C4'-C1'-B1-B1pair":1, "tHW_AG_B1-B1pair-C1'pair-C4'pair":2, "tHW_AG_alpha_1":2, "tHW_AG_alpha_2":2, "tHW AG dB1":2, "tHW AG dB2":2, 
    "cWS_AG_C1'-B1-B1pair":3, "cWS_AG_B1-B1pair-C1'pair":1, "cWS_AG_C4'-C1'-B1-B1pair":1, "cWS_AG_B1-B1pair-C1'pair-C4'pair":1, "cWS_AG_alpha_1":2, "cWS_AG_alpha_2":2, "cWS AG dB1":2, "cWS AG dB2":1, 
    "tWS_AG_C1'-B1-B1pair":1, "tWS_AG_B1-B1pair-C1'pair":2, "tWS_AG_C4'-C1'-B1-B1pair":2, "tWS_AG_B1-B1pair-C1'pair-C4'pair":1, "tWS_AG_alpha_1":2, "tWS_AG_alpha_2":2, "tWS AG dB1":1, "tWS AG dB2":3, 
    "cSW_AG_C1'-B1-B1pair":1, "cSW_AG_B1-B1pair-C1'pair":2, "cSW_AG_C4'-C1'-B1-B1pair":1, "cSW_AG_B1-B1pair-C1'pair-C4'pair":2, "cSW_AG_alpha_1":1, "cSW_AG_alpha_2":2, "cSW AG dB1":3, "cSW AG dB2":1, 
    "tSW_AG_C1'-B1-B1pair":3, "tSW_AG_B1-B1pair-C1'pair":2, "tSW_AG_C4'-C1'-B1-B1pair":2, "tSW_AG_B1-B1pair-C1'pair-C4'pair":2, "tSW_AG_alpha_1":2, "tSW_AG_alpha_2":2, "tSW AG dB1":3, "tSW AG dB2":3, 
    "cHH_AG_C1'-B1-B1pair":2, "cHH_AG_B1-B1pair-C1'pair":4, "cHH_AG_C4'-C1'-B1-B1pair":3, "cHH_AG_B1-B1pair-C1'pair-C4'pair":2, "cHH_AG_alpha_1":2, "cHH_AG_alpha_2":3, "cHH AG dB1":1, "cHH AG dB2":2, 
    "tHH_AG_C1'-B1-B1pair":3, "tHH_AG_B1-B1pair-C1'pair":3, "tHH_AG_C4'-C1'-B1-B1pair":3, "tHH_AG_B1-B1pair-C1'pair-C4'pair":2, "tHH_AG_alpha_1":3, "tHH_AG_alpha_2":3, "tHH AG dB1":1, "tHH AG dB2":2, 
    "cSH_AG_C1'-B1-B1pair":2, "cSH_AG_B1-B1pair-C1'pair":2, "cSH_AG_C4'-C1'-B1-B1pair":2, "cSH_AG_B1-B1pair-C1'pair-C4'pair":2, "cSH_AG_alpha_1":3, "cSH_AG_alpha_2":1, "cSH AG dB1":1, "cSH AG dB2":3, 
    "tSH_AG_C1'-B1-B1pair":2, "tSH_AG_B1-B1pair-C1'pair":2, "tSH_AG_C4'-C1'-B1-B1pair":2, "tSH_AG_B1-B1pair-C1'pair-C4'pair":3, "tSH_AG_alpha_1":2, "tSH_AG_alpha_2":4, "tSH AG dB1":3, "tSH AG dB2":2, 
    "cHS_AG_C1'-B1-B1pair":3, "cHS_AG_B1-B1pair-C1'pair":1, "cHS_AG_C4'-C1'-B1-B1pair":3, "cHS_AG_B1-B1pair-C1'pair-C4'pair":1, "cHS_AG_alpha_1":2, "cHS_AG_alpha_2":3, "cHS AG dB1":1, "cHS AG dB2":2, 
    "tHS_AG_C1'-B1-B1pair":1, "tHS_AG_B1-B1pair-C1'pair":2, "tHS_AG_C4'-C1'-B1-B1pair":2, "tHS_AG_B1-B1pair-C1'pair-C4'pair":2, "tHS_AG_alpha_1":1, "tHS_AG_alpha_2":2, "tHS AG dB1":2, "tHS AG dB2":1, 
    "cSS_AG_C1'-B1-B1pair":2, "cSS_AG_B1-B1pair-C1'pair":2, "cSS_AG_C4'-C1'-B1-B1pair":2, "cSS_AG_B1-B1pair-C1'pair-C4'pair":1, "cSS_AG_alpha_1":2, "cSS_AG_alpha_2":1, "cSS AG dB1":2, "cSS AG dB2":4, 
    "tSS_AG_C1'-B1-B1pair":3, "tSS_AG_B1-B1pair-C1'pair":1, "tSS_AG_C4'-C1'-B1-B1pair":2, "tSS_AG_B1-B1pair-C1'pair-C4'pair":1, "tSS_AG_alpha_1":2, "tSS_AG_alpha_2":1, "tSS AG dB1":2, "tSS AG dB2":4, 
    "cWW_AU_C1'-B1-B1pair":1, "cWW_AU_B1-B1pair-C1'pair":2, "cWW_AU_C4'-C1'-B1-B1pair":3, "cWW_AU_B1-B1pair-C1'pair-C4'pair":2, "cWW_AU_alpha_1":3, "cWW_AU_alpha_2":1, "cWW AU dB1":4, "cWW AU dB2":2, 
    "tWW_AU_C1'-B1-B1pair":3, "tWW_AU_B1-B1pair-C1'pair":3, "tWW_AU_C4'-C1'-B1-B1pair":2, "tWW_AU_B1-B1pair-C1'pair-C4'pair":2, "tWW_AU_alpha_1":3, "tWW_AU_alpha_2":2, "tWW AU dB1":3, "tWW AU dB2":2, 
    "cWH_AU_C1'-B1-B1pair":2, "cWH_AU_B1-B1pair-C1'pair":2, "cWH_AU_C4'-C1'-B1-B1pair":2, "cWH_AU_B1-B1pair-C1'pair-C4'pair":2, "cWH_AU_alpha_1":1, "cWH_AU_alpha_2":3, "cWH AU dB1":3, "cWH AU dB2":3, 
    "tWH_AU_C1'-B1-B1pair":1, "tWH_AU_B1-B1pair-C1'pair":3, "tWH_AU_C4'-C1'-B1-B1pair":2, "tWH_AU_B1-B1pair-C1'pair-C4'pair":2, "tWH_AU_alpha_1":2, "tWH_AU_alpha_2":2, "tWH AU dB1":1, "tWH AU dB2":3, 
    "cHW_AU_C1'-B1-B1pair":3, "cHW_AU_B1-B1pair-C1'pair":3, "cHW_AU_C4'-C1'-B1-B1pair":1, "cHW_AU_B1-B1pair-C1'pair-C4'pair":2, "cHW_AU_alpha_1":1, "cHW_AU_alpha_2":2, "cHW AU dB1":2, "cHW AU dB2":2, 
    "tHW_AU_C1'-B1-B1pair":2, "tHW_AU_B1-B1pair-C1'pair":2, "tHW_AU_C4'-C1'-B1-B1pair":1, "tHW_AU_B1-B1pair-C1'pair-C4'pair":2, "tHW_AU_alpha_1":2, "tHW_AU_alpha_2":1, "tHW AU dB1":1, "tHW AU dB2":4, 
    "cWS_AU_C1'-B1-B1pair":1, "cWS_AU_B1-B1pair-C1'pair":1, "cWS_AU_C4'-C1'-B1-B1pair":2, "cWS_AU_B1-B1pair-C1'pair-C4'pair":1, "cWS_AU_alpha_1":2, "cWS_AU_alpha_2":2, "cWS AU dB1":2, "cWS AU dB2":5, 
    "tWS_AU_C1'-B1-B1pair":2, "tWS_AU_B1-B1pair-C1'pair":2, "tWS_AU_C4'-C1'-B1-B1pair":2, "tWS_AU_B1-B1pair-C1'pair-C4'pair":1, "tWS_AU_alpha_1":2, "tWS_AU_alpha_2":2, "tWS AU dB1":3, "tWS AU dB2":4, 
    "cSW_AU_C1'-B1-B1pair":3, "cSW_AU_B1-B1pair-C1'pair":2, "cSW_AU_C4'-C1'-B1-B1pair":2, "cSW_AU_B1-B1pair-C1'pair-C4'pair":2, "cSW_AU_alpha_1":3, "cSW_AU_alpha_2":2, "cSW AU dB1":2, "cSW AU dB2":3, 
    "tSW_AU_C1'-B1-B1pair":2, "tSW_AU_B1-B1pair-C1'pair":3, "tSW_AU_C4'-C1'-B1-B1pair":3, "tSW_AU_B1-B1pair-C1'pair-C4'pair":2, "tSW_AU_alpha_1":2, "tSW_AU_alpha_2":1, "tSW AU dB1":3, "tSW AU dB2":4, 
    "cHH_AU_C1'-B1-B1pair":2, "cHH_AU_B1-B1pair-C1'pair":1, "cHH_AU_C4'-C1'-B1-B1pair":1, "cHH_AU_B1-B1pair-C1'pair-C4'pair":1, "cHH_AU_alpha_1":2, "cHH_AU_alpha_2":2, "cHH AU dB1":1, "cHH AU dB2":2, 
    "tHH_AU_C1'-B1-B1pair":3, "tHH_AU_B1-B1pair-C1'pair":3, "tHH_AU_C4'-C1'-B1-B1pair":3, "tHH_AU_B1-B1pair-C1'pair-C4'pair":2, "tHH_AU_alpha_1":3, "tHH_AU_alpha_2":3, "tHH AU dB1":1, "tHH AU dB2":3, 
    "cSH_AU_C1'-B1-B1pair":1, "cSH_AU_B1-B1pair-C1'pair":3, "cSH_AU_C4'-C1'-B1-B1pair":3, "cSH_AU_B1-B1pair-C1'pair-C4'pair":2, "cSH_AU_alpha_1":2, "cSH_AU_alpha_2":1, "cSH AU dB1":4, "cSH AU dB2":4, 
    "tSH_AU_C1'-B1-B1pair":3, "tSH_AU_B1-B1pair-C1'pair":1, "tSH_AU_C4'-C1'-B1-B1pair":1, "tSH_AU_B1-B1pair-C1'pair-C4'pair":2, "tSH_AU_alpha_1":3, "tSH_AU_alpha_2":3, "tSH AU dB1":3, "tSH AU dB2":4, 
    "cHS_AU_C1'-B1-B1pair":3, "cHS_AU_B1-B1pair-C1'pair":1, "cHS_AU_C4'-C1'-B1-B1pair":2, "cHS_AU_B1-B1pair-C1'pair-C4'pair":1, "cHS_AU_alpha_1":2, "cHS_AU_alpha_2":2, "cHS AU dB1":1, "cHS AU dB2":3, 
    "tHS_AU_C1'-B1-B1pair":2, "tHS_AU_B1-B1pair-C1'pair":2, "tHS_AU_C4'-C1'-B1-B1pair":2, "tHS_AU_B1-B1pair-C1'pair-C4'pair":3, "tHS_AU_alpha_1":3, "tHS_AU_alpha_2":2, "tHS AU dB1":3, "tHS AU dB2":3, 
    "cSS_AU_C1'-B1-B1pair":2, "cSS_AU_B1-B1pair-C1'pair":2, "cSS_AU_C4'-C1'-B1-B1pair":1, "cSS_AU_B1-B1pair-C1'pair-C4'pair":1, "cSS_AU_alpha_1":3, "cSS_AU_alpha_2":2, "cSS AU dB1":1, "cSS AU dB2":4, 
    "tSS_AU_C1'-B1-B1pair":2, "tSS_AU_B1-B1pair-C1'pair":1, "tSS_AU_C4'-C1'-B1-B1pair":3, "tSS_AU_B1-B1pair-C1'pair-C4'pair":2, "tSS_AU_alpha_1":2, "tSS_AU_alpha_2":3, "tSS AU dB1":3, "tSS AU dB2":8, 
    "cWW_CA_C1'-B1-B1pair":2, "cWW_CA_B1-B1pair-C1'pair":1, "cWW_CA_C4'-C1'-B1-B1pair":1, "cWW_CA_B1-B1pair-C1'pair-C4'pair":2, "cWW_CA_alpha_1":1, "cWW_CA_alpha_2":2, "cWW CA dB1":1, "cWW CA dB2":1, 
    "tWW_CA_C1'-B1-B1pair":2, "tWW_CA_B1-B1pair-C1'pair":2, "tWW_CA_C4'-C1'-B1-B1pair":3, "tWW_CA_B1-B1pair-C1'pair-C4'pair":2, "tWW_CA_alpha_1":2, "tWW_CA_alpha_2":1, "tWW CA dB1":4, "tWW CA dB2":2, 
    "cWH_CA_C1'-B1-B1pair":3, "cWH_CA_B1-B1pair-C1'pair":2, "cWH_CA_C4'-C1'-B1-B1pair":1, "cWH_CA_B1-B1pair-C1'pair-C4'pair":3, "cWH_CA_alpha_1":3, "cWH_CA_alpha_2":2, "cWH CA dB1":5, "cWH CA dB2":2, 
    "tWH_CA_C1'-B1-B1pair":1, "tWH_CA_B1-B1pair-C1'pair":1, "tWH_CA_C4'-C1'-B1-B1pair":1, "tWH_CA_B1-B1pair-C1'pair-C4'pair":2, "tWH_CA_alpha_1":3, "tWH_CA_alpha_2":1, "tWH CA dB1":3, "tWH CA dB2":2, 
    "cHW_CA_C1'-B1-B1pair":2, "cHW_CA_B1-B1pair-C1'pair":2, "cHW_CA_C4'-C1'-B1-B1pair":2, "cHW_CA_B1-B1pair-C1'pair-C4'pair":2, "cHW_CA_alpha_1":2, "cHW_CA_alpha_2":2, "cHW CA dB1":4, "cHW CA dB2":2, 
    "tHW_CA_C1'-B1-B1pair":2, "tHW_CA_B1-B1pair-C1'pair":2, "tHW_CA_C4'-C1'-B1-B1pair":2, "tHW_CA_B1-B1pair-C1'pair-C4'pair":2, "tHW_CA_alpha_1":2, "tHW_CA_alpha_2":2, "tHW CA dB1":6, "tHW CA dB2":2, 
    "cWS_CA_C1'-B1-B1pair":2, "cWS_CA_B1-B1pair-C1'pair":2, "cWS_CA_C4'-C1'-B1-B1pair":2, "cWS_CA_B1-B1pair-C1'pair-C4'pair":1, "cWS_CA_alpha_1":2, "cWS_CA_alpha_2":2, "cWS CA dB1":4, "cWS CA dB2":2, 
    "tWS_CA_C1'-B1-B1pair":3, "tWS_CA_B1-B1pair-C1'pair":1, "tWS_CA_C4'-C1'-B1-B1pair":3, "tWS_CA_B1-B1pair-C1'pair-C4'pair":2, "tWS_CA_alpha_1":3, "tWS_CA_alpha_2":1, "tWS CA dB1":1, "tWS CA dB2":1, 
    "cSW_CA_C1'-B1-B1pair":1, "cSW_CA_B1-B1pair-C1'pair":1, "cSW_CA_C4'-C1'-B1-B1pair":1, "cSW_CA_B1-B1pair-C1'pair-C4'pair":2, "cSW_CA_alpha_1":1, "cSW_CA_alpha_2":3, "cSW CA dB1":1, "cSW CA dB2":1, 
    "tSW_CA_C1'-B1-B1pair":2, "tSW_CA_B1-B1pair-C1'pair":2, "tSW_CA_C4'-C1'-B1-B1pair":1, "tSW_CA_B1-B1pair-C1'pair-C4'pair":1, "tSW_CA_alpha_1":2, "tSW_CA_alpha_2":3, "tSW CA dB1":3, "tSW CA dB2":1, 
    "cHH_CA_C1'-B1-B1pair":2, "cHH_CA_B1-B1pair-C1'pair":1, "cHH_CA_C4'-C1'-B1-B1pair":3, "cHH_CA_B1-B1pair-C1'pair-C4'pair":1, "cHH_CA_alpha_1":1, "cHH_CA_alpha_2":1, "cHH CA dB1":1, "cHH CA dB2":2, 
    "tHH_CA_C1'-B1-B1pair":2, "tHH_CA_B1-B1pair-C1'pair":2, "tHH_CA_C4'-C1'-B1-B1pair":3, "tHH_CA_B1-B1pair-C1'pair-C4'pair":3, "tHH_CA_alpha_1":2, "tHH_CA_alpha_2":1, "tHH CA dB1":3, "tHH CA dB2":5, 
    "cSH_CA_C1'-B1-B1pair":1, "cSH_CA_B1-B1pair-C1'pair":3, "cSH_CA_C4'-C1'-B1-B1pair":1, "cSH_CA_B1-B1pair-C1'pair-C4'pair":1, "cSH_CA_alpha_1":1, "cSH_CA_alpha_2":1, "cSH CA dB1":2, "cSH CA dB2":3, 
    "tSH_CA_C1'-B1-B1pair":1, "tSH_CA_B1-B1pair-C1'pair":2, "tSH_CA_C4'-C1'-B1-B1pair":2, "tSH_CA_B1-B1pair-C1'pair-C4'pair":1, "tSH_CA_alpha_1":3, "tSH_CA_alpha_2":2, "tSH CA dB1":6, "tSH CA dB2":4, 
    "cHS_CA_C1'-B1-B1pair":2, "cHS_CA_B1-B1pair-C1'pair":2, "cHS_CA_C4'-C1'-B1-B1pair":1, "cHS_CA_B1-B1pair-C1'pair-C4'pair":1, "cHS_CA_alpha_1":1, "cHS_CA_alpha_2":2, "cHS CA dB1":2, "cHS CA dB2":2, 
    "tHS_CA_C1'-B1-B1pair":2, "tHS_CA_B1-B1pair-C1'pair":1, "tHS_CA_C4'-C1'-B1-B1pair":2, "tHS_CA_B1-B1pair-C1'pair-C4'pair":2, "tHS_CA_alpha_1":3, "tHS_CA_alpha_2":3, "tHS CA dB1":2, "tHS CA dB2":1, 
    "cSS_CA_C1'-B1-B1pair":2, "cSS_CA_B1-B1pair-C1'pair":2, "cSS_CA_C4'-C1'-B1-B1pair":1, "cSS_CA_B1-B1pair-C1'pair-C4'pair":1, "cSS_CA_alpha_1":3, "cSS_CA_alpha_2":3, "cSS CA dB1":3, "cSS CA dB2":1, 
    "tSS_CA_C1'-B1-B1pair":2, "tSS_CA_B1-B1pair-C1'pair":2, "tSS_CA_C4'-C1'-B1-B1pair":2, "tSS_CA_B1-B1pair-C1'pair-C4'pair":1, "tSS_CA_alpha_1":2, "tSS_CA_alpha_2":2, "tSS CA dB1":4, "tSS CA dB2":2, 
    "cWW_CC_C1'-B1-B1pair":1, "cWW_CC_B1-B1pair-C1'pair":1, "cWW_CC_C4'-C1'-B1-B1pair":2, "cWW_CC_B1-B1pair-C1'pair-C4'pair":2, "cWW_CC_alpha_1":1, "cWW_CC_alpha_2":2, "cWW CC dB1":2, "cWW CC dB2":2, 
    "tWW_CC_C1'-B1-B1pair":3, "tWW_CC_B1-B1pair-C1'pair":3, "tWW_CC_C4'-C1'-B1-B1pair":3, "tWW_CC_B1-B1pair-C1'pair-C4'pair":3, "tWW_CC_alpha_1":2, "tWW_CC_alpha_2":2, "tWW CC dB1":6, "tWW CC dB2":3, 
    "cWH_CC_C1'-B1-B1pair":2, "cWH_CC_B1-B1pair-C1'pair":2, "cWH_CC_C4'-C1'-B1-B1pair":1, "cWH_CC_B1-B1pair-C1'pair-C4'pair":1, "cWH_CC_alpha_1":1, "cWH_CC_alpha_2":3, "cWH CC dB1":3, "cWH CC dB2":2, 
    "tWH_CC_C1'-B1-B1pair":1, "tWH_CC_B1-B1pair-C1'pair":3, "tWH_CC_C4'-C1'-B1-B1pair":2, "tWH_CC_B1-B1pair-C1'pair-C4'pair":1, "tWH_CC_alpha_1":3, "tWH_CC_alpha_2":1, "tWH CC dB1":3, "tWH CC dB2":3, 
    "cHW_CC_C1'-B1-B1pair":3, "cHW_CC_B1-B1pair-C1'pair":2, "cHW_CC_C4'-C1'-B1-B1pair":1, "cHW_CC_B1-B1pair-C1'pair-C4'pair":1, "cHW_CC_alpha_1":2, "cHW_CC_alpha_2":2, "cHW CC dB1":2, "cHW CC dB2":3, 
    "tHW_CC_C1'-B1-B1pair":1, "tHW_CC_B1-B1pair-C1'pair":3, "tHW_CC_C4'-C1'-B1-B1pair":3, "tHW_CC_B1-B1pair-C1'pair-C4'pair":1, "tHW_CC_alpha_1":2, "tHW_CC_alpha_2":2, "tHW CC dB1":3, "tHW CC dB2":3, 
    "cWS_CC_C1'-B1-B1pair":2, "cWS_CC_B1-B1pair-C1'pair":2, "cWS_CC_C4'-C1'-B1-B1pair":1, "cWS_CC_B1-B1pair-C1'pair-C4'pair":1, "cWS_CC_alpha_1":2, "cWS_CC_alpha_2":3, "cWS CC dB1":2, "cWS CC dB2":1, 
    "tWS_CC_C1'-B1-B1pair":2, "tWS_CC_B1-B1pair-C1'pair":2, "tWS_CC_C4'-C1'-B1-B1pair":2, "tWS_CC_B1-B1pair-C1'pair-C4'pair":1, "tWS_CC_alpha_1":2, "tWS_CC_alpha_2":2, "tWS CC dB1":2, "tWS CC dB2":2, 
    "cSW_CC_C1'-B1-B1pair":2, "cSW_CC_B1-B1pair-C1'pair":2, "cSW_CC_C4'-C1'-B1-B1pair":2, "cSW_CC_B1-B1pair-C1'pair-C4'pair":1, "cSW_CC_alpha_1":3, "cSW_CC_alpha_2":2, "cSW CC dB1":2, "cSW CC dB2":2, 
    "tSW_CC_C1'-B1-B1pair":1, "tSW_CC_B1-B1pair-C1'pair":2, "tSW_CC_C4'-C1'-B1-B1pair":1, "tSW_CC_B1-B1pair-C1'pair-C4'pair":2, "tSW_CC_alpha_1":1, "tSW_CC_alpha_2":2, "tSW CC dB1":3, "tSW CC dB2":2, 
    "cHH_CC_C1'-B1-B1pair":1, "cHH_CC_B1-B1pair-C1'pair":1, "cHH_CC_C4'-C1'-B1-B1pair":1, "cHH_CC_B1-B1pair-C1'pair-C4'pair":1, "cHH_CC_alpha_1":2, "cHH_CC_alpha_2":1, "cHH CC dB1":7, "cHH CC dB2":7, 
    "tHH_CC_C1'-B1-B1pair":3, "tHH_CC_B1-B1pair-C1'pair":2, "tHH_CC_C4'-C1'-B1-B1pair":3, "tHH_CC_B1-B1pair-C1'pair-C4'pair":2, "tHH_CC_alpha_1":1, "tHH_CC_alpha_2":3, "tHH CC dB1":5, "tHH CC dB2":5, 
    "cSH_CC_C1'-B1-B1pair":2, "cSH_CC_B1-B1pair-C1'pair":2, "cSH_CC_C4'-C1'-B1-B1pair":1, "cSH_CC_B1-B1pair-C1'pair-C4'pair":2, "cSH_CC_alpha_1":3, "cSH_CC_alpha_2":2, "cSH CC dB1":5, "cSH CC dB2":2, 
    "tSH_CC_C1'-B1-B1pair":2, "tSH_CC_B1-B1pair-C1'pair":1, "tSH_CC_C4'-C1'-B1-B1pair":2, "tSH_CC_B1-B1pair-C1'pair-C4'pair":2, "tSH_CC_alpha_1":3, "tSH_CC_alpha_2":1, "tSH CC dB1":4, "tSH CC dB2":2, 
    "cHS_CC_C1'-B1-B1pair":2, "cHS_CC_B1-B1pair-C1'pair":2, "cHS_CC_C4'-C1'-B1-B1pair":2, "cHS_CC_B1-B1pair-C1'pair-C4'pair":2, "cHS_CC_alpha_1":3, "cHS_CC_alpha_2":2, "cHS CC dB1":2, "cHS CC dB2":2, 
    "tHS_CC_C1'-B1-B1pair":3, "tHS_CC_B1-B1pair-C1'pair":1, "tHS_CC_C4'-C1'-B1-B1pair":2, "tHS_CC_B1-B1pair-C1'pair-C4'pair":3, "tHS_CC_alpha_1":1, "tHS_CC_alpha_2":2, "tHS CC dB1":4, "tHS CC dB2":4, 
    "cSS_CC_C1'-B1-B1pair":2, "cSS_CC_B1-B1pair-C1'pair":2, "cSS_CC_C4'-C1'-B1-B1pair":2, "cSS_CC_B1-B1pair-C1'pair-C4'pair":1, "cSS_CC_alpha_1":1, "cSS_CC_alpha_2":3, "cSS CC dB1":1, "cSS CC dB2":3, 
    "tSS_CC_C1'-B1-B1pair":2, "tSS_CC_B1-B1pair-C1'pair":2, "tSS_CC_C4'-C1'-B1-B1pair":3, "tSS_CC_B1-B1pair-C1'pair-C4'pair":2, "tSS_CC_alpha_1":3, "tSS_CC_alpha_2":2, "tSS CC dB1":2, "tSS CC dB2":1, 
    "cWW_CG_C1'-B1-B1pair":2, "cWW_CG_B1-B1pair-C1'pair":1, "cWW_CG_C4'-C1'-B1-B1pair":2, "cWW_CG_B1-B1pair-C1'pair-C4'pair":2, "cWW_CG_alpha_1":2, "cWW_CG_alpha_2":3, "cWW CG dB1":2, "cWW CG dB2":2, 
    "tWW_CG_C1'-B1-B1pair":1, "tWW_CG_B1-B1pair-C1'pair":2, "tWW_CG_C4'-C1'-B1-B1pair":1, "tWW_CG_B1-B1pair-C1'pair-C4'pair":2, "tWW_CG_alpha_1":2, "tWW_CG_alpha_2":1, "tWW CG dB1":1, "tWW CG dB2":4, 
    "cWH_CG_C1'-B1-B1pair":1, "cWH_CG_B1-B1pair-C1'pair":1, "cWH_CG_C4'-C1'-B1-B1pair":2, "cWH_CG_B1-B1pair-C1'pair-C4'pair":1, "cWH_CG_alpha_1":2, "cWH_CG_alpha_2":1, "cWH CG dB1":4, "cWH CG dB2":2, 
    "tWH_CG_C1'-B1-B1pair":2, "tWH_CG_B1-B1pair-C1'pair":1, "tWH_CG_C4'-C1'-B1-B1pair":1, "tWH_CG_B1-B1pair-C1'pair-C4'pair":3, "tWH_CG_alpha_1":2, "tWH_CG_alpha_2":1, "tWH CG dB1":3, "tWH CG dB2":2, 
    "cHW_CG_C1'-B1-B1pair":2, "cHW_CG_B1-B1pair-C1'pair":2, "cHW_CG_C4'-C1'-B1-B1pair":1, "cHW_CG_B1-B1pair-C1'pair-C4'pair":2, "cHW_CG_alpha_1":1, "cHW_CG_alpha_2":2, "cHW CG dB1":2, "cHW CG dB2":2, 
    "tHW_CG_C1'-B1-B1pair":1, "tHW_CG_B1-B1pair-C1'pair":2, "tHW_CG_C4'-C1'-B1-B1pair":1, "tHW_CG_B1-B1pair-C1'pair-C4'pair":2, "tHW_CG_alpha_1":3, "tHW_CG_alpha_2":2, "tHW CG dB1":4, "tHW CG dB2":3, 
    "cWS_CG_C1'-B1-B1pair":1, "cWS_CG_B1-B1pair-C1'pair":1, "cWS_CG_C4'-C1'-B1-B1pair":1, "cWS_CG_B1-B1pair-C1'pair-C4'pair":1, "cWS_CG_alpha_1":1, "cWS_CG_alpha_2":2, "cWS CG dB1":2, "cWS CG dB2":3, 
    "tWS_CG_C1'-B1-B1pair":3, "tWS_CG_B1-B1pair-C1'pair":1, "tWS_CG_C4'-C1'-B1-B1pair":1, "tWS_CG_B1-B1pair-C1'pair-C4'pair":1, "tWS_CG_alpha_1":2, "tWS_CG_alpha_2":1, "tWS CG dB1":2, "tWS CG dB2":4, 
    "cSW_CG_C1'-B1-B1pair":1, "cSW_CG_B1-B1pair-C1'pair":2, "cSW_CG_C4'-C1'-B1-B1pair":1, "cSW_CG_B1-B1pair-C1'pair-C4'pair":3, "cSW_CG_alpha_1":1, "cSW_CG_alpha_2":2, "cSW CG dB1":1, "cSW CG dB2":3, 
    "tSW_CG_C1'-B1-B1pair":1, "tSW_CG_B1-B1pair-C1'pair":2, "tSW_CG_C4'-C1'-B1-B1pair":3, "tSW_CG_B1-B1pair-C1'pair-C4'pair":2, "tSW_CG_alpha_1":1, "tSW_CG_alpha_2":2, "tSW CG dB1":7, "tSW CG dB2":2, 
    "cHH_CG_C1'-B1-B1pair":1, "cHH_CG_B1-B1pair-C1'pair":2, "cHH_CG_C4'-C1'-B1-B1pair":3, "cHH_CG_B1-B1pair-C1'pair-C4'pair":2, "cHH_CG_alpha_1":1, "cHH_CG_alpha_2":2, "cHH CG dB1":4, "cHH CG dB2":1, 
    "tHH_CG_C1'-B1-B1pair":2, "tHH_CG_B1-B1pair-C1'pair":2, "tHH_CG_C4'-C1'-B1-B1pair":3, "tHH_CG_B1-B1pair-C1'pair-C4'pair":1, "tHH_CG_alpha_1":2, "tHH_CG_alpha_2":3, "tHH CG dB1":3, "tHH CG dB2":4, 
    "cSH_CG_C1'-B1-B1pair":1, "cSH_CG_B1-B1pair-C1'pair":2, "cSH_CG_C4'-C1'-B1-B1pair":2, "cSH_CG_B1-B1pair-C1'pair-C4'pair":1, "cSH_CG_alpha_1":1, "cSH_CG_alpha_2":2, "cSH CG dB1":6, "cSH CG dB2":4, 
    "tSH_CG_C1'-B1-B1pair":1, "tSH_CG_B1-B1pair-C1'pair":2, "tSH_CG_C4'-C1'-B1-B1pair":2, "tSH_CG_B1-B1pair-C1'pair-C4'pair":1, "tSH_CG_alpha_1":1, "tSH_CG_alpha_2":3, "tSH CG dB1":2, "tSH CG dB2":3, 
    "cHS_CG_C1'-B1-B1pair":2, "cHS_CG_B1-B1pair-C1'pair":2, "cHS_CG_C4'-C1'-B1-B1pair":3, "cHS_CG_B1-B1pair-C1'pair-C4'pair":2, "cHS_CG_alpha_1":2, "cHS_CG_alpha_2":3, "cHS CG dB1":5, "cHS CG dB2":2, 
    "tHS_CG_C1'-B1-B1pair":1, "tHS_CG_B1-B1pair-C1'pair":2, "tHS_CG_C4'-C1'-B1-B1pair":3, "tHS_CG_B1-B1pair-C1'pair-C4'pair":1, "tHS_CG_alpha_1":1, "tHS_CG_alpha_2":1, "tHS CG dB1":3, "tHS CG dB2":2, 
    "cSS_CG_C1'-B1-B1pair":2, "cSS_CG_B1-B1pair-C1'pair":1, "cSS_CG_C4'-C1'-B1-B1pair":1, "cSS_CG_B1-B1pair-C1'pair-C4'pair":1, "cSS_CG_alpha_1":1, "cSS_CG_alpha_2":2, "cSS CG dB1":3, "cSS CG dB2":3, 
    "tSS_CG_C1'-B1-B1pair":2, "tSS_CG_B1-B1pair-C1'pair":2, "tSS_CG_C4'-C1'-B1-B1pair":1, "tSS_CG_B1-B1pair-C1'pair-C4'pair":2, "tSS_CG_alpha_1":1, "tSS_CG_alpha_2":2, "tSS CG dB1":1, "tSS CG dB2":2, 
    "cWW_CU_C1'-B1-B1pair":1, "cWW_CU_B1-B1pair-C1'pair":1, "cWW_CU_C4'-C1'-B1-B1pair":1, "cWW_CU_B1-B1pair-C1'pair-C4'pair":1, "cWW_CU_alpha_1":1, "cWW_CU_alpha_2":1, "cWW CU dB1":1, "cWW CU dB2":1, 
    "tWW_CU_C1'-B1-B1pair":2, "tWW_CU_B1-B1pair-C1'pair":2, "tWW_CU_C4'-C1'-B1-B1pair":2, "tWW_CU_B1-B1pair-C1'pair-C4'pair":2, "tWW_CU_alpha_1":1, "tWW_CU_alpha_2":2, "tWW CU dB1":2, "tWW CU dB2":1, 
    "cWH_CU_C1'-B1-B1pair":2, "cWH_CU_B1-B1pair-C1'pair":2, "cWH_CU_C4'-C1'-B1-B1pair":2, "cWH_CU_B1-B1pair-C1'pair-C4'pair":2, "cWH_CU_alpha_1":3, "cWH_CU_alpha_2":2, "cWH CU dB1":3, "cWH CU dB2":1, 
    "tWH_CU_C1'-B1-B1pair":2, "tWH_CU_B1-B1pair-C1'pair":2, "tWH_CU_C4'-C1'-B1-B1pair":3, "tWH_CU_B1-B1pair-C1'pair-C4'pair":2, "tWH_CU_alpha_1":3, "tWH_CU_alpha_2":3, "tWH CU dB1":5, "tWH CU dB2":2, 
    "cHW_CU_C1'-B1-B1pair":2, "cHW_CU_B1-B1pair-C1'pair":2, "cHW_CU_C4'-C1'-B1-B1pair":1, "cHW_CU_B1-B1pair-C1'pair-C4'pair":3, "cHW_CU_alpha_1":2, "cHW_CU_alpha_2":2, "cHW CU dB1":1, "cHW CU dB2":3, 
    "tHW_CU_C1'-B1-B1pair":1, "tHW_CU_B1-B1pair-C1'pair":1, "tHW_CU_C4'-C1'-B1-B1pair":3, "tHW_CU_B1-B1pair-C1'pair-C4'pair":1, "tHW_CU_alpha_1":1, "tHW_CU_alpha_2":2, "tHW CU dB1":3, "tHW CU dB2":3, 
    "cWS_CU_C1'-B1-B1pair":1, "cWS_CU_B1-B1pair-C1'pair":2, "cWS_CU_C4'-C1'-B1-B1pair":2, "cWS_CU_B1-B1pair-C1'pair-C4'pair":2, "cWS_CU_alpha_1":3, "cWS_CU_alpha_2":2, "cWS CU dB1":4, "cWS CU dB2":2, 
    "tWS_CU_C1'-B1-B1pair":3, "tWS_CU_B1-B1pair-C1'pair":1, "tWS_CU_C4'-C1'-B1-B1pair":1, "tWS_CU_B1-B1pair-C1'pair-C4'pair":2, "tWS_CU_alpha_1":2, "tWS_CU_alpha_2":1, "tWS CU dB1":3, "tWS CU dB2":5, 
    "cSW_CU_C1'-B1-B1pair":2, "cSW_CU_B1-B1pair-C1'pair":2, "cSW_CU_C4'-C1'-B1-B1pair":2, "cSW_CU_B1-B1pair-C1'pair-C4'pair":3, "cSW_CU_alpha_1":3, "cSW_CU_alpha_2":3, "cSW CU dB1":2, "cSW CU dB2":4, 
    "tSW_CU_C1'-B1-B1pair":2, "tSW_CU_B1-B1pair-C1'pair":2, "tSW_CU_C4'-C1'-B1-B1pair":2, "tSW_CU_B1-B1pair-C1'pair-C4'pair":2, "tSW_CU_alpha_1":2, "tSW_CU_alpha_2":2, "tSW CU dB1":2, "tSW CU dB2":2, 
    "cHH_CU_C1'-B1-B1pair":2, "cHH_CU_B1-B1pair-C1'pair":1, "cHH_CU_C4'-C1'-B1-B1pair":2, "cHH_CU_B1-B1pair-C1'pair-C4'pair":3, "cHH_CU_alpha_1":1, "cHH_CU_alpha_2":1, "cHH CU dB1":2, "cHH CU dB2":4, 
    "tHH_CU_C1'-B1-B1pair":3, "tHH_CU_B1-B1pair-C1'pair":2, "tHH_CU_C4'-C1'-B1-B1pair":2, "tHH_CU_B1-B1pair-C1'pair-C4'pair":1, "tHH_CU_alpha_1":2, "tHH_CU_alpha_2":2, "tHH CU dB1":2, "tHH CU dB2":2, 
    "cSH_CU_C1'-B1-B1pair":2, "cSH_CU_B1-B1pair-C1'pair":2, "cSH_CU_C4'-C1'-B1-B1pair":2, "cSH_CU_B1-B1pair-C1'pair-C4'pair":1, "cSH_CU_alpha_1":1, "cSH_CU_alpha_2":1, "cSH CU dB1":4, "cSH CU dB2":2, 
    "tSH_CU_C1'-B1-B1pair":2, "tSH_CU_B1-B1pair-C1'pair":3, "tSH_CU_C4'-C1'-B1-B1pair":2, "tSH_CU_B1-B1pair-C1'pair-C4'pair":2, "tSH_CU_alpha_1":3, "tSH_CU_alpha_2":3, "tSH CU dB1":4, "tSH CU dB2":2, 
    "cHS_CU_C1'-B1-B1pair":1, "cHS_CU_B1-B1pair-C1'pair":2, "cHS_CU_C4'-C1'-B1-B1pair":2, "cHS_CU_B1-B1pair-C1'pair-C4'pair":2, "cHS_CU_alpha_1":1, "cHS_CU_alpha_2":2, "cHS CU dB1":2, "cHS CU dB2":4, 
    "tHS_CU_C1'-B1-B1pair":2, "tHS_CU_B1-B1pair-C1'pair":1, "tHS_CU_C4'-C1'-B1-B1pair":2, "tHS_CU_B1-B1pair-C1'pair-C4'pair":2, "tHS_CU_alpha_1":2, "tHS_CU_alpha_2":2, "tHS CU dB1":3, "tHS CU dB2":4, 
    "cSS_CU_C1'-B1-B1pair":2, "cSS_CU_B1-B1pair-C1'pair":2, "cSS_CU_C4'-C1'-B1-B1pair":1, "cSS_CU_B1-B1pair-C1'pair-C4'pair":1, "cSS_CU_alpha_1":2, "cSS_CU_alpha_2":3, "cSS CU dB1":6, "cSS CU dB2":1, 
    "tSS_CU_C1'-B1-B1pair":2, "tSS_CU_B1-B1pair-C1'pair":3, "tSS_CU_C4'-C1'-B1-B1pair":2, "tSS_CU_B1-B1pair-C1'pair-C4'pair":2, "tSS_CU_alpha_1":3, "tSS_CU_alpha_2":3, "tSS CU dB1":7, "tSS CU dB2":2, 
    "cWW_GA_C1'-B1-B1pair":1, "cWW_GA_B1-B1pair-C1'pair":1, "cWW_GA_C4'-C1'-B1-B1pair":2, "cWW_GA_B1-B1pair-C1'pair-C4'pair":2, "cWW_GA_alpha_1":1, "cWW_GA_alpha_2":1, "cWW GA dB1":2, "cWW GA dB2":1, 
    "tWW_GA_C1'-B1-B1pair":1, "tWW_GA_B1-B1pair-C1'pair":1, "tWW_GA_C4'-C1'-B1-B1pair":1, "tWW_GA_B1-B1pair-C1'pair-C4'pair":2, "tWW_GA_alpha_1":1, "tWW_GA_alpha_2":2, "tWW GA dB1":1, "tWW GA dB2":2, 
    "cWH_GA_C1'-B1-B1pair":1, "cWH_GA_B1-B1pair-C1'pair":1, "cWH_GA_C4'-C1'-B1-B1pair":3, "cWH_GA_B1-B1pair-C1'pair-C4'pair":2, "cWH_GA_alpha_1":2, "cWH_GA_alpha_2":1, "cWH GA dB1":2, "cWH GA dB2":2, 
    "tWH_GA_C1'-B1-B1pair":1, "tWH_GA_B1-B1pair-C1'pair":2, "tWH_GA_C4'-C1'-B1-B1pair":1, "tWH_GA_B1-B1pair-C1'pair-C4'pair":1, "tWH_GA_alpha_1":2, "tWH_GA_alpha_2":2, "tWH GA dB1":1, "tWH GA dB2":6, 
    "cHW_GA_C1'-B1-B1pair":2, "cHW_GA_B1-B1pair-C1'pair":2, "cHW_GA_C4'-C1'-B1-B1pair":1, "cHW_GA_B1-B1pair-C1'pair-C4'pair":2, "cHW_GA_alpha_1":1, "cHW_GA_alpha_2":2, "cHW GA dB1":1, "cHW GA dB2":4, 
    "tHW_GA_C1'-B1-B1pair":2, "tHW_GA_B1-B1pair-C1'pair":1, "tHW_GA_C4'-C1'-B1-B1pair":2, "tHW_GA_B1-B1pair-C1'pair-C4'pair":2, "tHW_GA_alpha_1":1, "tHW_GA_alpha_2":1, "tHW GA dB1":3, "tHW GA dB2":1, 
    "cWS_GA_C1'-B1-B1pair":3, "cWS_GA_B1-B1pair-C1'pair":2, "cWS_GA_C4'-C1'-B1-B1pair":2, "cWS_GA_B1-B1pair-C1'pair-C4'pair":1, "cWS_GA_alpha_1":2, "cWS_GA_alpha_2":3, "cWS GA dB1":3, "cWS GA dB2":4, 
    "tWS_GA_C1'-B1-B1pair":3, "tWS_GA_B1-B1pair-C1'pair":2, "tWS_GA_C4'-C1'-B1-B1pair":1, "tWS_GA_B1-B1pair-C1'pair-C4'pair":1, "tWS_GA_alpha_1":1, "tWS_GA_alpha_2":2, "tWS GA dB1":2, "tWS GA dB2":5, 
    "cSW_GA_C1'-B1-B1pair":1, "cSW_GA_B1-B1pair-C1'pair":1, "cSW_GA_C4'-C1'-B1-B1pair":1, "cSW_GA_B1-B1pair-C1'pair-C4'pair":1, "cSW_GA_alpha_1":1, "cSW_GA_alpha_2":2, "cSW GA dB1":1, "cSW GA dB2":2, 
    "tSW_GA_C1'-B1-B1pair":1, "tSW_GA_B1-B1pair-C1'pair":2, "tSW_GA_C4'-C1'-B1-B1pair":1, "tSW_GA_B1-B1pair-C1'pair-C4'pair":2, "tSW_GA_alpha_1":1, "tSW_GA_alpha_2":3, "tSW GA dB1":2, "tSW GA dB2":2, 
    "cHH_GA_C1'-B1-B1pair":2, "cHH_GA_B1-B1pair-C1'pair":2, "cHH_GA_C4'-C1'-B1-B1pair":2, "cHH_GA_B1-B1pair-C1'pair-C4'pair":2, "cHH_GA_alpha_1":2, "cHH_GA_alpha_2":3, "cHH GA dB1":2, "cHH GA dB2":3, 
    "tHH_GA_C1'-B1-B1pair":3, "tHH_GA_B1-B1pair-C1'pair":2, "tHH_GA_C4'-C1'-B1-B1pair":2, "tHH_GA_B1-B1pair-C1'pair-C4'pair":2, "tHH_GA_alpha_1":1, "tHH_GA_alpha_2":2, "tHH GA dB1":3, "tHH GA dB2":2, 
    "cSH_GA_C1'-B1-B1pair":2, "cSH_GA_B1-B1pair-C1'pair":2, "cSH_GA_C4'-C1'-B1-B1pair":2, "cSH_GA_B1-B1pair-C1'pair-C4'pair":2, "cSH_GA_alpha_1":1, "cSH_GA_alpha_2":2, "cSH GA dB1":2, "cSH GA dB2":1, 
    "tSH_GA_C1'-B1-B1pair":1, "tSH_GA_B1-B1pair-C1'pair":1, "tSH_GA_C4'-C1'-B1-B1pair":2, "tSH_GA_B1-B1pair-C1'pair-C4'pair":2, "tSH_GA_alpha_1":2, "tSH_GA_alpha_2":2, "tSH GA dB1":2, "tSH GA dB2":7, 
    "cHS_GA_C1'-B1-B1pair":3, "cHS_GA_B1-B1pair-C1'pair":3, "cHS_GA_C4'-C1'-B1-B1pair":3, "cHS_GA_B1-B1pair-C1'pair-C4'pair":2, "cHS_GA_alpha_1":2, "cHS_GA_alpha_2":2, "cHS GA dB1":3, "cHS GA dB2":4, 
    "tHS_GA_C1'-B1-B1pair":3, "tHS_GA_B1-B1pair-C1'pair":1, "tHS_GA_C4'-C1'-B1-B1pair":3, "tHS_GA_B1-B1pair-C1'pair-C4'pair":2, "tHS_GA_alpha_1":2, "tHS_GA_alpha_2":1, "tHS GA dB1":1, "tHS GA dB2":2, 
    "cSS_GA_C1'-B1-B1pair":3, "cSS_GA_B1-B1pair-C1'pair":2, "cSS_GA_C4'-C1'-B1-B1pair":1, "cSS_GA_B1-B1pair-C1'pair-C4'pair":1, "cSS_GA_alpha_1":2, "cSS_GA_alpha_2":1, "cSS GA dB1":1, "cSS GA dB2":1, 
    "tSS_GA_C1'-B1-B1pair":1, "tSS_GA_B1-B1pair-C1'pair":1, "tSS_GA_C4'-C1'-B1-B1pair":1, "tSS_GA_B1-B1pair-C1'pair-C4'pair":1, "tSS_GA_alpha_1":1, "tSS_GA_alpha_2":2, "tSS GA dB1":5, "tSS GA dB2":2, 
    "cWW_GC_C1'-B1-B1pair":1, "cWW_GC_B1-B1pair-C1'pair":2, "cWW_GC_C4'-C1'-B1-B1pair":2, "cWW_GC_B1-B1pair-C1'pair-C4'pair":2, "cWW_GC_alpha_1":2, "cWW_GC_alpha_2":1, "cWW GC dB1":2, "cWW GC dB2":3, 
    "tWW_GC_C1'-B1-B1pair":1, "tWW_GC_B1-B1pair-C1'pair":2, "tWW_GC_C4'-C1'-B1-B1pair":1, "tWW_GC_B1-B1pair-C1'pair-C4'pair":2, "tWW_GC_alpha_1":1, "tWW_GC_alpha_2":2, "tWW GC dB1":3, "tWW GC dB2":4, 
    "cWH_GC_C1'-B1-B1pair":2, "cWH_GC_B1-B1pair-C1'pair":2, "cWH_GC_C4'-C1'-B1-B1pair":2, "cWH_GC_B1-B1pair-C1'pair-C4'pair":1, "cWH_GC_alpha_1":2, "cWH_GC_alpha_2":2, "cWH GC dB1":2, "cWH GC dB2":3, 
    "tWH_GC_C1'-B1-B1pair":1, "tWH_GC_B1-B1pair-C1'pair":1, "tWH_GC_C4'-C1'-B1-B1pair":2, "tWH_GC_B1-B1pair-C1'pair-C4'pair":2, "tWH_GC_alpha_1":3, "tWH_GC_alpha_2":3, "tWH GC dB1":2, "tWH GC dB2":2, 
    "cHW_GC_C1'-B1-B1pair":1, "cHW_GC_B1-B1pair-C1'pair":1, "cHW_GC_C4'-C1'-B1-B1pair":1, "cHW_GC_B1-B1pair-C1'pair-C4'pair":1, "cHW_GC_alpha_1":1, "cHW_GC_alpha_2":1, "cHW GC dB1":3, "cHW GC dB2":4, 
    "tHW_GC_C1'-B1-B1pair":2, "tHW_GC_B1-B1pair-C1'pair":2, "tHW_GC_C4'-C1'-B1-B1pair":2, "tHW_GC_B1-B1pair-C1'pair-C4'pair":1, "tHW_GC_alpha_1":2, "tHW_GC_alpha_2":2, "tHW GC dB1":2, "tHW GC dB2":4, 
    "cWS_GC_C1'-B1-B1pair":1, "cWS_GC_B1-B1pair-C1'pair":1, "cWS_GC_C4'-C1'-B1-B1pair":1, "cWS_GC_B1-B1pair-C1'pair-C4'pair":1, "cWS_GC_alpha_1":2, "cWS_GC_alpha_2":1, "cWS GC dB1":2, "cWS GC dB2":1, 
    "tWS_GC_C1'-B1-B1pair":1, "tWS_GC_B1-B1pair-C1'pair":1, "tWS_GC_C4'-C1'-B1-B1pair":3, "tWS_GC_B1-B1pair-C1'pair-C4'pair":1, "tWS_GC_alpha_1":1, "tWS_GC_alpha_2":1, "tWS GC dB1":4, "tWS GC dB2":5, 
    "cSW_GC_C1'-B1-B1pair":2, "cSW_GC_B1-B1pair-C1'pair":3, "cSW_GC_C4'-C1'-B1-B1pair":1, "cSW_GC_B1-B1pair-C1'pair-C4'pair":1, "cSW_GC_alpha_1":3, "cSW_GC_alpha_2":2, "cSW GC dB1":3, "cSW GC dB2":2, 
    "tSW_GC_C1'-B1-B1pair":1, "tSW_GC_B1-B1pair-C1'pair":3, "tSW_GC_C4'-C1'-B1-B1pair":1, "tSW_GC_B1-B1pair-C1'pair-C4'pair":2, "tSW_GC_alpha_1":2, "tSW_GC_alpha_2":2, "tSW GC dB1":4, "tSW GC dB2":2, 
    "cHH_GC_C1'-B1-B1pair":3, "cHH_GC_B1-B1pair-C1'pair":1, "cHH_GC_C4'-C1'-B1-B1pair":2, "cHH_GC_B1-B1pair-C1'pair-C4'pair":1, "cHH_GC_alpha_1":2, "cHH_GC_alpha_2":2, "cHH GC dB1":3, "cHH GC dB2":3, 
    "tHH_GC_C1'-B1-B1pair":2, "tHH_GC_B1-B1pair-C1'pair":1, "tHH_GC_C4'-C1'-B1-B1pair":1, "tHH_GC_B1-B1pair-C1'pair-C4'pair":2, "tHH_GC_alpha_1":3, "tHH_GC_alpha_2":1, "tHH GC dB1":6, "tHH GC dB2":3, 
    "cSH_GC_C1'-B1-B1pair":2, "cSH_GC_B1-B1pair-C1'pair":3, "cSH_GC_C4'-C1'-B1-B1pair":1, "cSH_GC_B1-B1pair-C1'pair-C4'pair":3, "cSH_GC_alpha_1":2, "cSH_GC_alpha_2":2, "cSH GC dB1":5, "cSH GC dB2":4, 
    "tSH_GC_C1'-B1-B1pair":1, "tSH_GC_B1-B1pair-C1'pair":2, "tSH_GC_C4'-C1'-B1-B1pair":1, "tSH_GC_B1-B1pair-C1'pair-C4'pair":4, "tSH_GC_alpha_1":1, "tSH_GC_alpha_2":2, "tSH GC dB1":2, "tSH GC dB2":3, 
    "cHS_GC_C1'-B1-B1pair":2, "cHS_GC_B1-B1pair-C1'pair":2, "cHS_GC_C4'-C1'-B1-B1pair":2, "cHS_GC_B1-B1pair-C1'pair-C4'pair":2, "cHS_GC_alpha_1":3, "cHS_GC_alpha_2":1, "cHS GC dB1":2, "cHS GC dB2":5, 
    "tHS_GC_C1'-B1-B1pair":2, "tHS_GC_B1-B1pair-C1'pair":2, "tHS_GC_C4'-C1'-B1-B1pair":2, "tHS_GC_B1-B1pair-C1'pair-C4'pair":3, "tHS_GC_alpha_1":2, "tHS_GC_alpha_2":2, "tHS GC dB1":2, "tHS GC dB2":2, 
    "cSS_GC_C1'-B1-B1pair":2, "cSS_GC_B1-B1pair-C1'pair":2, "cSS_GC_C4'-C1'-B1-B1pair":1, "cSS_GC_B1-B1pair-C1'pair-C4'pair":1, "cSS_GC_alpha_1":2, "cSS_GC_alpha_2":3, "cSS GC dB1":3, "cSS GC dB2":3, 
    "tSS_GC_C1'-B1-B1pair":2, "tSS_GC_B1-B1pair-C1'pair":2, "tSS_GC_C4'-C1'-B1-B1pair":1, "tSS_GC_B1-B1pair-C1'pair-C4'pair":1, "tSS_GC_alpha_1":2, "tSS_GC_alpha_2":3, "tSS GC dB1":2, "tSS GC dB2":1, 
    "cWW_GG_C1'-B1-B1pair":1, "cWW_GG_B1-B1pair-C1'pair":1, "cWW_GG_C4'-C1'-B1-B1pair":2, "cWW_GG_B1-B1pair-C1'pair-C4'pair":1, "cWW_GG_alpha_1":1, "cWW_GG_alpha_2":1, "cWW GG dB1":2, "cWW GG dB2":2, 
    "tWW_GG_C1'-B1-B1pair":1, "tWW_GG_B1-B1pair-C1'pair":1, "tWW_GG_C4'-C1'-B1-B1pair":2, "tWW_GG_B1-B1pair-C1'pair-C4'pair":2, "tWW_GG_alpha_1":2, "tWW_GG_alpha_2":2, "tWW GG dB1":1, "tWW GG dB2":2, 
    "cWH_GG_C1'-B1-B1pair":2, "cWH_GG_B1-B1pair-C1'pair":2, "cWH_GG_C4'-C1'-B1-B1pair":2, "cWH_GG_B1-B1pair-C1'pair-C4'pair":2, "cWH_GG_alpha_1":2, "cWH_GG_alpha_2":2, "cWH GG dB1":4, "cWH GG dB2":3, 
    "tWH_GG_C1'-B1-B1pair":1, "tWH_GG_B1-B1pair-C1'pair":2, "tWH_GG_C4'-C1'-B1-B1pair":2, "tWH_GG_B1-B1pair-C1'pair-C4'pair":2, "tWH_GG_alpha_1":2, "tWH_GG_alpha_2":2, "tWH GG dB1":2, "tWH GG dB2":3, 
    "cHW_GG_C1'-B1-B1pair":2, "cHW_GG_B1-B1pair-C1'pair":2, "cHW_GG_C4'-C1'-B1-B1pair":1, "cHW_GG_B1-B1pair-C1'pair-C4'pair":1, "cHW_GG_alpha_1":1, "cHW_GG_alpha_2":1, "cHW GG dB1":2, "cHW GG dB2":2, 
    "tHW_GG_C1'-B1-B1pair":2, "tHW_GG_B1-B1pair-C1'pair":2, "tHW_GG_C4'-C1'-B1-B1pair":1, "tHW_GG_B1-B1pair-C1'pair-C4'pair":2, "tHW_GG_alpha_1":2, "tHW_GG_alpha_2":2, "tHW GG dB1":1, "tHW GG dB2":4, 
    "cWS_GG_C1'-B1-B1pair":1, "cWS_GG_B1-B1pair-C1'pair":1, "cWS_GG_C4'-C1'-B1-B1pair":2, "cWS_GG_B1-B1pair-C1'pair-C4'pair":1, "cWS_GG_alpha_1":2, "cWS_GG_alpha_2":2, "cWS GG dB1":4, "cWS GG dB2":3, 
    "tWS_GG_C1'-B1-B1pair":3, "tWS_GG_B1-B1pair-C1'pair":2, "tWS_GG_C4'-C1'-B1-B1pair":3, "tWS_GG_B1-B1pair-C1'pair-C4'pair":2, "tWS_GG_alpha_1":1, "tWS_GG_alpha_2":1, "tWS GG dB1":1, "tWS GG dB2":3, 
    "cSW_GG_C1'-B1-B1pair":1, "cSW_GG_B1-B1pair-C1'pair":1, "cSW_GG_C4'-C1'-B1-B1pair":1, "cSW_GG_B1-B1pair-C1'pair-C4'pair":2, "cSW_GG_alpha_1":2, "cSW_GG_alpha_2":2, "cSW GG dB1":2, "cSW GG dB2":2, 
    "tSW_GG_C1'-B1-B1pair":3, "tSW_GG_B1-B1pair-C1'pair":2, "tSW_GG_C4'-C1'-B1-B1pair":3, "tSW_GG_B1-B1pair-C1'pair-C4'pair":2, "tSW_GG_alpha_1":1, "tSW_GG_alpha_2":3, "tSW GG dB1":2, "tSW GG dB2":1, 
    "cHH_GG_C1'-B1-B1pair":1, "cHH_GG_B1-B1pair-C1'pair":1, "cHH_GG_C4'-C1'-B1-B1pair":2, "cHH_GG_B1-B1pair-C1'pair-C4'pair":3, "cHH_GG_alpha_1":1, "cHH_GG_alpha_2":2, "cHH GG dB1":2, "cHH GG dB2":3, 
    "tHH_GG_C1'-B1-B1pair":2, "tHH_GG_B1-B1pair-C1'pair":2, "tHH_GG_C4'-C1'-B1-B1pair":2, "tHH_GG_B1-B1pair-C1'pair-C4'pair":3, "tHH_GG_alpha_1":2, "tHH_GG_alpha_2":2, "tHH GG dB1":2, "tHH GG dB2":3, 
    "cSH_GG_C1'-B1-B1pair":2, "cSH_GG_B1-B1pair-C1'pair":1, "cSH_GG_C4'-C1'-B1-B1pair":1, "cSH_GG_B1-B1pair-C1'pair-C4'pair":1, "cSH_GG_alpha_1":2, "cSH_GG_alpha_2":1, "cSH GG dB1":1, "cSH GG dB2":1, 
    "tSH_GG_C1'-B1-B1pair":2, "tSH_GG_B1-B1pair-C1'pair":2, "tSH_GG_C4'-C1'-B1-B1pair":2, "tSH_GG_B1-B1pair-C1'pair-C4'pair":2, "tSH_GG_alpha_1":2, "tSH_GG_alpha_2":2, "tSH GG dB1":1, "tSH GG dB2":2, 
    "cHS_GG_C1'-B1-B1pair":1, "cHS_GG_B1-B1pair-C1'pair":2, "cHS_GG_C4'-C1'-B1-B1pair":1, "cHS_GG_B1-B1pair-C1'pair-C4'pair":1, "cHS_GG_alpha_1":1, "cHS_GG_alpha_2":2, "cHS GG dB1":1, "cHS GG dB2":2, 
    "tHS_GG_C1'-B1-B1pair":2, "tHS_GG_B1-B1pair-C1'pair":2, "tHS_GG_C4'-C1'-B1-B1pair":2, "tHS_GG_B1-B1pair-C1'pair-C4'pair":1, "tHS_GG_alpha_1":2, "tHS_GG_alpha_2":3, "tHS GG dB1":2, "tHS GG dB2":1, 
    "cSS_GG_C1'-B1-B1pair":2, "cSS_GG_B1-B1pair-C1'pair":2, "cSS_GG_C4'-C1'-B1-B1pair":1, "cSS_GG_B1-B1pair-C1'pair-C4'pair":1, "cSS_GG_alpha_1":2, "cSS_GG_alpha_2":3, "cSS GG dB1":3, "cSS GG dB2":5, 
    "tSS_GG_C1'-B1-B1pair":3, "tSS_GG_B1-B1pair-C1'pair":2, "tSS_GG_C4'-C1'-B1-B1pair":2, "tSS_GG_B1-B1pair-C1'pair-C4'pair":1, "tSS_GG_alpha_1":1, "tSS_GG_alpha_2":3, "tSS GG dB1":3, "tSS GG dB2":2, 
    "cWW_GU_C1'-B1-B1pair":2, "cWW_GU_B1-B1pair-C1'pair":2, "cWW_GU_C4'-C1'-B1-B1pair":1, "cWW_GU_B1-B1pair-C1'pair-C4'pair":1, "cWW_GU_alpha_1":3, "cWW_GU_alpha_2":2, "cWW GU dB1":4, "cWW GU dB2":3, 
    "tWW_GU_C1'-B1-B1pair":3, "tWW_GU_B1-B1pair-C1'pair":2, "tWW_GU_C4'-C1'-B1-B1pair":2, "tWW_GU_B1-B1pair-C1'pair-C4'pair":3, "tWW_GU_alpha_1":2, "tWW_GU_alpha_2":2, "tWW GU dB1":3, "tWW GU dB2":3, 
    "cWH_GU_C1'-B1-B1pair":1, "cWH_GU_B1-B1pair-C1'pair":2, "cWH_GU_C4'-C1'-B1-B1pair":1, "cWH_GU_B1-B1pair-C1'pair-C4'pair":2, "cWH_GU_alpha_1":2, "cWH_GU_alpha_2":4, "cWH GU dB1":3, "cWH GU dB2":1, 
    "tWH_GU_C1'-B1-B1pair":1, "tWH_GU_B1-B1pair-C1'pair":2, "tWH_GU_C4'-C1'-B1-B1pair":2, "tWH_GU_B1-B1pair-C1'pair-C4'pair":2, "tWH_GU_alpha_1":2, "tWH_GU_alpha_2":2, "tWH GU dB1":3, "tWH GU dB2":1, 
    "cHW_GU_C1'-B1-B1pair":2, "cHW_GU_B1-B1pair-C1'pair":1, "cHW_GU_C4'-C1'-B1-B1pair":1, "cHW_GU_B1-B1pair-C1'pair-C4'pair":2, "cHW_GU_alpha_1":2, "cHW_GU_alpha_2":2, "cHW GU dB1":3, "cHW GU dB2":3, 
    "tHW_GU_C1'-B1-B1pair":3, "tHW_GU_B1-B1pair-C1'pair":1, "tHW_GU_C4'-C1'-B1-B1pair":2, "tHW_GU_B1-B1pair-C1'pair-C4'pair":3, "tHW_GU_alpha_1":3, "tHW_GU_alpha_2":1, "tHW GU dB1":2, "tHW GU dB2":5, 
    "cWS_GU_C1'-B1-B1pair":1, "cWS_GU_B1-B1pair-C1'pair":1, "cWS_GU_C4'-C1'-B1-B1pair":1, "cWS_GU_B1-B1pair-C1'pair-C4'pair":2, "cWS_GU_alpha_1":3, "cWS_GU_alpha_2":3, "cWS GU dB1":2, "cWS GU dB2":3, 
    "tWS_GU_C1'-B1-B1pair":3, "tWS_GU_B1-B1pair-C1'pair":1, "tWS_GU_C4'-C1'-B1-B1pair":2, "tWS_GU_B1-B1pair-C1'pair-C4'pair":1, "tWS_GU_alpha_1":1, "tWS_GU_alpha_2":2, "tWS GU dB1":3, "tWS GU dB2":3, 
    "cSW_GU_C1'-B1-B1pair":2, "cSW_GU_B1-B1pair-C1'pair":2, "cSW_GU_C4'-C1'-B1-B1pair":2, "cSW_GU_B1-B1pair-C1'pair-C4'pair":2, "cSW_GU_alpha_1":1, "cSW_GU_alpha_2":1, "cSW GU dB1":3, "cSW GU dB2":2, 
    "tSW_GU_C1'-B1-B1pair":1, "tSW_GU_B1-B1pair-C1'pair":2, "tSW_GU_C4'-C1'-B1-B1pair":2, "tSW_GU_B1-B1pair-C1'pair-C4'pair":2, "tSW_GU_alpha_1":1, "tSW_GU_alpha_2":2, "tSW GU dB1":5, "tSW GU dB2":1, 
    "cHH_GU_C1'-B1-B1pair":2, "cHH_GU_B1-B1pair-C1'pair":3, "cHH_GU_C4'-C1'-B1-B1pair":2, "cHH_GU_B1-B1pair-C1'pair-C4'pair":2, "cHH_GU_alpha_1":2, "cHH_GU_alpha_2":2, "cHH GU dB1":5, "cHH GU dB2":3, 
    "tHH_GU_C1'-B1-B1pair":2, "tHH_GU_B1-B1pair-C1'pair":1, "tHH_GU_C4'-C1'-B1-B1pair":1, "tHH_GU_B1-B1pair-C1'pair-C4'pair":2, "tHH_GU_alpha_1":2, "tHH_GU_alpha_2":1, "tHH GU dB1":8, "tHH GU dB2":2, 
    "cSH_GU_C1'-B1-B1pair":1, "cSH_GU_B1-B1pair-C1'pair":2, "cSH_GU_C4'-C1'-B1-B1pair":3, "cSH_GU_B1-B1pair-C1'pair-C4'pair":2, "cSH_GU_alpha_1":2, "cSH_GU_alpha_2":1, "cSH GU dB1":2, "cSH GU dB2":2, 
    "tSH_GU_C1'-B1-B1pair":2, "tSH_GU_B1-B1pair-C1'pair":2, "tSH_GU_C4'-C1'-B1-B1pair":1, "tSH_GU_B1-B1pair-C1'pair-C4'pair":1, "tSH_GU_alpha_1":2, "tSH_GU_alpha_2":3, "tSH GU dB1":3, "tSH GU dB2":3, 
    "cHS_GU_C1'-B1-B1pair":1, "cHS_GU_B1-B1pair-C1'pair":1, "cHS_GU_C4'-C1'-B1-B1pair":1, "cHS_GU_B1-B1pair-C1'pair-C4'pair":2, "cHS_GU_alpha_1":1, "cHS_GU_alpha_2":1, "cHS GU dB1":4, "cHS GU dB2":3, 
    "tHS_GU_C1'-B1-B1pair":4, "tHS_GU_B1-B1pair-C1'pair":2, "tHS_GU_C4'-C1'-B1-B1pair":2, "tHS_GU_B1-B1pair-C1'pair-C4'pair":1, "tHS_GU_alpha_1":2, "tHS_GU_alpha_2":1, "tHS GU dB1":1, "tHS GU dB2":3, 
    "cSS_GU_C1'-B1-B1pair":3, "cSS_GU_B1-B1pair-C1'pair":2, "cSS_GU_C4'-C1'-B1-B1pair":2, "cSS_GU_B1-B1pair-C1'pair-C4'pair":2, "cSS_GU_alpha_1":2, "cSS_GU_alpha_2":1, "cSS GU dB1":3, "cSS GU dB2":4, 
    "tSS_GU_C1'-B1-B1pair":2, "tSS_GU_B1-B1pair-C1'pair":2, "tSS_GU_C4'-C1'-B1-B1pair":1, "tSS_GU_B1-B1pair-C1'pair-C4'pair":3, "tSS_GU_alpha_1":2, "tSS_GU_alpha_2":2, "tSS GU dB1":2, "tSS GU dB2":6, 
    "cWW_UA_C1'-B1-B1pair":2, "cWW_UA_B1-B1pair-C1'pair":2, "cWW_UA_C4'-C1'-B1-B1pair":1, "cWW_UA_B1-B1pair-C1'pair-C4'pair":2, "cWW_UA_alpha_1":2, "cWW_UA_alpha_2":2, "cWW UA dB1":2, "cWW UA dB2":7, 
    "tWW_UA_C1'-B1-B1pair":1, "tWW_UA_B1-B1pair-C1'pair":2, "tWW_UA_C4'-C1'-B1-B1pair":1, "tWW_UA_B1-B1pair-C1'pair-C4'pair":1, "tWW_UA_alpha_1":2, "tWW_UA_alpha_2":1, "tWW UA dB1":6, "tWW UA dB2":1, 
    "cWH_UA_C1'-B1-B1pair":3, "cWH_UA_B1-B1pair-C1'pair":3, "cWH_UA_C4'-C1'-B1-B1pair":3, "cWH_UA_B1-B1pair-C1'pair-C4'pair":2, "cWH_UA_alpha_1":2, "cWH_UA_alpha_2":3, "cWH UA dB1":4, "cWH UA dB2":3, 
    "tWH_UA_C1'-B1-B1pair":2, "tWH_UA_B1-B1pair-C1'pair":1, "tWH_UA_C4'-C1'-B1-B1pair":2, "tWH_UA_B1-B1pair-C1'pair-C4'pair":2, "tWH_UA_alpha_1":1, "tWH_UA_alpha_2":2, "tWH UA dB1":3, "tWH UA dB2":2, 
    "cHW_UA_C1'-B1-B1pair":1, "cHW_UA_B1-B1pair-C1'pair":1, "cHW_UA_C4'-C1'-B1-B1pair":3, "cHW_UA_B1-B1pair-C1'pair-C4'pair":1, "cHW_UA_alpha_1":1, "cHW_UA_alpha_2":1, "cHW UA dB1":3, "cHW UA dB2":1, 
    "tHW_UA_C1'-B1-B1pair":3, "tHW_UA_B1-B1pair-C1'pair":2, "tHW_UA_C4'-C1'-B1-B1pair":1, "tHW_UA_B1-B1pair-C1'pair-C4'pair":2, "tHW_UA_alpha_1":3, "tHW_UA_alpha_2":3, "tHW UA dB1":2, "tHW UA dB2":1, 
    "cWS_UA_C1'-B1-B1pair":2, "cWS_UA_B1-B1pair-C1'pair":3, "cWS_UA_C4'-C1'-B1-B1pair":2, "cWS_UA_B1-B1pair-C1'pair-C4'pair":1, "cWS_UA_alpha_1":2, "cWS_UA_alpha_2":2, "cWS UA dB1":3, "cWS UA dB2":4, 
    "tWS_UA_C1'-B1-B1pair":1, "tWS_UA_B1-B1pair-C1'pair":2, "tWS_UA_C4'-C1'-B1-B1pair":1, "tWS_UA_B1-B1pair-C1'pair-C4'pair":1, "tWS_UA_alpha_1":1, "tWS_UA_alpha_2":3, "tWS UA dB1":1, "tWS UA dB2":1, 
    "cSW_UA_C1'-B1-B1pair":1, "cSW_UA_B1-B1pair-C1'pair":1, "cSW_UA_C4'-C1'-B1-B1pair":2, "cSW_UA_B1-B1pair-C1'pair-C4'pair":2, "cSW_UA_alpha_1":2, "cSW_UA_alpha_2":3, "cSW UA dB1":3, "cSW UA dB2":3, 
    "tSW_UA_C1'-B1-B1pair":1, "tSW_UA_B1-B1pair-C1'pair":2, "tSW_UA_C4'-C1'-B1-B1pair":1, "tSW_UA_B1-B1pair-C1'pair-C4'pair":1, "tSW_UA_alpha_1":2, "tSW_UA_alpha_2":2, "tSW UA dB1":3, "tSW UA dB2":2, 
    "cHH_UA_C1'-B1-B1pair":1, "cHH_UA_B1-B1pair-C1'pair":1, "cHH_UA_C4'-C1'-B1-B1pair":1, "cHH_UA_B1-B1pair-C1'pair-C4'pair":1, "cHH_UA_alpha_1":2, "cHH_UA_alpha_2":2, "cHH UA dB1":5, "cHH UA dB2":2, 
    "tHH_UA_C1'-B1-B1pair":2, "tHH_UA_B1-B1pair-C1'pair":2, "tHH_UA_C4'-C1'-B1-B1pair":2, "tHH_UA_B1-B1pair-C1'pair-C4'pair":2, "tHH_UA_alpha_1":2, "tHH_UA_alpha_2":3, "tHH UA dB1":3, "tHH UA dB2":1, 
    "cSH_UA_C1'-B1-B1pair":1, "cSH_UA_B1-B1pair-C1'pair":1, "cSH_UA_C4'-C1'-B1-B1pair":2, "cSH_UA_B1-B1pair-C1'pair-C4'pair":1, "cSH_UA_alpha_1":2, "cSH_UA_alpha_2":2, "cSH UA dB1":3, "cSH UA dB2":2, 
    "tSH_UA_C1'-B1-B1pair":2, "tSH_UA_B1-B1pair-C1'pair":2, "tSH_UA_C4'-C1'-B1-B1pair":3, "tSH_UA_B1-B1pair-C1'pair-C4'pair":2, "tSH_UA_alpha_1":3, "tSH_UA_alpha_2":2, "tSH UA dB1":4, "tSH UA dB2":1, 
    "cHS_UA_C1'-B1-B1pair":2, "cHS_UA_B1-B1pair-C1'pair":2, "cHS_UA_C4'-C1'-B1-B1pair":2, "cHS_UA_B1-B1pair-C1'pair-C4'pair":2, "cHS_UA_alpha_1":2, "cHS_UA_alpha_2":2, "cHS UA dB1":1, "cHS UA dB2":3, 
    "tHS_UA_C1'-B1-B1pair":2, "tHS_UA_B1-B1pair-C1'pair":2, "tHS_UA_C4'-C1'-B1-B1pair":3, "tHS_UA_B1-B1pair-C1'pair-C4'pair":1, "tHS_UA_alpha_1":3, "tHS_UA_alpha_2":3, "tHS UA dB1":2, "tHS UA dB2":7, 
    "cSS_UA_C1'-B1-B1pair":2, "cSS_UA_B1-B1pair-C1'pair":2, "cSS_UA_C4'-C1'-B1-B1pair":2, "cSS_UA_B1-B1pair-C1'pair-C4'pair":1, "cSS_UA_alpha_1":1, "cSS_UA_alpha_2":1, "cSS UA dB1":2, "cSS UA dB2":1, 
    "tSS_UA_C1'-B1-B1pair":1, "tSS_UA_B1-B1pair-C1'pair":3, "tSS_UA_C4'-C1'-B1-B1pair":2, "tSS_UA_B1-B1pair-C1'pair-C4'pair":3, "tSS_UA_alpha_1":2, "tSS_UA_alpha_2":2, "tSS UA dB1":4, "tSS UA dB2":4, 
    "cWW_UC_C1'-B1-B1pair":1, "cWW_UC_B1-B1pair-C1'pair":2, "cWW_UC_C4'-C1'-B1-B1pair":2, "cWW_UC_B1-B1pair-C1'pair-C4'pair":2, "cWW_UC_alpha_1":2, "cWW_UC_alpha_2":1, "cWW UC dB1":1, "cWW UC dB2":2, 
    "tWW_UC_C1'-B1-B1pair":2, "tWW_UC_B1-B1pair-C1'pair":2, "tWW_UC_C4'-C1'-B1-B1pair":2, "tWW_UC_B1-B1pair-C1'pair-C4'pair":2, "tWW_UC_alpha_1":3, "tWW_UC_alpha_2":1, "tWW UC dB1":1, "tWW UC dB2":4, 
    "cWH_UC_C1'-B1-B1pair":2, "cWH_UC_B1-B1pair-C1'pair":2, "cWH_UC_C4'-C1'-B1-B1pair":2, "cWH_UC_B1-B1pair-C1'pair-C4'pair":4, "cWH_UC_alpha_1":2, "cWH_UC_alpha_2":3, "cWH UC dB1":3, "cWH UC dB2":3, 
    "tWH_UC_C1'-B1-B1pair":3, "tWH_UC_B1-B1pair-C1'pair":2, "tWH_UC_C4'-C1'-B1-B1pair":3, "tWH_UC_B1-B1pair-C1'pair-C4'pair":1, "tWH_UC_alpha_1":4, "tWH_UC_alpha_2":1, "tWH UC dB1":4, "tWH UC dB2":2, 
    "cHW_UC_C1'-B1-B1pair":2, "cHW_UC_B1-B1pair-C1'pair":2, "cHW_UC_C4'-C1'-B1-B1pair":1, "cHW_UC_B1-B1pair-C1'pair-C4'pair":2, "cHW_UC_alpha_1":1, "cHW_UC_alpha_2":2, "cHW UC dB1":2, "cHW UC dB2":6, 
    "tHW_UC_C1'-B1-B1pair":2, "tHW_UC_B1-B1pair-C1'pair":2, "tHW_UC_C4'-C1'-B1-B1pair":3, "tHW_UC_B1-B1pair-C1'pair-C4'pair":2, "tHW_UC_alpha_1":2, "tHW_UC_alpha_2":4, "tHW UC dB1":4, "tHW UC dB2":4, 
    "cWS_UC_C1'-B1-B1pair":2, "cWS_UC_B1-B1pair-C1'pair":2, "cWS_UC_C4'-C1'-B1-B1pair":2, "cWS_UC_B1-B1pair-C1'pair-C4'pair":1, "cWS_UC_alpha_1":3, "cWS_UC_alpha_2":2, "cWS UC dB1":3, "cWS UC dB2":2, 
    "tWS_UC_C1'-B1-B1pair":2, "tWS_UC_B1-B1pair-C1'pair":1, "tWS_UC_C4'-C1'-B1-B1pair":2, "tWS_UC_B1-B1pair-C1'pair-C4'pair":2, "tWS_UC_alpha_1":2, "tWS_UC_alpha_2":1, "tWS UC dB1":3, "tWS UC dB2":2, 
    "cSW_UC_C1'-B1-B1pair":1, "cSW_UC_B1-B1pair-C1'pair":2, "cSW_UC_C4'-C1'-B1-B1pair":2, "cSW_UC_B1-B1pair-C1'pair-C4'pair":2, "cSW_UC_alpha_1":2, "cSW_UC_alpha_2":3, "cSW UC dB1":3, "cSW UC dB2":6, 
    "tSW_UC_C1'-B1-B1pair":1, "tSW_UC_B1-B1pair-C1'pair":2, "tSW_UC_C4'-C1'-B1-B1pair":3, "tSW_UC_B1-B1pair-C1'pair-C4'pair":1, "tSW_UC_alpha_1":2, "tSW_UC_alpha_2":2, "tSW UC dB1":2, "tSW UC dB2":1, 
    "cHH_UC_C1'-B1-B1pair":2, "cHH_UC_B1-B1pair-C1'pair":1, "cHH_UC_C4'-C1'-B1-B1pair":2, "cHH_UC_B1-B1pair-C1'pair-C4'pair":2, "cHH_UC_alpha_1":1, "cHH_UC_alpha_2":3, "cHH UC dB1":7, "cHH UC dB2":3, 
    "tHH_UC_C1'-B1-B1pair":1, "tHH_UC_B1-B1pair-C1'pair":1, "tHH_UC_C4'-C1'-B1-B1pair":2, "tHH_UC_B1-B1pair-C1'pair-C4'pair":3, "tHH_UC_alpha_1":2, "tHH_UC_alpha_2":2, "tHH UC dB1":8, "tHH UC dB2":8, 
    "cSH_UC_C1'-B1-B1pair":2, "cSH_UC_B1-B1pair-C1'pair":2, "cSH_UC_C4'-C1'-B1-B1pair":2, "cSH_UC_B1-B1pair-C1'pair-C4'pair":1, "cSH_UC_alpha_1":2, "cSH_UC_alpha_2":3, "cSH UC dB1":5, "cSH UC dB2":3, 
    "tSH_UC_C1'-B1-B1pair":1, "tSH_UC_B1-B1pair-C1'pair":1, "tSH_UC_C4'-C1'-B1-B1pair":2, "tSH_UC_B1-B1pair-C1'pair-C4'pair":1, "tSH_UC_alpha_1":2, "tSH_UC_alpha_2":2, "tSH UC dB1":2, "tSH UC dB2":7, 
    "cHS_UC_C1'-B1-B1pair":2, "cHS_UC_B1-B1pair-C1'pair":2, "cHS_UC_C4'-C1'-B1-B1pair":1, "cHS_UC_B1-B1pair-C1'pair-C4'pair":3, "cHS_UC_alpha_1":3, "cHS_UC_alpha_2":2, "cHS UC dB1":6, "cHS UC dB2":7, 
    "tHS_UC_C1'-B1-B1pair":3, "tHS_UC_B1-B1pair-C1'pair":2, "tHS_UC_C4'-C1'-B1-B1pair":2, "tHS_UC_B1-B1pair-C1'pair-C4'pair":3, "tHS_UC_alpha_1":3, "tHS_UC_alpha_2":1, "tHS UC dB1":5, "tHS UC dB2":7, 
    "cSS_UC_C1'-B1-B1pair":2, "cSS_UC_B1-B1pair-C1'pair":1, "cSS_UC_C4'-C1'-B1-B1pair":3, "cSS_UC_B1-B1pair-C1'pair-C4'pair":1, "cSS_UC_alpha_1":3, "cSS_UC_alpha_2":3, "cSS UC dB1":8, "cSS UC dB2":5, 
    "tSS_UC_C1'-B1-B1pair":2, "tSS_UC_B1-B1pair-C1'pair":1, "tSS_UC_C4'-C1'-B1-B1pair":3, "tSS_UC_B1-B1pair-C1'pair-C4'pair":3, "tSS_UC_alpha_1":3, "tSS_UC_alpha_2":1, "tSS UC dB1":8, "tSS UC dB2":7, 
    "cWW_UG_C1'-B1-B1pair":2, "cWW_UG_B1-B1pair-C1'pair":3, "cWW_UG_C4'-C1'-B1-B1pair":2, "cWW_UG_B1-B1pair-C1'pair-C4'pair":2, "cWW_UG_alpha_1":2, "cWW_UG_alpha_2":3, "cWW UG dB1":4, "cWW UG dB2":3, 
    "tWW_UG_C1'-B1-B1pair":1, "tWW_UG_B1-B1pair-C1'pair":1, "tWW_UG_C4'-C1'-B1-B1pair":2, "tWW_UG_B1-B1pair-C1'pair-C4'pair":2, "tWW_UG_alpha_1":3, "tWW_UG_alpha_2":3, "tWW UG dB1":3, "tWW UG dB2":4, 
    "cWH_UG_C1'-B1-B1pair":1, "cWH_UG_B1-B1pair-C1'pair":2, "cWH_UG_C4'-C1'-B1-B1pair":1, "cWH_UG_B1-B1pair-C1'pair-C4'pair":1, "cWH_UG_alpha_1":2, "cWH_UG_alpha_2":2, "cWH UG dB1":2, "cWH UG dB2":2, 
    "tWH_UG_C1'-B1-B1pair":2, "tWH_UG_B1-B1pair-C1'pair":2, "tWH_UG_C4'-C1'-B1-B1pair":1, "tWH_UG_B1-B1pair-C1'pair-C4'pair":2, "tWH_UG_alpha_1":2, "tWH_UG_alpha_2":2, "tWH UG dB1":6, "tWH UG dB2":2, 
    "cHW_UG_C1'-B1-B1pair":2, "cHW_UG_B1-B1pair-C1'pair":2, "cHW_UG_C4'-C1'-B1-B1pair":1, "cHW_UG_B1-B1pair-C1'pair-C4'pair":2, "cHW_UG_alpha_1":1, "cHW_UG_alpha_2":2, "cHW UG dB1":4, "cHW UG dB2":4, 
    "tHW_UG_C1'-B1-B1pair":2, "tHW_UG_B1-B1pair-C1'pair":1, "tHW_UG_C4'-C1'-B1-B1pair":2, "tHW_UG_B1-B1pair-C1'pair-C4'pair":2, "tHW_UG_alpha_1":3, "tHW_UG_alpha_2":2, "tHW UG dB1":6, "tHW UG dB2":3, 
    "cWS_UG_C1'-B1-B1pair":4, "cWS_UG_B1-B1pair-C1'pair":2, "cWS_UG_C4'-C1'-B1-B1pair":3, "cWS_UG_B1-B1pair-C1'pair-C4'pair":2, "cWS_UG_alpha_1":2, "cWS_UG_alpha_2":2, "cWS UG dB1":2, "cWS UG dB2":2, 
    "tWS_UG_C1'-B1-B1pair":2, "tWS_UG_B1-B1pair-C1'pair":2, "tWS_UG_C4'-C1'-B1-B1pair":2, "tWS_UG_B1-B1pair-C1'pair-C4'pair":2, "tWS_UG_alpha_1":2, "tWS_UG_alpha_2":1, "tWS UG dB1":3, "tWS UG dB2":5, 
    "cSW_UG_C1'-B1-B1pair":2, "cSW_UG_B1-B1pair-C1'pair":3, "cSW_UG_C4'-C1'-B1-B1pair":2, "cSW_UG_B1-B1pair-C1'pair-C4'pair":1, "cSW_UG_alpha_1":2, "cSW_UG_alpha_2":2, "cSW UG dB1":3, "cSW UG dB2":2, 
    "tSW_UG_C1'-B1-B1pair":1, "tSW_UG_B1-B1pair-C1'pair":1, "tSW_UG_C4'-C1'-B1-B1pair":1, "tSW_UG_B1-B1pair-C1'pair-C4'pair":2, "tSW_UG_alpha_1":2, "tSW_UG_alpha_2":2, "tSW UG dB1":2, "tSW UG dB2":2, 
    "cHH_UG_C1'-B1-B1pair":3, "cHH_UG_B1-B1pair-C1'pair":2, "cHH_UG_C4'-C1'-B1-B1pair":2, "cHH_UG_B1-B1pair-C1'pair-C4'pair":2, "cHH_UG_alpha_1":2, "cHH_UG_alpha_2":3, "cHH UG dB1":4, "cHH UG dB2":5, 
    "tHH_UG_C1'-B1-B1pair":2, "tHH_UG_B1-B1pair-C1'pair":2, "tHH_UG_C4'-C1'-B1-B1pair":2, "tHH_UG_B1-B1pair-C1'pair-C4'pair":3, "tHH_UG_alpha_1":3, "tHH_UG_alpha_2":2, "tHH UG dB1":3, "tHH UG dB2":2, 
    "cSH_UG_C1'-B1-B1pair":1, "cSH_UG_B1-B1pair-C1'pair":2, "cSH_UG_C4'-C1'-B1-B1pair":2, "cSH_UG_B1-B1pair-C1'pair-C4'pair":2, "cSH_UG_alpha_1":2, "cSH_UG_alpha_2":2, "cSH UG dB1":3, "cSH UG dB2":4, 
    "tSH_UG_C1'-B1-B1pair":2, "tSH_UG_B1-B1pair-C1'pair":1, "tSH_UG_C4'-C1'-B1-B1pair":2, "tSH_UG_B1-B1pair-C1'pair-C4'pair":1, "tSH_UG_alpha_1":3, "tSH_UG_alpha_2":1, "tSH UG dB1":2, "tSH UG dB2":2, 
    "cHS_UG_C1'-B1-B1pair":2, "cHS_UG_B1-B1pair-C1'pair":3, "cHS_UG_C4'-C1'-B1-B1pair":2, "cHS_UG_B1-B1pair-C1'pair-C4'pair":4, "cHS_UG_alpha_1":2, "cHS_UG_alpha_2":3, "cHS UG dB1":3, "cHS UG dB2":4, 
    "tHS_UG_C1'-B1-B1pair":1, "tHS_UG_B1-B1pair-C1'pair":3, "tHS_UG_C4'-C1'-B1-B1pair":2, "tHS_UG_B1-B1pair-C1'pair-C4'pair":1, "tHS_UG_alpha_1":2, "tHS_UG_alpha_2":3, "tHS UG dB1":2, "tHS UG dB2":1, 
    "cSS_UG_C1'-B1-B1pair":2, "cSS_UG_B1-B1pair-C1'pair":2, "cSS_UG_C4'-C1'-B1-B1pair":2, "cSS_UG_B1-B1pair-C1'pair-C4'pair":2, "cSS_UG_alpha_1":1, "cSS_UG_alpha_2":2, "cSS UG dB1":2, "cSS UG dB2":3, 
    "tSS_UG_C1'-B1-B1pair":2, "tSS_UG_B1-B1pair-C1'pair":2, "tSS_UG_C4'-C1'-B1-B1pair":1, "tSS_UG_B1-B1pair-C1'pair-C4'pair":2, "tSS_UG_alpha_1":2, "tSS_UG_alpha_2":2, "tSS UG dB1":3, "tSS UG dB2":4, 
    "cWW_UU_C1'-B1-B1pair":2, "cWW_UU_B1-B1pair-C1'pair":3, "cWW_UU_C4'-C1'-B1-B1pair":3, "cWW_UU_B1-B1pair-C1'pair-C4'pair":2, "cWW_UU_alpha_1":2, "cWW_UU_alpha_2":2, "cWW UU dB1":2, "cWW UU dB2":1, 
    "tWW_UU_C1'-B1-B1pair":2, "tWW_UU_B1-B1pair-C1'pair":2, "tWW_UU_C4'-C1'-B1-B1pair":2, "tWW_UU_B1-B1pair-C1'pair-C4'pair":2, "tWW_UU_alpha_1":2, "tWW_UU_alpha_2":2, "tWW UU dB1":4, "tWW UU dB2":5, 
    "cWH_UU_C1'-B1-B1pair":2, "cWH_UU_B1-B1pair-C1'pair":2, "cWH_UU_C4'-C1'-B1-B1pair":3, "cWH_UU_B1-B1pair-C1'pair-C4'pair":3, "cWH_UU_alpha_1":2, "cWH_UU_alpha_2":3, "cWH UU dB1":3, "cWH UU dB2":5, 
    "tWH_UU_C1'-B1-B1pair":2, "tWH_UU_B1-B1pair-C1'pair":2, "tWH_UU_C4'-C1'-B1-B1pair":2, "tWH_UU_B1-B1pair-C1'pair-C4'pair":2, "tWH_UU_alpha_1":3, "tWH_UU_alpha_2":3, "tWH UU dB1":2, "tWH UU dB2":2, 
    "cHW_UU_C1'-B1-B1pair":2, "cHW_UU_B1-B1pair-C1'pair":3, "cHW_UU_C4'-C1'-B1-B1pair":1, "cHW_UU_B1-B1pair-C1'pair-C4'pair":3, "cHW_UU_alpha_1":1, "cHW_UU_alpha_2":2, "cHW UU dB1":3, "cHW UU dB2":4, 
    "tHW_UU_C1'-B1-B1pair":3, "tHW_UU_B1-B1pair-C1'pair":2, "tHW_UU_C4'-C1'-B1-B1pair":2, "tHW_UU_B1-B1pair-C1'pair-C4'pair":2, "tHW_UU_alpha_1":2, "tHW_UU_alpha_2":3, "tHW UU dB1":2, "tHW UU dB2":2, 
    "cWS_UU_C1'-B1-B1pair":1, "cWS_UU_B1-B1pair-C1'pair":1, "cWS_UU_C4'-C1'-B1-B1pair":2, "cWS_UU_B1-B1pair-C1'pair-C4'pair":3, "cWS_UU_alpha_1":2, "cWS_UU_alpha_2":1, "cWS UU dB1":2, "cWS UU dB2":1, 
    "tWS_UU_C1'-B1-B1pair":2, "tWS_UU_B1-B1pair-C1'pair":2, "tWS_UU_C4'-C1'-B1-B1pair":3, "tWS_UU_B1-B1pair-C1'pair-C4'pair":2, "tWS_UU_alpha_1":2, "tWS_UU_alpha_2":2, "tWS UU dB1":3, "tWS UU dB2":3, 
    "cSW_UU_C1'-B1-B1pair":1, "cSW_UU_B1-B1pair-C1'pair":3, "cSW_UU_C4'-C1'-B1-B1pair":2, "cSW_UU_B1-B1pair-C1'pair-C4'pair":3, "cSW_UU_alpha_1":2, "cSW_UU_alpha_2":3, "cSW UU dB1":1, "cSW UU dB2":4, 
    "tSW_UU_C1'-B1-B1pair":3, "tSW_UU_B1-B1pair-C1'pair":1, "tSW_UU_C4'-C1'-B1-B1pair":2, "tSW_UU_B1-B1pair-C1'pair-C4'pair":2, "tSW_UU_alpha_1":1, "tSW_UU_alpha_2":2, "tSW UU dB1":3, "tSW UU dB2":3, 
    "cHH_UU_C1'-B1-B1pair":1, "cHH_UU_B1-B1pair-C1'pair":1, "cHH_UU_C4'-C1'-B1-B1pair":3, "cHH_UU_B1-B1pair-C1'pair-C4'pair":2, "cHH_UU_alpha_1":2, "cHH_UU_alpha_2":2, "cHH UU dB1":1, "cHH UU dB2":5, 
    "tHH_UU_C1'-B1-B1pair":2, "tHH_UU_B1-B1pair-C1'pair":3, "tHH_UU_C4'-C1'-B1-B1pair":1, "tHH_UU_B1-B1pair-C1'pair-C4'pair":3, "tHH_UU_alpha_1":2, "tHH_UU_alpha_2":4, "tHH UU dB1":4, "tHH UU dB2":5, 
    "cSH_UU_C1'-B1-B1pair":1, "cSH_UU_B1-B1pair-C1'pair":3, "cSH_UU_C4'-C1'-B1-B1pair":2, "cSH_UU_B1-B1pair-C1'pair-C4'pair":2, "cSH_UU_alpha_1":3, "cSH_UU_alpha_2":2, "cSH UU dB1":2, "cSH UU dB2":5, 
    "tSH_UU_C1'-B1-B1pair":2, "tSH_UU_B1-B1pair-C1'pair":1, "tSH_UU_C4'-C1'-B1-B1pair":3, "tSH_UU_B1-B1pair-C1'pair-C4'pair":3, "tSH_UU_alpha_1":1, "tSH_UU_alpha_2":1, "tSH UU dB1":1, "tSH UU dB2":5, 
    "cHS_UU_C1'-B1-B1pair":2, "cHS_UU_B1-B1pair-C1'pair":2, "cHS_UU_C4'-C1'-B1-B1pair":2, "cHS_UU_B1-B1pair-C1'pair-C4'pair":2, "cHS_UU_alpha_1":2, "cHS_UU_alpha_2":2, "cHS UU dB1":3, "cHS UU dB2":2, 
    "tHS_UU_C1'-B1-B1pair":1, "tHS_UU_B1-B1pair-C1'pair":2, "tHS_UU_C4'-C1'-B1-B1pair":2, "tHS_UU_B1-B1pair-C1'pair-C4'pair":1, "tHS_UU_alpha_1":1, "tHS_UU_alpha_2":2, "tHS UU dB1":4, "tHS UU dB2":1, 
    "cSS_UU_C1'-B1-B1pair":2, "cSS_UU_B1-B1pair-C1'pair":2, "cSS_UU_C4'-C1'-B1-B1pair":2, "cSS_UU_B1-B1pair-C1'pair-C4'pair":3, "cSS_UU_alpha_1":2, "cSS_UU_alpha_2":2, "cSS UU dB1":6, "cSS UU dB2":4, 
    "tSS_UU_C1'-B1-B1pair":1, "tSS_UU_B1-B1pair-C1'pair":1, "tSS_UU_C4'-C1'-B1-B1pair":2, "tSS_UU_B1-B1pair-C1'pair-C4'pair":1, "tSS_UU_alpha_1":1, "tSS_UU_alpha_2":2, "tSS UU dB1":3, "tSS UU dB2":4, 
}  

def retrieve_angles(db): 
    """
    Retrieve torsion angles from RNANet.db and convert them to degrees
    """

    # Retrieve angle values
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, db)
    database = sqlite3.connect(db_path)
    cursor = database.cursor()
    cursor.execute("SELECT chain_id, nt_name, alpha, beta, gamma, delta, epsilon, zeta, chi FROM nucleotide WHERE nt_name='A' OR nt_name='C' OR nt_name='G' OR nt_name='U' ;")
    l = []
    for nt in cursor.fetchall(): # retrieve the angle measurements and put them in a list
        l.append(nt)

    # Convert to degrees
    angles_torsion = []
    for nt in l :
        angles_deg = []
        angles_deg.append(nt[0]) #chain_id
        angles_deg.append(nt[1]) #nt_name
        for i in range (2,9): # on all angles
            angle = 0
            if nt[i] == None : 
                angle = None
            elif nt[i]<=np.pi: #if angle value <pi, positive
                angle = (180/np.pi)*nt[i]
            elif np.pi < nt[i] <= 2*np.pi : #if value of the angle between pi and 2pi, negative
                angle = ((180/np.pi)*nt[i]) - 360
            else:
                angle = nt[i] # in case some angles still in degrees
            angles_deg.append(angle)
        angles_torsion.append(angles_deg)
    return angles_torsion

def retrieve_eta_theta(db):
    """
    Retrieve pseudotorsions from RNANet.db and convert them to degrees
    """
    # Retrieve angle values
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, db)
    database = sqlite3.connect(db_path)
    cursor = database.cursor()
    cursor.execute("SELECT chain_id, nt_name, eta, theta, eta_prime, theta_prime, eta_base, theta_base FROM nucleotide WHERE nt_name='A' OR nt_name='C' OR nt_name='G' OR nt_name='U';")
    l = []
    for nt in cursor.fetchall(): 
        l.append(nt)

    # convert to degrees
    pseudotorsions=[]
    for nt in l :
        angles_deg = []
        angles_deg.append(nt[0]) #chain_id
        angles_deg.append(nt[1]) #nt_name
        for i in range (2,8): 
            angle = 0
            if nt[i] == None : 
                angle=None
            elif nt[i]<=np.pi:
                angle = (180/np.pi)*nt[i]
            elif np.pi < nt[i] <= 2*np.pi : 
                angle = ((180/np.pi)*nt[i]) - 360
            else:
                angle = nt[i] 
            angles_deg.append(angle)
        pseudotorsions.append(angles_deg)
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

def measure_from_structure(f):
    """
    Do geometric measures required on a given filename
    """

    name = f.split('.')[0]

    global idxQueue
    thr_idx = idxQueue.get()
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} measure_from_structure({f})")

    # Open the structure 
    with warnings.catch_warnings():
        # Ignore the PDB problems. This mostly warns that some chain is discontinuous.
        warnings.simplefilter('ignore', Bio.PDB.PDBExceptions.PDBConstructionWarning)
        warnings.simplefilter('ignore', Bio.PDB.PDBExceptions.BiopythonWarning)
        parser = MMCIFParser()
        s = parser.get_structure(f, os.path.abspath(path_to_3D_data+ "rna_only/" + f))
    
    #pyle_measures(name, s, thr_idx)
    measures_aa(name, s, thr_idx)
    if DO_HIRE_RNA_MEASURES:
        measures_hrna(name, s, thr_idx)
        measures_hrna_basepairs(name, s, thr_idx)
    if DO_WADLEY_ANALYSIS:
        measures_pyle(name, s, thr_idx)
    
    idxQueue.put(thr_idx) # replace the thread index in the queue
    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} finished")

@trace_unhandled_exceptions
def measures_aa(name, s, thr_idx):
    """
    Measures the distance between atoms linked by covalent bonds
    """

    # do not recompute something already computed
    if os.isfile(runDir + "/results/geometry/all-atoms/distances/dist_atoms_" + name + ".csv"):
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

    df_comm = pd.DataFrame(l_common, columns=["Residu", "O3'-P", "P-OP1", "P-OP2", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-O4'", "C4'-C3'", "O4'-C1'", "C1'-C2'", "C2'-O2'", "C2'-C3'", "C3'-O3'"])
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

    df = pd.DataFrame(l_dist, columns=["Residu", "C1'-P", "P-C1'", "C4'-P", "P-C4'"])
    df.to_csv(runDir + "/results/geometry/Pyle/distances/distances_pyle_" + name + ".csv")
    df = pd.DataFrame(l_angl, columns=["Residu", "P-C1'-P°", "C1'-P°-C1'°"])
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

    l_dist=[]
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
            atom_p = [ atom.get_coord() for atom in res if atom.get_name() ==  "P"]
            atom_o5p= [ atom.get_coord() for atom in res if "O5'" in atom.get_fullname() ]
            atom_c5p = [ atom.get_coord() for atom in res if "C5'" in atom.get_fullname() ]
            atom_c4p = [ atom.get_coord() for atom in res if "C4'" in atom.get_fullname() ]
            atom_c1p = [ atom.get_coord() for atom in res if "C1'" in atom.get_fullname() ]
            atom_b1 = pos_b1(res) # position b1 to be calculated, depending on the case
            atom_b2 = pos_b2(res) # position b2 to be calculated only for those with 2 cycles

            # Distances. If one of the atoms is empty, the euclidian distance returns NaN.
            last_c4p_p = get_euclidian_distance(last_c4p, atom_p)
            p_o5p = get_euclidian_distance(atom_p, atom_o5p)
            o5p_c5p = get_euclidian_distance(atom_o5p, atom_c5p)
            c5p_c4p = get_euclidian_distance(atom_c5p, atom_c4p)
            c4p_c1p = get_euclidian_distance(atom_c4p, atom_c1p)
            c1p_b1 = get_euclidian_distance(atom_c1p, atom_b1)
            b1_b2 = get_euclidian_distance(atom_b1, atom_b2)

            # flat angles. Same.
            lastc4p_p_o5p = get_flat_angle(last_c4p, atom_p, atom_o5p)
            lastc1p_lastc4p_p = get_flat_angle(last_c1p, last_c4p, atom_p)
            lastc5p_lastc4p_p = get_flat_angle(last_c5p, last_c4p, atom_p)
            p_o5p_c5p = get_flat_angle(atom_p, atom_o5p, atom_c5p)
            o5p_c5p_c4p = get_flat_angle(atom_o5p, atom_c5p, atom_c4p)
            c5p_c4p_c1p = get_flat_angle(atom_c5p, atom_c4p, atom_c1p)
            c4p_c1p_b1 = get_flat_angle(atom_c4p, atom_c1p, atom_b1)
            c1p_b1_b2 = get_flat_angle(atom_c1p, atom_b1, atom_b2)

            # torsions. Idem.
            p_o5_c5_c4 = get_torsion_angle(atom_p, atom_o5p, atom_c5p, atom_c4p)
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
    df = pd.DataFrame(l_dist, columns=["Residu", "C4'-P", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-C1'", "C1'-B1", "B1-B2"])
    df.to_csv(runDir + '/results/geometry/HiRE-RNA/distances/distances_HiRERNA '+name+'.csv')
    df = pd.DataFrame(l_angl, columns=["Residu", "C4'-P-O5'", "C1'-C4'-P", "C5'-C4'-P", "P-O5'-C5'", "O5'-C5'-C4'", "C5'-C4'-C1'", "C4'-C1'-B1", "C1'-B1-B2"])
    df.to_csv(runDir + '/results/geometry/HiRE-RNA/angles/angles_HiRERNA ' + name + ".csv")
    df=pd.DataFrame(l_tors, columns=["Residu", "P-O5'-C5'-C4'", "O5'-C5'-C4'-C1'", "C5'-C4'-C1'-B1", "C4'-C1'-B1-B2", "O5'-C5'-C4'-P°", "C5'-C4'-P°-O5'°", "C4'-P°-O5'°-C5'°", "C1'-C4'-P°-O5'°"])
    df.to_csv(runDir + '/results/geometry/HiRE-RNA/torsions/torsions_HiRERNA '+name+'.csv')

@trace_unhandled_exceptions
def measures_hrna_basepairs(name, s, thr_idx):
    """
    Open a rna_only/ file, and run measures_hrna_basepairs_chain() on every chain
    """  

    setproctitle(f"RNANet statistics.py Worker {thr_idx+1} measures_hrna_basepairs({name})")
    
    l=[]
    chain = next(s[0].get_chains())
            
    # do not recompute something already computed
    if os.path.isfile(runDir + "/results/geometry/HiRE-RNA/basepairs/basepairs_"+name+".csv"):
        return

    df=pd.read_csv(os.path.abspath(path_to_3D_data +"datapoints/" + name))

    if df['index_chain'][0] == 1: # ignore files with numbering errors : TODO : remove when we get DSSR Pro, there should not be numbering errors anymore
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
    # a = calc_angle(a1_res, a2_res, a3_res)*(180/np.pi)    # not required
    # b = calc_angle(a2_res, a1_res, a1_pair)*(180/np.pi)
    # c = calc_angle(a1_res, a1_pair, a2_pair)*(180/np.pi)
    # d = calc_angle(a3_pair, a2_pair, a1_pair)*(180/np.pi)   # not required

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
            gmm = GaussianMixture(n_components=n_comp).fit(md)
            # aic.append(abs(gmm.aic(md)))
            # bic.append(abs(gmm.bic(md)))
            maxlogv.append(gmm.lower_bound_)
            if gmm.lower_bound_== max(maxlogv) : # takes the maximum
                nb_components = n_comp
                # if there is convergence, keep the first maximum found
                if abs(gmm.lower_bound_-log_max) < 0.02 : #threshold=0.02
                    nb_components = nb_log_max
                    break
            log_max = max(maxlogv)
            nb_log_max = n_comp
    else:
        try:
            n_components = data[name_data]
        except KeyError:
            return # unexpected atom ? skip it...

    
    # Now compute the final GMM
    obs = np.array(data).reshape(-1,1) # still on extended data
    g = GaussianMixture(n_components=nb_components)
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
        try:
            plt.plot(newx, sum(curves), c=col, label=name_data)
        except TypeError:
            print("N curves:", len(curves))
            for c in curves:
                print(c)
        plt.legend()

        # Save the json
        with open(runDir + "/results/geometry/json/" +name_data + ".json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)

@trace_unhandled_exceptions
def gmm_aa_dists(scan):
    """
    Draw the figures representing the data on the measurements of distances between atoms
    """

    setproctitle("GMM (all atoms, distances)")

    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/all-atoms/distances/dist_atoms.csv"))

    last_o3p_p = list(df["O3'-P"][~ np.isnan(df["O3'-P"])])
    # op3_p = list(df["OP3-P"][~ np.isnan(df["OP3-P"])])
    p_op1 = list(df["P-OP1"][~ np.isnan(df["P-OP1"])])
    p_op2 = list(df["P-OP2"][~ np.isnan(df["P-OP2"])])
    p_o5p = list(df["P-O5'"][~ np.isnan(df["P-O5'"])])
    o5p_c5p = list(df["O5'-C5'"][~ np.isnan(df["O5'-C5'"])])
    c5p_c4p = list(df["C5'-C4'"][~ np.isnan(df["C5'-C4'"])])
    c4p_o4p = list(df["C4'-O4'"][~ np.isnan(df["C4'-O4'"])])
    o4p_c1p = list(df["O4'-C1'"][~ np.isnan(df["O4'-C1'"])])
    c1p_c2p = list(df["C1'-C2'"][~ np.isnan(df["C1'-C2'"])])
    c2p_o2p = list(df["C2'-O2'"][~ np.isnan(df["C2'-O2'"])])
    c2p_c3p = list(df["C2'-C3'"][~ np.isnan(df["C2'-C3'"])])
    c3p_o3p = list(df["C3'-O3'"][~ np.isnan(df["C3'-O3'"])])
    c4p_c3p = list(df["C4'-C3'"][~ np.isnan(df["C4'-C3'"])])
    
    #if res = A ou G
    c1p_n9 = list(df["C1'-N9"][~ np.isnan(df["C1'-N9"])])
    n9_c8 = list(df["N9-C8"][~ np.isnan(df["N9-C8"])])
    c8_n7 = list(df["C8-N7"][~ np.isnan(df["C8-N7"])])
    n7_c5 = list(df["N7-C5"][~ np.isnan(df["N7-C5"])])
    c5_c6 = list(df["C5-C6"][~ np.isnan(df["C5-C6"])])
    c6_n1 = list(df["C6-N1"][~ np.isnan(df["C6-N1"])])
    n1_c2 = list(df["N1-C2"][~ np.isnan(df["N1-C2"])])
    c2_n3 = list(df["C2-N3"][~ np.isnan(df["C2-N3"])])
    n3_c4 = list(df["N3-C4"][~ np.isnan(df["N3-C4"])])
    c4_n9 = list(df["C4-N9"][~ np.isnan(df["C4-N9"])])
    c4_c5 = list(df["C4-C5"][~ np.isnan(df["C4-C5"])])
    #if res = G
    c6_o6 = list(df["C6-O6"][~ np.isnan(df["C6-O6"])])
    c2_n2 = list(df["C2-N2"][~ np.isnan(df["C2-N2"])])
    #if res = A
    c6_n6 = list(df["C6-N6"][~ np.isnan(df["C6-N6"])])
    #if res = C ou U
    c1p_n1 = list(df["C1'-N1"][~ np.isnan(df["C1'-N1"])])
    n1_c6 = list(df["N1-C6"][~ np.isnan(df["N1-C6"])])
    c6_c5 = list(df["C6-C5"][~ np.isnan(df["C6-C5"])])
    c5_c4 = list(df["C5-C4"][~ np.isnan(df["C5-C4"])])
    c4_n3 = list(df["C4-N3"][~ np.isnan(df["C4-N3"])])
    n3_c2 = list(df["N3-C2"][~ np.isnan(df["N3-C2"])])
    c2_n1 = list(df["C2-N1"][~ np.isnan(df["C2-N1"])])
    c2_o2 = list(df["C2-O2"][~ np.isnan(df["C2-O2"])])
    #if res =C
    c4_n4 = list(df["C4-N4"][~ np.isnan(df["C4-N4"])])
    #if res=U
    c4_o4 = list(df["C4-O4"][~ np.isnan(df["C4-O4"])])

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
def gmm_aa_torsions(scan):
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
    for angles_deg in retrieve_angles(runDir + "/results/RNANet.db"): 
        alpha.append(angles_deg[2])
        beta.append(angles_deg[3])
        gamma.append(angles_deg[4])
        delta.append(angles_deg[5])
        epsilon.append(angles_deg[6])
        zeta.append(angles_deg[7])
        chi.append(angles_deg[8])

    # we remove the null values
    alpha = [i for i in alpha if i != None]
    beta = [i for i in beta if i != None]
    gamma = [i for i in gamma if i != None]
    delta = [i for i in delta if i != None]
    epsilon = [i for i in epsilon if i != None]
    zeta = [i for i in zeta if i != None]
    chi = [i for i in chi if i != None]

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
def gmm_pyle(scan):

    setproctitle("GMM (Pyle model)")

    # Distances
    df = pd.read_csv(os.path.abspath(runDir + "/results/geometry/Pyle/distances/distances_pyle.csv"))  

    p_c1p = list(df["C1'-P"][~ np.isnan(df["C1'-P"])])
    c1p_p = list(df["P-C1'"][~ np.isnan(df["P-C1'"])])
    p_c4p = list(df["C4'-P"][~ np.isnan(df["C4'-P"])])
    c4p_p = list(df["P-C4'"][~ np.isnan(df["P-C4'"])])

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

    for angles_deg in retrieve_eta_theta(runDir + "/results/RNANet.db"): 
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
    GMM_histo(data["Distance"], f"Distance between {type_LW} {ntpair} tips", scan, toric=False, hist=False, col="cyan")
    GMM_histo(data["dB1"], f"{type_LW} {ntpair} dB1", scan, toric=False, hist=False, col="tomato")
    GMM_histo(data["dB2"], f"{type_LW} {ntpair} dB2", scan, toric=False, hist=False, col="goldenrod")
    plt.xlabel("Distance (Angströms)")
    plt.title(f"GMM of distances for {type_LW} {ntpair} basepairs", fontsize=10)
    
    plt.savefig(f"{type_LW}_{ntpair}_basepairs.png" )
    plt.close()
    setproctitle(f"GMM (HiRE-RNA {type_LW} {ntpair} basepairs) finished")

def merge_jsons():
    """
    Reads the tons of JSON files produced by the geometric analyses, and compiles them into fewer files.
    It is simple concatenation of the JSONs.
    The original files are then deleted.
    """

    # All atom distances
    bonds = ["O3'-P", "OP3-P", "P-OP1", "P-OP2", "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-O4'", "C4'-C3'", "O4'-C1'", "C1'-C2'", "C2'-O2'", "C2'-C3'", "C3'-O3'", "C1'-N9",
             "N9-C8", "C8-N7", "N7-C5", "C5-C6", "C6-O6", "C6-N6", "C6-N1", "N1-C2", "C2-N2", "C2-N3", "N3-C4", "C4-N9", "C4-C5", 
             "C1'-N1", "N1-C6", "C6-C5", "C5-C4", "C4-N3", "N3-C2", "C2-O2", "C2-N1", "C4-N4", "C4-O4"]
    bonds = [ runDir + "/results/geometry/json/" + x + ".json" for x in bonds ]
    concat_jsons(bonds, runDir + "/results/geometry/json/all_atom_distances.json")
    

    # All atom torsions
    torsions = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Xhi", "Zeta"]
    torsions = [ runDir + "/results/geometry/json/" + x + ".json" for x in torsions ]
    concat_jsons(torsions, runDir + "/results/geometry/json/all_atom_torsions.json")
 
    # HiRE-RNA distances
    hrnabonds = ["P-O5'", "O5'-C5'", "C5'-C4'", "C4'-C1'", "C1'-B1", "B1-B2", "C4'-P"]
    hrnabonds = [ runDir + "/results/geometry/json/" + x + ".json" for x in hrnabonds ]
    concat_jsons(hrnabonds, runDir + "/results/geometry/json/hirerna_distances.json")

    # HiRE-RNA angles
    hrnaangles = ["P-O5'-C5'", "O5'-C5'-C4'", "C5'-C4'-C1'", "C4'-C1'-B1", "C1'-B1-B2", "C4'-P-O5'", "C5'-C4'-P", "C1'-C4'-P"]
    hrnaangles = [ runDir + "/results/geometry/json/" + x + ".json" for x in hrnaangles ]
    concat_jsons(hrnaangles, runDir + "/results/geometry/json/hirerna_angles.json")

    # HiRE-RNA torsions
    hrnators = ["P-O5'-C5'-C4'", "O5'-C5'-C4'-C1'", "C5'-C4'-C1'-B1", "C4'-C1'-B1-B2", "C4'-P°-O5'°-C5'°", "C5'-C4'-P°-O5'°", "C1'-C4'-P°-O5'°", "O5'-C5'-C4'-P°"]
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
    for f in glob.glob(runDir + "/results/geometry/json/Distance*.json"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    return pd.read_csv(f)

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
def concat_dataframes(fpath, outfilename):
    """
    Concatenates the CSV files from fpath folder into a DataFrame gathering all.
    The function splits the file list into nworkers concatenation workers, and then merges the nworkers dataframes.
    """
    global nworkers
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
    p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=nworkers)
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