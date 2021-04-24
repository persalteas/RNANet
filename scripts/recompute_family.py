#!python3
import subprocess, os, sys
from RNAnet import *


# Put a list of problematic families here, they will be properly deleted and recomputed
families = [
    "RF00005"
]

# provide the path to your data folders, the RNANet.db file, and the RNANet.py file as arguments to this script
path_to_3D_data = "/home/lbecquey/Data/RNA/3D/"
path_to_seq_data = "/home/lbecquey/Data/RNA/sequences/"
path_to_db = "/home/lbecquey/Projects/RNANet/results/RNANet.db"

for fam in families:
    print()
    print()
    print()
    print(f"Removing {fam} files...")

    # Remove the datapoints files
    files = [ f for f in os.listdir(path_to_3D_data + "/datapoints") if fam in f ]
    for f in files:
        subprocess.run(["rm", '-f', path_to_3D_data + f"/datapoints/{f}"])

    # Remove the alignments
    files = [ f for f in os.listdir(path_to_seq_data + "/realigned") if fam in f ]
    for f in files:
        subprocess.run(["rm", '-f', path_to_seq_data + f"/realigned/{f}"])

    # Delete the family from the database, and the associated nucleotides and re_mappings, using foreign keys
    command = ["sqlite3", path_to_db, f"PRAGMA foreign_keys=ON; delete from family where rfam_acc=\"{fam}\";"]
    print(' '.join(command))
    subprocess.run(command)

# Now re run RNANet normally.
command = ["python3.8", "./RNAnet.py", "--3d-folder", path_to_3D_data, "--seq-folder", path_to_seq_data, "-r", "20.0",
            "--redundant", "--sina", "--extract", "-s", "--stats-opts=\"--wadley --distance-matrices\""]
print(' '.join(command))
subprocess.run(command)