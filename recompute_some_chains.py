#!python3
import subprocess, os, sys

# Put a list of problematic chains here, they will be properly deleted and recomputed
problems = [
"4v7f_1_1_4-3396",
"4v7f_1_1_1-3167",
"4v7f_1_1_1-3255",
"6g90_1_1_1-407",
"3q3z_1_A_1-74",
"3q3z_1_V_1-74",
"4v7f_1_3_1-121"
]

path_to_3D_data = sys.argv[1]
path_to_seq_data = sys.argv[2]

for p in problems:
    print()
    print()
    print()
    print()

    # Remove the datapoints files and 3D files
    subprocess.run(["rm", '-f', path_to_3D_data + f"/rna_mapped_to_Rfam/{p}.cif"])
    files = [ f for f in os.listdir(path_to_3D_data + "/datapoints") if p in f ]
    for f in files:
        subprocess.run(["rm", '-f', path_to_3D_data + f"/datapoints/{f}"])

    # Find more information
    structure = p.split('_')[0]
    chain = p.split('_')[2]
    families = [ f.split('.')[1] for f in files ] # The RFAM families this chain has been mapped onto

    # Delete the chain from the database, and the associated nucleotides and re_mappings, using foreign keys
    for fam in families:
        command = ["sqlite3", "results/RNANet.db", f"PRAGMA foreign_keys=ON; delete from chain where structure_id=\"{structure}\" and chain_name=\"{chain}\" and rfam_acc=\"{fam}\";"]
        print(' '.join(command))
        subprocess.run(command)

    # Re-run RNANet
    command = ["python3.8", "RNAnet.py", "--3d-folder", path_to_3D_data, "--seq-folder", path_to_seq_data, "-r", "20.0", "--extract", "--only", p]
    print('\n',' '.join(command),'\n')
    subprocess.run(command)

# run statistics
