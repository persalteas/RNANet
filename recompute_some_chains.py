#!python3
import subprocess, os, sys

# Put a list of problematic chains here, they will be properly deleted and recomputed
problems = [
    "1k73_1_A",
    "1k73_1_B"
]

path_to_3D_data = sys.argv[1]
path_to_seq_data = sys.argv[2]

for p in problems:
    print()
    print()
    print()
    print()
    homology = ('-' in p)

    # Remove the datapoints files and 3D files
    subprocess.run(["rm", '-f', path_to_3D_data + f"/rna_mapped_to_Rfam/{p}.cif"])
    files = [ f for f in os.listdir(path_to_3D_data + "/datapoints") if p in f ]
    for f in files:
        subprocess.run(["rm", '-f', path_to_3D_data + f"/datapoints/{f}"])

    # Find more information
    structure = p.split('_')[0]
    chain = p.split('_')[2]
    if homology:
        families = [ f.split('.')[1] for f in files ] # The RFAM families this chain has been mapped onto

        # Delete the chain from the database, and the associated nucleotides and re_mappings, using foreign keys
        for fam in families:
            command = ["sqlite3", "results/RNANet.db", f"PRAGMA foreign_keys=ON; delete from chain where structure_id=\"{structure}\" and chain_name=\"{chain}\" and rfam_acc=\"{fam}\";"]
            print(' '.join(command))
            subprocess.run(command)

        command = ["python3.8", "RNAnet.py", "--3d-folder", path_to_3D_data, "--seq-folder", path_to_seq_data, "-r", "20.0", "--extract", "--only", p]
    else:
        # Delete the chain from the database, and the associated nucleotides and re_mappings, using foreign keys
        command = ["sqlite3", "results/RNANet.db", f"PRAGMA foreign_keys=ON; delete from chain where structure_id=\"{structure}\" and chain_name=\"{chain}\" and rfam_acc is null;"]
        print(' '.join(command))
        subprocess.run(command)

        command = ["python3.8", "RNAnet.py", "--3d-folder", path_to_3D_data, "--seq-folder", path_to_seq_data, "-r", "20.0", "--no-homology", "--extract", "--only", p]

    # Re-run RNANet
    print('\n',' '.join(command),'\n')
    subprocess.run(command)

# run statistics
