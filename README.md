# RNANet
Building a dataset following the ProteinNet philosophy, but for RNA.

We use the Rfam mappings between 3D structures and known Rfam families, using the sequences that are known to belong to an Rfam family (hits provided in RF0XXXX.fasta files from Rfam).

Future versions might compute a real MSA-based clusering directly with Rfamseq ncRNA sequences, like ProteinNet does with protein sequences, but this requires a tool similar to jackHMMER in the Infernal software suite, which is not available yet.

This script prepares the dataset from available public data in PDB and Rfam.
It requires solid hardware to run. (Tested on a server with 32 cores and 48GB of RAM.)

# Dependencies
You need to install Infernal, DSSR, and SINA before running this.
I moved to python3.8.1. Unfortunately, python3.6 is no longer supported, because of changes in the multiprocessing and Threading packages. Untested with Python 3.7.*.

Packages numpy, pandas, matplotlib, requests, psutil, biopython, sqlalchemy and tqdm are required.
`python3.8 -m pip install numpy pandas matplotlib pymysql requests psutil biopython sqlalchemy tqdm`

Before use, please set the two variables `path_to_3D_data` and `path_to_seq_data` (around line 30 of RNAnet.py) to two folders where you want to store RNA 3D structures and RNA sequences. A few gigabytes will be produced.

# What it does
The script follows these steps:
* Gets a list of 3D structures containing RNA from BGSU's non-redundant list (but keeps the redundant structures /!\\),
* Asks Rfam for mappings of these structures onto Rfam families (~ a half of structures have a mapping)
* Downloads the corresponding 3D structures (mmCIFs)
* Extracts the right chain portions that map onto an Rfam family

Now, compute the features:

* Extract the sequence for every 3D chain
* Downloads Rfamseq ncRNA sequence hits for the concerned Rfam families
* Realigns Rfamseq hits and sequences from the 3D structures together to obtain a multiple sequence alignment for each Rfam family (using cmalign, except for ribosomal LSU and SSU, where SINA is used)
* Computes nucleotide frequencies at every position for each alignment
* For each aligned 3D chain, get the nucleotide frequencies in the corresponding RNA family for each residue

Then, compute the labels:

* Run DSSR on every chain to get a variety of descriptors per position, describing secondary and tertiary structure
* This also permits to identify missing residues and compute a mask for every chain.

Finally, store this data into files.

# Dataset quality
The file statistics.py is supposed to give a summary on the produced dataset. See the results/ folder.

# Contact
louis.becquey@univ-evry.fr
