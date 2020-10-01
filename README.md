# RNANet
Building a dataset following the ProteinNet philosophy, but for RNA.

We use the Rfam mappings between 3D structures and known Rfam families, using the sequences that are known to belong to an Rfam family (hits provided in RF0XXXX.fasta files from Rfam).
Future versions might compute a real MSA-based clusering directly with Rfamseq ncRNA sequences, like ProteinNet does with protein sequences, but this requires a tool similar to jackHMMER in the Infernal software suite, which is not available yet.

This script prepares the dataset from available public data in PDB and Rfam.

Contents:
* [What it does](#what-it-does)
* [Output files](#output-files)
* [How to run](#how-to-run)
    * [Required computational resources](#required-computational-resources)
    * [Using Docker](#using-docker)
    * [Using classical command line installation](#using-classical-command-line-installation)
    * [Post-computation task: estimate quality](#post-computation-task:-estimate-quality)
* [How to further filter the dataset](#how-to-further-filter-the-dataset)
    * [Filter on 3D structure resolution](#filter-on-3D-structure-resolution)
    * [Filter on 3D structure publication date](#filter-on-3d-structure-publication-date)
    * [Filter to avoid chain redundancy when several mappings are available](#filter-to-avoid-chain-redundancy-when-several-mappings-are-available)
* [More about the database structure](#more-about-the-database-structure)
* [Troubleshooting](#troubleshooting)
* [Contact](#contact)

**Please cite**: *Coming soon, expect it in 2021*

# What it does
The script follows these steps:
* Gets a list of 3D structures containing RNA from BGSU's non-redundant list (but keeps the redundant structures /!\\),
* Asks Rfam for mappings of these structures onto Rfam families (~ a half of structures have a direct mapping, some more are inferred using the redundancy list)
* Downloads the corresponding 3D structures (mmCIFs)
* If desired, extracts the right chain portions that map onto an Rfam family

Now, compute the features:

* Extract the sequence for every 3D chain
* Downloads Rfamseq ncRNA sequence hits for the concerned Rfam families
* Realigns Rfamseq hits and sequences from the 3D structures together to obtain a multiple sequence alignment for each Rfam family (using cmalign, except for ribosomal LSU and SSU, where SINA is used)
* Computes nucleotide frequencies at every position for each alignment
* For each aligned 3D chain, get the nucleotide frequencies in the corresponding RNA family for each residue

Then, compute the labels:

* Run DSSR on every RNA structure to get a variety of descriptors per position, describing secondary and tertiary structure. Basepair types annotations include intra-chain and inter-chain interactions.

Finally, export this data from the SQLite database into flat CSV files.

# Output files

* `results/RNANet.db` is a SQLite database file containing several tables with all the information, which you can query yourself with your custom requests,
* `3D-folder-you-passed-in-option/datapoints/*` are flat text CSV files, one for one RNA chain mapped to one RNA family, gathering the per-position nucleotide descriptors,
* `results/RNANET_datapoints_latest.tar.gz` is a compressed archive of the above CSV files (only if you passed the --archive option)
* `path-to-3D-folder-you-passed-in-option/rna_mapped_to_Rfam` If you used the --extract option, this folder contains one mmCIF file per RNA chain mapped to one RNA family, without other chains, proteins (nor ions and ligands by default)
* `results/summary_latest.csv` summarizes information about the RNA chains
* `results/families_latest.csv` summarizes information about the RNA families

If you launch successive executions of RNANet, the previous tar.gz archive and the two summary CSV files are stored in the `results/archive/` folder.

Other folders are created and not deleted, which you might want to conserve to avoid re-computations in later runs:

* `path-to-sequence-folder-you-passed-in-option/rfam_sequences/fasta/` contains compressed FASTA files of the homologous sequences used, by Rfam family.
* `path-to-sequence-folder-you-passed-in-option/realigned/` contains families covariance models (\*.cm), unaligned list of sequences (\*.fa), and multiple sequence alignments in both formats Stockholm and Aligned-FASTA (\*.stk and \*.afa). Also contains SINA homolgous sequences databases LSU.arb and SSU.arb, and their index files (\*.sidx).
* `path-to-3D-folder-you-passed-in-option/RNAcifs/` contains mmCIF structures directly downloaded from the PDB, which contain RNA chains,
* `path-to-3D-folder-you-passed-in-option/annotations/` contains the raw JSON annotation files of the previous mmCIF structures. You may find additional information into them which is not properly supported by RNANet yet.

# How to run (on Linux x86-64 only)

## Required computational resources
- CPU: no requirements. The program is optimized for multi-core CPUs, you might want to use Intel Xeons, AMD Ryzens, etc.
- GPU: not required
- RAM: 16 GB with a large swap partition is okay. 32 GB is recommended (usage peaks at ~27 GB)
- Storage: to date, it takes 60 GB for the 3D data (36 GB if you don't use the --extract option), 11 GB for the sequence data, and 7GB for the outputs (5.6 GB database, 1 GB archive of CSV files). You need to add a few more for the dependencies. Pick a 100GB partition and you are good to go. The computation speed is way better if you use a fast storage device (e.g. SSD instead of hard drive, or even better, a NVMe SSD) because of constant I/O with the SQlite database.
- Network : We query the Rfam public MySQL server on port 4497. Make sure your network enables communication (there should not be any issue on private networks, but maybe you company/university closes ports by default). You will get an error message if the port is not open. Around 30 GB of data is downloaded.

To give you an estimation, our last full run took exactly 12h, excluding the time to download the MMCIF files containing RNA (around 25GB to download) and the time to compute statistics.
Measured the 23rd of June 2020 on a 16-core AMD Ryzen 7 3700X CPU @3.60GHz, plus 32 Go RAM, and a 7200rpm Hard drive. Total CPU time spent: 135 hours (user+kernel modes), corresponding to 12h (actual time spent with the 16-core CPU). 

Update runs are much quicker, around 3 hours. It depends mostly on what RNA families are concerned by the update.

## Using Docker

* Step 1 : Download the [Docker container](https://entrepot.ibisc.univ-evry.fr/f/e5edece989884a7294a6/?dl=1). Open a terminal and move to the appropriate directory.
* Step 2 : Extract the archive to a Docker image named *rnanet* in your local installation
```
$ docker image import rnanet_v1.2_docker.tar rnanet
```
* Step 3 : Run the container, giving it 3 folders to mount as volumes: a first to store the 3D data, a second to store the sequence data and alignments, and a third to output the results, data and logs:
```
$ docker run --rm -v path/to/3D/data/folder:/3D -v path/to/sequence/data/folder:/sequences -v path/to/experiment/results/folder:/runDir rnanet [ - other options ]
```

The detailed list of options is below:

```
-h [ --help ]                   Print this help message
--version                       Print the program version

-f [ --full-inference ]         Infer new 3D->family mappings even if Rfam already provides some. Yields more copies of chains
                                mapped to different families.
-r 4.0 [ --resolution=4.0 ]     Maximum 3D structure resolution to consider a RNA chain.
-s                              Run statistics computations after completion
--extract                       Extract the portions of 3D RNA chains to individual mmCIF files.
--keep-hetatm=False             (True | False) Keep ions, waters and ligands in produced mmCIF files. 
                                Does not affect the descriptors.
--fill-gaps=True                (True | False) Replace gaps in nt_align_code field due to unresolved residues
                                by the most common nucleotide at this position in the alignment.
--3d-folder=…                   Path to a folder to store the 3D data files. Subfolders will contain:
                                        RNAcifs/                Full structures containing RNA, in mmCIF format
                                        rna_mapped_to_Rfam/     Extracted 'pure' RNA chains
                                        datapoints/             Final results in CSV file format.
--seq-folder=…                  Path to a folder to store the sequence and alignment files. Subfolders will be:
                                        rfam_sequences/fasta/   Compressed hits to Rfam families
                                        realigned/              Sequences, covariance models, and alignments by family
--no-homology                   Do not try to compute PSSMs and do not align sequences.
                                Allows to yield more 3D data (consider chains without a Rfam mapping).

--all                           Build chains even if they already are in the database.
--only                          Ask to process a specific chain label only
--ignore-issues                 Do not ignore already known issues and attempt to compute them
--update-homologous             Re-download Rfam and SILVA databases, realign all families, and recompute all CSV files
--from-scratch                  Delete database, local 3D and sequence files, and known issues, and recompute.
--archive                       Create a tar.gz archive of the datapoints text files, and update the link to the latest archive
--no-logs                       Do not save per-chain logs of the numbering modifications
```
You may not use the --3d-folder and --seq-folder options, they are set by default to the paths you provide with the -v options when running Docker.

Typical usage:
```
nohup bash -c 'time docker run --rm -v /path/to/3D/data/folder:/3D -v /path/to/sequence/data/folder:/sequences -v /path/to/experiment/folder:/runDir rnanet -s --no-logs ' &
```

## Using classical command line installation

You need to install the dependencies:
- DSSR, you need to register to the X3DNA forum [here](http://forum.x3dna.org/site-announcements/download-instructions/) and then download the DSSR binary [on that page](http://forum.x3dna.org/downloads/3dna-download/).  Make sure to have the `x3dna-dssr` binary in your $PATH variable so that RNANet.py finds it.
- Infernal, to download at [Eddylab](http://eddylab.org/infernal/), several options are available depending on your preferences. Make sure to have the `cmalign`, `esl-alimanip`, `esl-alipid` and `esl-reformat` binaries in your $PATH variable, so that RNANet.py can find them.
- SINA, follow [these instructions](https://sina.readthedocs.io/en/latest/install.html) for example. Make sure to have the `sina` binary in your $PATH.
- Sqlite 3, available under the name *sqlite* in every distro's package manager,
- Python >= 3.8, (Unfortunately, python3.6 is no longer supported, because of changes in the multiprocessing and Threading packages. Untested with Python 3.7.\*)
- The following Python packages: `python3.8 -m pip install biopython==1.76 matplotlib pandas psutil pymysql requests scipy setproctitle sqlalchemy tqdm`. Note that Biopython versions 1.77 or later do not work (yet) since they removed the alphabet system.

Then, run it from the command line, preferably using nohup if your shell will be interrupted:
```
 ./RNANet.py --3d-folder path/to/3D/data/folder --seq-folder path/to/sequence/data/folder [ - other options ]
```
See the list of possible options juste above in the [Using Docker](#using-docker) section. Expect hours (maybe days) of computation.

Typical usage:
```
nohup bash -c 'time ~/Projects/RNANet/RNAnet.py --3d-folder ~/Data/RNA/3D/ --seq-folder ~/Data/RNA/sequences --no-logs -s' &
```

## Post-computation task: estimate quality
If your did not ask for automatic run of statistics over the produced dataset with the `-s` option, you can run them later using the file statistics.py. 
```
python3.8 statistics.py --3d-folder path/to/3D/data/folder --seq-folder path/to/sequence/data/folder -r 20.0
```
/!\ Beware, if not precised with option `-r`, no resolution threshold is applied and all the data in RNANet.db is used.

If you have run RNANet twice, once with option `--no-homology`, and once without, you unlock new statistics over unmapped chains. You will also be allowed to use option `--wadley` to reproduce Wadley & al. (2007) results automatically.

# How to further filter the dataset
You may want to build your own sub-dataset by querying the results/RNANet.db file. Here are quick examples using Python3 and its sqlite3 package.

*Note: you cannot install the sqlite3 package through pip. Install it using your OS' package manager, search for 'sqlite'.*

## Filter on 3D structure resolution

We need to import sqlite3 and pandas packages first.

```
import sqlite3
import pandas as pd
```

Step 1 : We first get a list of chains that are below our favorite resolution threshold (here 4.0 Angströms):
```
with sqlite3.connect("results/RNANet.db) as connection:
    chain_list = pd.read_sql("""SELECT chain_id, structure_id, chain_name
                                FROM chain JOIN structure 
                                ON chain.structure_id = structure.pdb_id
                                WHERE resolution < 4.0 
                                ORDER BY structure_id ASC;""",
                            con=connection)

```

Step 2 : Then, we define a template string, containing the SQL request we use to get all information of one RNA chain, with brackets { } at the place we will insert every chain_id. 
You can remove fields you are not interested in.
```
req = """SELECT index_chain, old_nt_resnum, nt_position, nt_name, nt_code, nt_align_code, 
                is_A, is_C, is_G, is_U, is_other, freq_A, freq_C, freq_G, freq_U, freq_other, dbn,
                paired, nb_interact, pair_type_LW, pair_type_DSSR, alpha, beta, gamma, delta, epsilon, zeta, epsilon_zeta,
                chi, bb_type, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
                v0, v1, v2, v3, v4, amplitude, phase_angle, puckering 
                FROM 
                (SELECT chain_id, rfam_acc from chain WHERE chain_id = {})
                NATURAL JOIN re_mapping
                NATURAL JOIN nucleotide
                NATURAL JOIN align_column;"""
```

Step 3 : Finally, we iterate over this list of chains and save their information in CSV files:

```
with sqlite3.connect("results/RNANet.db) as connection:
    for chain in chain_list.iterrows():
        df = pd.read_sql(req.format(chain.chain_id), connection)
        filename = chain.structure_id + '-' + chain.chain_name + '.csv'
        df.to_csv(filename, float_format="%.2f", index=False)

```

## Filter on 3D structure publication date

You might want to get only the dataset you would have had in a past year, to compare yourself with the competitors of a RNA-Puzzles problem for example.
We will simply modify the Step 1 above:

```
with sqlite3.connect("results/RNANet.db) as connection:
    chain_list = pd.read_sql("""SELECT chain_id, structure_id, chain_name
                                FROM chain JOIN structure 
                                ON chain.structure_id = structure.pdb_id
                                WHERE date < "2018-06-01" 
                                ORDER BY structure_id ASC;""",
                            con=connection)
```
Then proceed to steps 2 and 3.

## Filter to avoid chain redundancy when several mappings are available
Some chains can be mapped to two (or more) RNA families, and exist several times in the database.
If you want just one example of each RNA 3D chain, use in Step 1:

```
with sqlite3.connect("results/RNANet.db) as connection:
    chain_list = pd.read_sql("""SELECT DISTINCT chain_id, structure_id, chain_name
                                FROM chain JOIN structure
                                ON chain.structure_id = structure.pdb_id
                                ORDER BY structure_id ASC;""",
                            con=connection)
```
Then proceed to steps 2 and 3.

# More about the database structure
To help you design your own requests, here follows a description of the database tables and fields.

## Table `family`, for Rfam families and their properties
* `rfam_acc`: The family codename, from Rfam's numbering (Rfam accession number)
* `description`: What RNAs fit in this family
* `nb_homologs`: The number of hits known to be homologous downloaded from Rfam to compute nucleotide frequencies
* `nb_3d_chains`: The number of 3D RNA chains mapped to the family (from Rfam-PDB mappings, or inferred using the redundancy list)
* `nb_total_homol`: Sum of the two previous fields, the number of sequences in the multiple sequence alignment, used to compute nucleotide frequencies
* `max_len`: The longest RNA sequence among the homologs (in bases)
* `comput_time`: Time required to compute the family's multiple sequence alignment in seconds,
* `comput_peak_mem`: RAM (or swap) required to compute the family's multiple sequence alignment in megabytes,
* `idty_percent`: Average identity percentage over pairs of the 3D chains' sequences from the family

## Table `structure`, for 3D structures of the PDB
* `pdb_id`: The 4-char PDB identifier
* `pdb_model`: The model used in the PDB file
* `date`: The first submission date of the 3D structure to a public database
* `exp_method`: A string to know wether the structure as been obtained by X-ray crystallography ('X-RAY DIFFRACTION'), electron microscopy ('ELECTRON MICROSCOPY'), or NMR (not seen yet)
* `resolution`: Resolution of the structure, in Angstöms

## Table `chain`, for the datapoints: one chain mapped to one Rfam family
* `chain_id`: A unique identifier
* `structure_id`: The `pdb_id` where the chain comes from
* `chain_name`: The chain label, extracted from the 3D file
* `eq_class`: The BGSU equivalence class label containing this chain
* `rfam_acc`: The family which the chain is mapped to (if not mapped, value is *unmappd*)
* `pdb_start`: Position in the chain where the mapping to Rfam begins (absolute position, not residue number)
* `pdb_end`: Position in the chain where the mapping to Rfam ends (absolute position, not residue number)
* `reversed`: Wether the mapping numbering order differs from the residue numbering order in the mmCIF file (eg 4c9d, chains C and D)
* `issue`: Wether an issue occurred with this structure while downloading, extracting, annotating or parsing the annotation. See the file known_issues_reasons.txt for more information about why your chain is marked as an issue.
* `inferred`: Wether the mapping has been inferred using the redundancy list (value is 1) or just known from Rfam-PDB mappings (value is 0)
* `chain_freq_A`, `chain_freq_C`, `chain_freq_G`, `chain_freq_U`, `chain_freq_other`: Nucleotide frequencies in the chain
* `pair_count_cWW`, `pair_count_cWH`, ... `pair_count_tSS`: Counts of the non-canonical base-pair types in the chain (intra-chain counts only)

## Table `nucleotide`, for individual nucleotide descriptors
* `nt_id`: A unique identifier
* `chain_id`: The chain the nucleotide belongs to
* `index_chain`: its absolute position within the portion of chain mapped to Rfam, from 1 to X. This is completely uncorrelated to any gene start or 3D chain residue numbers.
* `nt_position`: relative position within the portion of chain mapped to RFam, from 0 to 1
* `old_nt_resnum`: The residue number in the 3D mmCIF file (it's a string actually, some contain a letter like '37A')
* `nt_name`: The residue type. This includes modified nucleotide names (e.g. 5MC for 5-methylcytosine)
* `nt_code`: One-letter name. Lowercase "acgu" letters are used for modified "ACGU" bases.
* `nt_align_code`: One-letter name used for sequence alignment. Contains "ACGUN-" only first, and then, gaps may be replaced by the most common letter at this position (default)
* `is_A`, `is_C`, `is_G`, `is_U`, `is_other`: One-hot encoding of the nucleotide base
* `dbn`: character used at this position if we look at the dot-bracket encoding of the secondary structure. Includes inter-chain (RNA complexes) contacts.
* `paired`: empty, or comma separated list of `index_chain` values referring to nucleotides the base is interacting with. Up to 3 values. Inter-chain interactions are marked paired to '0'.
* `nb_interact`: number of interactions with other nucleotides. Up to 3 values. Includes inter-chain interactions.
* `pair_type_LW`: The Leontis-Westhof nomenclature codes of the interactions. The first letter concerns cis/trans orientation, the second this base's side interacting, and the third the other base's side.
* `pair_type_DSSR`: Same but using the DSSR nomenclature (Hoogsteen edge approximately corresponds to Major-groove and Sugar edge to minor-groove)
* `alpha`, `beta`, `gamma`, `delta`, `epsilon`, `zeta`: The 6 torsion angles of the RNA backabone for this nucleotide
* `epsilon_zeta`: Difference between epsilon and zeta angles
* `bb_type`: conformation of the backbone (BI, BII or ..)
* `chi`: torsion angle between the sugar and base (O-C1'-N-C4)
* `glyco_bond`: syn or anti configuration of the sugar-base bond
* `v0`, `v1`, `v2`, `v3`, `v4`: 5 torsion angles of the ribose cycle
* `form`: if the nucleotide is involved in a stem, the stem type (A, B or Z)
* `ssZp`: Z-coordinate of the 3’ phosphorus atom with reference to the5’ base plane
* `Dp`: Perpendicular distance of the 3’ P atom to the glycosidic bond
* `eta`, `theta`: Pseudotorsions of the backbone, using phosphorus and carbon 4'
* `eta_prime`, `theta_prime`: Pseudotorsions of the backbone, using phosphorus and carbon 1'
* `eta_base`, `theta_base`: Pseudotorsions of the backbone, using phosphorus and the base center
* `phase_angle`: Conformation of the ribose cycle
* `amplitude`: Amplitude of the sugar puckering
* `puckering`: Conformation of the ribose cycle (10 classes depending on the phase_angle value)

## Table `align_column`, for positions in multiple sequence alignments
* `column_id`: A unique identifier
* `rfam_acc`: The family's MSA the column belongs to
* `index_ali`: Position of the column in the alignment (starts at 1)
* `freq_A`, `freq_C`, `freq_G`, `freq_U`, `freq_other`: Nucleotide frequencies in the alignment at this position

There always is an entry, for each family (rfam_acc), with index_ali = zero and nucleotide frequencies set to freq_other = 1.0. This entry is used when the nucleotide frequencies cannot be determined because of local alignment issues.

## Table `re_mapping`, to map a nucleotide to an alignment column
* `remapping_id`: A unique identifier
* `chain_id`: The chain which is mapped to an alignment
* `index_chain`: The absolute position of the nucleotide in the chain (from 1 to X)
* `index_ali` The position of that nucleotide in its family alignment

# Troubleshooting

## Understanding the warnings and errors

* **Could not load X.json with JSON package** : 
The JSON format produced as DSSR output could not be loaded by Python. Try deleting the file and re-running DSSR (through RNANet).
* **Found DSSR warning in annotation X.json: no nucleotides found. Ignoring X.** : 
DSSR complains because the CIF structure does not seem to contain nucleotides. This can happen on low resolution structures where only P atoms are solved, you should ignore them. This also can happen if the .cif file is corrupted (failed download, etc). Check with a 3D visualization software if your chain contains well-defined nucleotides. Try deleting the .cif and retry. If the problem persists, just ignore the chain.
* **Could not find nucleotides of chain X in annotation X.json. Ignoring chain X.** : Basically the same as above, but some nucleotides have been observed in another chain of the same structure. 
* **Could not find real nucleotides of chain X between START and STOP. Ignoring chain X."** : Same as the two above, but nucleotides can be found outside of the mapping interval. This can happen if there is a mapping problem, e.g., considered absolute interval when it should not.
* **Error while parsing DSSR X.json output: {custom-error}** : The DSSR annotations lack some of our required fields. It is likely that DSSR changed something in their fields names. Contact us so that we fix the problem with the latest DSSR version.
* **Mapping is reversed, this case is not supported (yet). Ignoring chain X.** : The mapping coordinates, as obtained from Rfam, have an end position coming before the start position (meaning, the sequence has to be reversed to map the RNA covariance model). We do not support this yet, we ignore this chain.
* **Error with parsing of X duplicate residue numbers. Ignoring it.** : This 3D chain contains new kind(s) of issue(s) in the residue numberings that are not part of the issues we already know how to tackle. Contact us, so that we add support for this entry.
* **Found duplicated index_chain N in X. Keeping only the first.** : This RNA 3D chain contains two (or more) residues with the same numbering N. This often happens when a nucleic-like ligand is annotated as part of the RNA chain, and DSSR considers it a nucleotide. By default, RNANet keeps only the first of the multiple residues with the same number. You may want to check that the produced 3D structure contains the appropriate nucleotide and no ligand.
* **Missing index_chain N in X !** : DSSR annotations for chain X are discontinuous, position N is missing. This means residue N has not been recognized as a nucleotide by DSSR. Is the .cif structure file corrupted ? Delete it and retry.
* **X sequence is too short, let's ignore it.** : We discard very short RNA chains.
* **Error downloading and/or extracting Rfam.cm !** : We cannot retrieve the Rfam covariance models file. RNANet tries to find it at ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz so, check that your network is not blocking the FTP protocol (port 21 is open on your network), and check that the adress has not changed. If so, contact us so that we update RNANet with the correct address.
* **Something's wrong with the SQL database. Check mysql-rfam-public.ebi.ac.uk status and try again later. Not printing statistics.** : We cannot retrieve family statistics from Rfam public server. Check if you can connect to it by hand : `mysql -u rfamro -P 4497 -D Rfam -h mysql-rfam-public.ebi.ac.uk`. if not, check that the port 497 is opened on your network.
* **Error downloading RFXXXXX.fa.gz: {custom-error}** : We cannot reach the Rfam FTP server to download homologous sequences. We look in ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/ so, check if you can access it from your network (check that port 21 is opened on your network). Check if the address has changed and notify us.
* **Error downloading NR list !** : We cannot download BGSU's equivalence classes from their website. Check if you can access http://rna.bgsu.edu/rna3dhub/nrlist/download/current/20.0A/csv from a web browser. It actually happens that their website is not responding, the previous download will be re-used.
* **Error downloading the LSU/SSU database from SILVA** : We cannot reach SILVA's arb files. We are looking for http://www.arb-silva.de/fileadmin/arb_web_db/release_132/ARB_files/SILVA_132_LSURef_07_12_17_opt.arb.gz and http://www.arb-silva.de/fileadmin/silva_databases/release_138/ARB_files/SILVA_138_SSURef_05_01_20_opt.arb.gz , can you download and extract them from your web browser and place them in the realigned/ subfolder ?
* **Assuming mapping to RFXXXXX is an absolute position interval.** : The mapping provided by Rfam concerns a nucleotide interval START-END, but no nucleotides are defined in 3D in that interval. When this happens, we assume that the numbering is not relative to the residue numbers in the 3D file, but to the absolute position in the chain, starting at 1. And yes, we tried to apply this behavior to all mappings, this yields the opposite issue where some mappings get outside the available nucleotides. To be solved the day Rfam explains how they get build the mappings.
* **Added newly discovered issues to known issues** : You discovered new chains that cannot be perfectly understood as they actually are, congrats. For each chain of the list, another warning has been raised, refer to them. 
* **Structures without referenced chains have been detected.** : Something went wrong, because the database contains references to 3D structures that are not used by any entry in the `chain` table. You should rerun RNANet. The option `--only` may help to rerun it just for one chain.
* **Chains without referenced structures have been detected** : 
Something went wrong, because the database contains references to 3D chains that are not used by any entry in the `structure` table. You should rerun RNANet. The option `--only` may help to rerun it just for one chain.
* **Chains were not remapped** : Something went wrong, because the database contains references to 3D chains that are not used by any entry in the `re_mapping` table, assuming you were interested in homology data. You should rerun RNANet. The option `--only` may help to rerun it just for one chain. If you are not interested in homology data, use option `--no-homology` to skip alignment and remapping steps.
* **Operational Error: database is locked, retrying in 0.2s** : Too many workers are trying to access the database at the same time. Do not try to run several instances of RNANet in parallel. Even with only one instance, this might still happen if your device has slow I/O delays. Try to run RNANet from a SSD ?
* **Tried to reach database 100 times and failed. Aborting.** : Same as above, but in a more serious way.
* **Nothing to do !** : RNANet is up-to-date, or did not detect any modification to do, so nothing changed in your database.
* **KeyboardInterrupt, terminating workers.** : You interrupted the computation by pressing Ctrl+C. The database may be in an unstable state, rerun RNANet to solve the problem.
* **Found mappings to RFXXXXX in both directions on the same interval, keeping only the 5'->3' one.**  : A chain has been mapped to family RFXXXXX, but the mapping has been found twice, with the limits inverted. We only keep one (in 5'->3' sense).
* **There are mappings for RFXXXXX in both directions** : A chain has been mapped to family RFXXXXX several times, and the mappings are not in the same sequence sense (some are reverted, with END < START). Then, we do not know what to decide for this chain, and we abort. 
* **Unable to download XXXX.cif. Ignoring it.** :  We cannot access a certain 3D structure from RCSB's download site, can you access it from your web browser and put it in the RNAcifs/ folder ? We look at http://files.rcsb.org/download/XXXX.cif , replacing XXXX by the right PDB code.
* **Wtf, structure XXXX has no resolution ? Check https://files.rcsb.org/header/XXXX.cif to figure it out.** : We cannot find the resolution of structure XXXX from the .cif file. We are looking for it in the fields `_refine.ls_d_res_high`, `_refine.ls_d_res_low`, and `_em_3d_reconstruction.resolution`. Maybe the information is stored in another field ? If you find it, contact us so that we support this new CIF field.
* **Could not find annotations for X, ignoring it.** : It seems that DSSR has not been run for structure X, or failed. Rerun RNANet.
* **Nucleotides not inserted: {custom-error}** : For some reason, no nucleotides were saved to the database for this chain. Contact us.
* **Removing N doublons from existing RFXXXXX++.fa and using their newest version** : You are trying to re-compute sequence alignments of 3D structures that had already been computed in the past. They will be removed from the alignment and recomputed, for the case the sequences have changed.
* **Removing N doublons from existing RFXXXXX++.stk and using their newest version** :  Same as above.
* **Error during sequence alignment: {custom-error}** : Something went wrong during sequence alignment. Recompute the alignments using the `--update-homologous` option.
* **Failed to realign RFXXXXX (killed)** : You ran out of memory while computing multiple sequence alignments. Try to run RNANet of a machine with at least 32 GB of RAM.
* **RFXXXXX's alignment is wrong. Recompute it and retry.** : We could not load RFXXXXX's multiple sequence alignment. It may have failed to compute, or be corrupted. Recompute the alignments using the `--update-homologous` option.

## Not enough memory
If you run out of memory, you may want to reduce the number of jobs run in parallel. #TODO: explain how

# Contact
louis.becquey@univ-evry.fr
