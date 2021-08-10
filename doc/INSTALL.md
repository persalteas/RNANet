
* [Required computational resources](#required-computational-resources)
* [Method 1 : Using Docker](#method-1-:-installation-using-docker)
* [Method 2 : Classical command-line installation](#method-2-:-classical-command-line-installation-linux-only)
* [Command options](#command-options)
* [Computation time](#computation-time)
* [Post-computation tasks](#post-computation-tasks-estimate-quality)
* [Output files](#output-files)

# Required computational resources
- CPU: no requirements. The program is optimized for multi-core CPUs, you might want to use Intel Xeons, AMD Ryzens, etc.
- GPU: not required
- RAM: 16 GB with a large swap partition is okay. 32 GB is recommended (usage peaks at ~27 GB, but this number depends on your number of CPU cores)
- Storage: to date, it takes 60 GB for the 3D data (36 GB if you don't use the --extract option), 11 GB for the sequence data, and 7GB for the outputs (5.6 GB database, 1 GB archive of CSV files). You need to add a few more for the dependencies. Pick a 100GB partition and you are good to go. The computation speed is way better if you use a fast storage device (e.g. SSD instead of hard drive, or even better, a NVMe SSD) because of constant I/O with the SQlite database.
- Network : We query the Rfam public MySQL server on port 4497. Make sure your network enables communication (there should not be any issue on private networks, but maybe you company/university closes ports by default). You will get an error message if the port is not open. Around 30 GB of data is downloaded.

# Method 1 : Installation using Docker

* Step 1 : Download the [Docker container](https://entrepot.ibisc.univ-evry.fr/d/1aff90a9ef214a19b848/files/?p=/rnanet_v1.5b_docker.tar&dl=1). Open a terminal and move to the appropriate directory.
* Step 2 : Extract the archive to a Docker image named *rnanet* in your local installation
```
$ docker load -i rnanet_v1.5b_docker.tar
```
* Step 3 : Run the container, giving it 3 folders to mount as volumes: a first to store the 3D data, a second to store the sequence data and alignments, and a third to output the results, data and logs:
```
$ docker run --rm -v path/to/3D/data/folder:/3D -v path/to/sequence/data/folder:/sequences -v path/to/experiment/results/folder:/runDir rnanet [ - other options ]
```

Typical usage:
```
nohup bash -c 'time docker run --rm -v /path/to/3D/data/folder:/3D -v /path/to/sequence/data/folder:/sequences -v /path/to/experiment/folder:/runDir rnanet -s --no-logs ' &
```


# Method 2 : Classical command line installation (Linux only)

You need to install the dependencies:
- DSSR 1.9.9 or newer, you need to register to ask for a DSSR (academic) license [on that page](http://innovation.columbia.edu/technologies/CU20391). Make sure to have the `x3dna-dssr` binary in your $PATH variable so that RNANet.py finds it.
- Infernal 1.1.4 or newer, to download at [Eddylab](http://eddylab.org/infernal/), several options are available depending on your preferences. Make sure to have the `cmalign`, `cmfetch`, `cmbuild`, `esl-alimanip`, `esl-alimerge`, `esl-alipid` and `esl-reformat` binaries in your $PATH variable, so that RNANet.py can find them.
- SINA (if you plan to use it), follow [these instructions](https://sina.readthedocs.io/en/latest/install.html) for example. Make sure to have the `sina` binary in your $PATH.
- Sqlite 3, available under the name *sqlite* in every distro's package manager,
- Python >= 3.8, (Unfortunately, python3.6 is no longer supported, because of changes in the multiprocessing and Threading packages. Untested with Python 3.7.\*). 
- The following Python packages: `python3.8 -m pip install biopython matplotlib pandas psutil pymysql requests scipy setproctitle sqlalchemy tqdm`. 

Then, run it from the command line, preferably using nohup if your shell will be interrupted:
```
 ./RNANet.py --3d-folder path/to/3D/data/folder --seq-folder path/to/sequence/data/folder [ - other options ]
```

Typical usage:
```
nohup bash -c 'time ~/Projects/RNANet/RNAnet.py --3d-folder ~/Data/RNA/3D/ --seq-folder ~/Data/RNA/sequences -s --no-logs' &
```

# Command options

The detailed list of options is below:

```
-h [ --help ]			Print this help message
--version			Print the program version

Select what to do:
--------------------------------------------------------------------------------------------------------------
-f [ --full-inference ]		Infer new mappings even if Rfam already provides some. Yields more copies of
				 chains mapped to different families.
-s				Run statistics computations after completion
--stats-opts=…			Pass additional command line options to the statistics.py script, e.g. "--wadley --distance-matrices"
--extract			Extract the portions of 3D RNA chains to individual mmCIF files.
--keep-hetatm=False		(True | False) Keep ions, waters and ligands in produced mmCIF files. 
				 Does not affect the descriptors.
--no-homology			Do not try to compute PSSMs and do not align sequences.
				 Allows to yield more 3D data (consider chains without a Rfam mapping).

Select how to do it:
--------------------------------------------------------------------------------------------------------------
--3d-folder=…			Path to a folder to store the 3D data files. Subfolders will contain:
					RNAcifs/		Full structures containing RNA, in mmCIF format
					rna_mapped_to_Rfam/	Extracted 'pure' portions of RNA chains mapped to families
					rna_only/	Extracted 'pure' RNA chains, not truncated
					datapoints/		Final results in CSV file format.
--seq-folder=…			Path to a folder to store the sequence and alignment files. Subfolders will be:
					rfam_sequences/fasta/	Compressed hits to Rfam families
					realigned/		Sequences, covariance models, and alignments by family
--sina				Align large subunit LSU and small subunit SSU ribosomal RNA using SINA instead of Infernal,
				 the other RNA families will be aligned using infernal.
--maxcores=…			Limit the number of cores to use in parallel portions to reduce the simultaneous
				 need of RAM. Should be a number between 1 and your number of CPUs. Note that portions
				 of the pipeline already limit themselves to 50% or 70% of that number by default.
--cmalign-opts=…		A string of additional options to pass to cmalign aligner, e.g. "--nonbanded --mxsize 2048"
--archive			Create tar.gz archives of the datapoints text files and the alignments,
				 and update the link to the latest archive. 
--no-logs			Do not save per-chain logs of the numbering modifications.

Select which data we are interested in:
--------------------------------------------------------------------------------------------------------------
-r 4.0 [ --resolution=4.0 ]	Maximum 3D structure resolution to consider a RNA chain.
--all				Process chains even if they already are in the database.
--redundant			Process all members of the equivalence classes not only the representative.
--only				Ask to process a specific chains only (e.g. 4v49, 4v49_1_AA, or 4v49_1_AA_5-1523).
--ignore-issues			Do not ignore already known issues and attempt to compute them.
--update-homologous		Re-download Rfam and SILVA databases, realign all families, and recompute all CSV files.
--from-scratch			Delete database, local 3D and sequence files, and known issues, and recompute.

```
Options --3d-folder and --seq-folder are mandatory for command-line installations, but should not be used for installations with Docker. In the Docker container, they are set by default to the paths you provide with the -v options.

The most useful options in that list are 
* ` --extract`, to actually produce some re-numbered 3D mmCIF files of the RNA chains individually,
* ` --no-homology`, to ignore the family mapping and sequence alignment parts and only focus on 3D data download and annotation. This would yield more data since many RNAs are not mapped to any Rfam family,
* ` -s`, to run the "statistics" which are a few useful post-computation tasks such as:
    * Computation of sequence identity matrices
    * Statistics over the sequence lengths, nucleotide frequencies, and basepair types by RNA family
    * Overall database content statistics
    * Detailed analysis of the eta-theta pseudotorsion angles  (use `--stats-opts="--wadley"` after `-s`) or 3D distance matrices and their averages per family (use `--stats-opts="--distance-matrices"`)
* ` --redundant`, to yield all the available data and not only the BGSU NR-List respresentatives

# Computation time 

To give you an estimation, our last full run took exactly 12h, excluding the time to download the MMCIF files containing RNA (around 25GB to download) and the time to compute statistics.
Measured the 23rd of June 2020 on a 16-core AMD Ryzen 7 3700X CPU @3.60GHz, plus 32 Go RAM, and a 7200rpm Hard drive. Total CPU time spent: 135 hours (user+kernel modes), corresponding to 12h (actual time spent with the 16-core CPU). 
Another recent full run, including the MMCIF downloads and computation of heavy statistics (`--wadley --distance-matrices`) last 13h (real time) on a 60-core Xeon E7-4850v4@2.10GHz and 120 Go of RAM. The user+kernel time was about 300h.

Update runs are much quicker, around 3 hours. It depends mostly on what RNA families are concerned by the update.


# Post-computation tasks (estimate quality)
If your did not ask for automatic run of statistics over the produced dataset with the `-s` option, you can run them later using the file statistics.py. 
```
python3.8 statistics.py --3d-folder path/to/3D/data/folder --seq-folder path/to/sequence/data/folder -r 20.0
```
/!\ Beware, if not precised with option `-r`, no resolution threshold is applied and all the data in RNANet.db is used.

By default, this computes:
* Computation of sequence identity matrices
* Statistics over the sequence lengths, nucleotide frequencies, and basepair types by RNA family
* Overall database content statistics

If you have run RNANet once with option `--extract`, additionally, you can compute more by passing the options:
*  With option `--distance-matrices` to compute pairwise residue distances within the chain for every chain, and compute average and standard deviations by RNA families. This is supposed to capture the average shape of an RNA family. The distance matrices are the size of the family's covariance model (match states). Unresolved nucleotides or deletions to the covariance model are NaNs.

If you have run RNANet once with options `--no-homology` and `--extract`, you unlock new statistics over unmapped chains.
* You will be allowed to use option `--wadley` to reproduce Wadley & al. (2007) results automatically. These are clustering results of the pseudotorsions angles of the backbone.

# Output files

* `results/RNANet.db` is a SQLite database file containing several tables with all the information, which you can query yourself with your custom requests,
* `3D-folder-you-passed-in-option/datapoints/*` are flat text CSV files, one for one RNA chain mapped to one RNA family, gathering the per-position nucleotide descriptors,
* `archive/RNANET_datapoints_{DATE}.tar.gz` is a compressed archive of the above CSV files (only if you passed the --archive option)
* `archive/RNANET_alignments_latest.tar.gz` is a compressed archive of multiple sequence alignments in FASTA format, one per RNA family, including only the portions of chains with a 3D structure which are mapped to a family. The alignment has been computed with all the RFam sequences of that family, but they have been removed then.
* `path-to-3D-folder-you-passed-in-option/rna_mapped_to_Rfam` If you used the `--extract` option, this folder contains one mmCIF file per RNA chain mapped to one RNA family, without other chains, proteins (nor ions and ligands by default). If you used both `--extract` and `--no-homology`, this folder is called `rna_only`.
* `results/summary.csv` summarizes information about the RNA chains
* `results/families.csv` summarizes information about the RNA families
* `results/pair_types.csv` summarizes statistics about base-pair types in every family.
* `results/frequencies.csv` summarizes statistics about nucleotides frequencies in every family (including all known modified bases)

Other folders are created and not deleted, which you might want to conserve to avoid re-computations in later runs:

* `path-to-sequence-folder-you-passed-in-option/rfam_sequences/fasta/` contains compressed FASTA files of the homologous sequences used, by Rfam family.
* `path-to-sequence-folder-you-passed-in-option/realigned/` contains families covariance models (\*.cm), unaligned list of sequences (\*.fa), and multiple sequence alignments in both formats Stockholm and Aligned-FASTA (\*.stk and \*.afa). Also contains SINA homolgous sequences databases LSU.arb and SSU.arb, and their index files (\*.sidx).
* `path-to-3D-folder-you-passed-in-option/RNAcifs/` contains mmCIF structures directly downloaded from the PDB, which contain RNA chains,
* `path-to-3D-folder-you-passed-in-option/annotations/` contains the raw JSON annotation files of the previous mmCIF structures. You may find additional information into them which is not properly supported by RNANet yet.