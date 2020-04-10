# RNANet
Building a dataset following the ProteinNet philosophy, but for RNA.

We use the Rfam mappings between 3D structures and known Rfam families, using the sequences that are known to belong to an Rfam family (hits provided in RF0XXXX.fasta files from Rfam).
Future versions might compute a real MSA-based clusering directly with Rfamseq ncRNA sequences, like ProteinNet does with protein sequences, but this requires a tool similar to jackHMMER in the Infernal software suite, which is not available yet.

This script prepares the dataset from available public data in PDB and Rfam.


**Please cite**: *Coming soon, expect it summer 2020*

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

* Run DSSR on every chain to get a variety of descriptors per position, describing secondary and tertiary structure

Finally, store this data into a SQLite database and CSV files.

# Output files

* `results/RNANet.db` is a SQLite database file containing several tables with all the information, which you can query yourself with your custom requests,
* `3D-folder-you-passed-in-option/datapoints/*.csv` are flat text CSV files, one for one RNA chain mapped to one RNA family, gathering the per-position nucleotide descriptors,
* `path-to-3D-folder-you-passed-in-option/rna_mapped_to_Rfam` If you used the --extract option, this folder contains one mmCIF file per RNA chain mapped to one RNA family, without other chains, proteins (nor ions and ligands by default)
* `results/summary.csv` summarizes information about the RNA families.

Other folders are created and not deleted, which you might want to conserve to avoid later re-computation:

* `path-to-sequence-folder-you-passed-in-option/rfam_sequences/fasta/` contains compressed FASTA files of the homologous sequences used, by Rfam family.
* `path-to-sequence-folder-you-passed-in-option/realigned/` contains families covariance models (\*.cm), unaligned list of sequences (\*.fa), and multiple sequence alignments in both formats Stockholm and Aligned-FASTA (\*.stk and \*.afa). Also contains SINA homolgous sequences databases LSU.arb and SSU.arb, and their index files (\*.sidx).
* `path-to-3D-folder-you-passed-in-option/RNAcifs/` contains mmCIF structures directly downloaded from the PDB, which contain RNA chains,
* `path-to-3D-folder-you-passed-in-option/annotations/` contains the raw JSON annotation files of the previous mmCIF structures. You may find additional information into them which is not properly supported by RNANet yet.

# How to run
## Dependencies
You need to install:
- DSSR, you need to register to the X3DNA forum [here](http://forum.x3dna.org/site-announcements/download-instructions/) and then download the DSSR binary [on that page](http://forum.x3dna.org/downloads/3dna-download/). 
- Infernal, to download at [Eddylab](http://eddylab.org/infernal/), several options are available depending on your preferences. Make sure to have the `cmalign` and `esl-reformat` binaries in your $PATH variable, so that RNANet.py can find them.You don't need the whole X3DNA suite of tools, just DSSR is fine. Make sure to have the `x3dna-dssr` binary in your $PATH variable so that RNANet.py finds it.
- SINA, follow [these instructions](https://sina.readthedocs.io/en/latest/install.html) for example. Make sure to have the `sina` binary in your $PATH.
- Python >= 3.8, (Unfortunately, python3.6 is no longer supported, because of changes in the multiprocessing and Threading packages. Untested with Python 3.7.\*)
- The following Python packages: `python3.8 -m pip install numpy matplotlib pandas biopython psutil pymysql requests sqlalchemy sqlite3 tqdm`

## Command line
Run `./RNANet.py --3d-folder path/to/3D/data/folder --seq-folder path/to/sequence/data/folder [ - other options ]`. 
It requires solid hardware to run. It takes around 15 hours the first time, and 9h then, tested on a server with 32 cores and 48GB of RAM.
The detailed list of options is below:

```
-h [ --help ]                   Print this help message
--version                       Print the program version

-r 4.0 [ --resolution=4.0 ]     (1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 20.0)
                                Minimum 3D structure resolution to consider a RNA chain.
-s                              Run statistics computations after completion
--keep-hetatm=False             (True | False) Keep ions, waters and ligands in produced mmCIF files. 
                                Does not affect the descriptors.
--fill-gaps=True                (True | False) Replace gaps in sequence due to unresolved residues
                                by the most common nucleotide at this position in the alignment.
--3d-folder=…                   Path to a folder to store the 3D data files. Subfolders will contain:
                                        RNAcifs/                Full structures containing RNA, in mmCIF format
                                        rna_mapped_to_Rfam/     Extracted 'pure' RNA chains
                                        datapoints/             Final results in CSV file format.
--seq-folder=…                  Path to a folder to store the sequence and alignment files.
                                        rfam_sequences/fasta/   Compressed hits to Rfam families
                                        realigned/              Sequences, covariance models, and alignments by family
--no-homology                   Do not try to compute PSSMs and do not align sequences.
                                Allows to yield more 3D data (consider chains without a Rfam mapping).
--from-scratch                  Delete already computed data and known issues, and recompute.
--retry-issues                  Ignore already known issues, and retry to build them from scratch.
```

# Dataset quality
The file statistics.py is supposed to give a summary on the produced dataset. See the results/ folder. It can be run automatically after RNANet if you pass the `-s` option.

# Database structure
You might want to build your own sub-dataset by querying the results/RNANet.db file.

For example in Python3,
```
import sqlite3
import pandas as pd

with sqlite3.connect("results/RNANet.db) as connection:
    df = pd.read_sql("""SELECT structure_id, chain_name
                        FROM chain JOIN structure 
                        WHERE resolution < 4.0 
                        ORDER BY date ASC;""",
                     con=connection)

df.to_csv("my_custom_results.csv")
```
To help you, here follows a description of the database tables and fields.

## Table `family`, for Rfam families and their properties
* `rfam_acc`: The family codename, from Rfam's numbering (Rfam accession number)
* `nb_homologs`: The number of hits known to be homologous downloaded from Rfam to compute nucleotide frequencies
* `nb_3d_chains`: The number of 3D RNA chains mapped to the family (from Rfam-PDB mappings, or inferred using the redundancy list)
* `nb_total_homol`: Sum of the two previous fields, the number of sequences in the multiple sequence alignment, used to compute nucleotide frequencies
* `max_len`: The longest RNA sequence among the homologs
* `comput_time`: Time required to compute the family's multiple sequence alignment,
* `comput_peak_mem`: RAM (or swap) required to compute the family's multiple sequence alignment,
* `idty_percent`: Identity of the 3D chains' sequences from the family

## Table `structure`, for 3D structures of the PDB
* `pdb_id`: The 4-char PDB identifier
* `pdb_model`: The model used in the PDB file
* `date`: The first submission date of the 3D structure to a public database
* `exp_method`: A string to know wether the structure as been obtained by X-ray crystallography, NMR, or electron microscopy
* `resolution`: Resolution of the structure, in Angstöms

## Table `chain`, for the datapoints: one chain mapped to one Rfam family
* `chain_id`: A unique identifier
* `structure_id`: The `pdb_id` where the chain comes from
* `chain_name`: The chain label, extracted from the 3D file
* `pdb_start`: Position in the chain where the mapping to Rfam begins (absolute position, not residue number)
* `pdb_end`: Position in the chain where the mapping to Rfam ends (absolute position, not residue number)
* `pdb_start`: Position in the chain where the mapping to Rfam begins (absolute position, not residue number)
* `pdb_start`: Position in the chain where the mapping to Rfam begins (absolute position, not residue number)
* `reversed`: Wether the mapping numbering order differs from the residue numbering order in the mmCIF file (eg 4c9d, chains C and D)
* `issue`: Wether an issue occurred with this structure while downloading, extracting, annotating or parsing the annotation. Chains with issues are removed from the dataset (Only one known to date: 1gsg, chain T, which is too short)
* `rfam_acc`: The family which the chain is mapped to
* `inferred`: Wether the mapping has been inferred using the redundancy list (value is 1) or just known from Rfam-PDB mappings (value is 0)
* `chain_freq_A`, `chain_freq_C`, `chain_freq_G`, `chain_freq_U`, `chain_freq_other`: Nucleotide frequencies in the chain
* `pair_count_cWW`, `pair_count_cWH`, ... `pair_count_tSS`: Counts of the non-canonical base-pair types in the chain (intra-chain counts only)

## Table `nucleotide`, for individual nucleotide descriptors
* `nt_id`: A unique identifier
* `chain_id`: The chain the nucleotide belongs to
* `index_chain`: its absolute position within the portion of chain mapped to Rfam, from 1 to X. This is completely uncorrelated to any gene start or 3D chain residue numbers.
* `nt_position`: relative position within the portion of chain mapped to RFam, from 0 to 1
* `nt_resnum`: The residue number in the 3D mmCIF file
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

## Example
By default, the CSV files produced by RNANet are retrieved using the following query, for each chain_id in the table chain (replace {self.db_chain_id} by some chain_id)

```
SELECT (index_chain, nt_resnum, position, nt_name, nt_code, nt_align_code, 
        is_A, is_C, is_G, is_U, is_other, freq_A, freq_C, freq_G, freq_U, freq_other, dbn,
        paired, nb_interact, pair_type_LW, pair_type_DSSR, alpha, beta, gamma, delta, epsilon, zeta, epsilon_zeta,
        chi, bb_type, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
        v0, v1, v2, v3, v4, amlitude, phase_angle, puckering) FROM (
            (SELECT (chain_id, rfam_acc) from chain WHERE chain_id = {self.db_chain_id})
            NATURAL JOIN re_mapping
            NATURAL JOIN nucleotide
            NATURAL JOIN align_column
        );
```

# Contact
louis.becquey@univ-evry.fr
