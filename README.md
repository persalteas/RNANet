# RNANet

Contents:
* [What is RNANet ?](#what-is-rnanet-?)
* [Install and run RNANet](doc/INSTALL.md)
* [How to further filter the dataset](#how-to-further-filter-the-dataset)
    * [Filter on 3D structure resolution](#filter-on-3D-structure-resolution)
    * [Filter on 3D structure publication date](#filter-on-3d-structure-publication-date)
    * [Filter to avoid chain redundancy when several mappings are available](#filter-to-avoid-chain-redundancy-when-several-mappings-are-available)
* [Database tables documentation](doc/Database.md)
* [FAQ](doc/FAQ.md)
* [Troubleshooting](#troubleshooting)
* [Contact](#contact)

## Cite us

* Louis Becquey, Eric Angel, and Fariza Tahi, (2020) **RNANet: an automatically built dual-source dataset integrating homologous sequences and RNA structures**, *Bioinformatics*, 2020, btaa944, [DOI](https://doi.org/10.1093/bioinformatics/btaa944), [Read the OpenAccess paper here](https://doi.org/10.1093/bioinformatics/btaa944)

Additional relevant references:

If you use our annotations by DSSR, you might want to cite:
* Lu, X.-J.et al.(2015). **DSSR: An integrated software tool for dissecting the spatial structure of RNA.** *Nucleic Acids Research*, 43(21), e142–e142.

If you use our multiple sequence alignments and homology data, you might want to cite:
* Nawrocki, E. P. and Eddy, S. R. (2013). **Infernal 1.1: 100-fold faster RNA homology searches.** *Bioinformatics*, 29(22), 2933–2935.
* Pruesse, E. et al.(2012). **Sina: accurate high-throughput multiple sequence alignment of ribosomal RNA genes.** *Bioinformatics*, 28(14), 1823–1829



# What is RNANet ?
RNANet is a multiscale dataset of non-coding RNA structures, including sequences, secondary structures, non-canonical interactions, 3D geometrical descriptors, and sequence homology.

It is available in machine-learning ready formats like CSV files (one per RNA chain) or as a SQL database.

Most interestingly, nucleotides have been renumered in a standardized way, and the 3D chains have been re-aligned with homologous sequences and covariance models from the [Rfam](https://rfam.org/) database.


## Methodology
We use the Rfam mappings between 3D structures and known Rfam families, using the sequences that are known to belong to an Rfam family (hits provided in RF0XXXX.fasta files from Rfam).
Future versions might compute a real MSA-based clusering directly with Rfamseq ncRNA sequences, like ProteinNet does with protein sequences, but this requires a tool similar to jackHMMER in the Infernal software suite, which is not available yet. 
If interested by such approaches, the user may check tools like RNAlien.

This script prepares the dataset from available public data in PDB, RNA 3D Hub, Rfam and SILVA.


## Pipeline
The script follows these steps:

To gather structures:
* Gets a list of 3D structures containing RNA from BGSU's non-redundant list (redundancy can be kept or eliminated, see command line option `--redundant`),
* Asks Rfam for mappings of these structures onto Rfam families (~50% of structures have a direct mapping, some more are inferred using the redundancy list)
* Downloads the corresponding 3D structures (mmCIFs)
* Standardizes the residue numbering from 1 to N, including missing residues (gaps)
* If desired, extracts the renumbered chain portions that map onto an Rfam family to a separate mmCIF file

To compute homology information:
* Extracts the sequence of every 3D chain
* Downloads Rfamseq ncRNA sequence hits for the concerned Rfam families (or ARB databases of SSU or LSU sequences from SILVA for rRNAs)
* Realigns Rfamseq hits and sequences from the 3D structures together to obtain a multiple sequence alignment for each Rfam family (using `cmalign`, but SINA can be used for ribosomal LSU and SSU)
* Computes nucleotide frequencies at every position for each alignment
* Map each nucleotide of a 3D chain to its position in the corresponding family sequence alignment

To compute 3D annotations:
* Run DSSR on every RNA structure to get a variety of descriptors per position, describing secondary and tertiary structure. Basepair types annotations include intra-chain and inter-chain interactions.

Finally, export this data from the SQLite database into flat CSV files.

Statistical analysis of the structures:
* Computes statistics about the amount of data from various resolutions and experimental methods (by RNA family)
* Computes basic statistics about the frequency of (modified) nucleotides by chain and by family,
* Computes basic statistics about the frequencies of non-canonical interactions,
* Computes density estimations (using Gaussian mixtures) for various geometrical parameters like distances and torsion angles for different representations : all-atom, the Pyle/VFold model, and the HiRE-RNA model,
* Computes pairwise residue distance matrices for each chain, and average + std-dev by RNA family
* Computes sequence identity matrices for each RNA family (based on the alignments)
* Saves covariance models (Infernal .cm files) for each RNA family

## Data provided

We provide couple of resources to exploit this dataset. You can download them on [EvryRNA](https://evryrna.ibisc.univ-evry.fr/evryrna/rnanet/rnanet_home).
* A series of tables in the SQLite3 database, see [the database documentation](doc/Database.md) and [examples of useful queries](#how-to-further-filter-the-dataset),
* One CSV file per RNA chain, summarizing all the relevant information about it,
* Filtered alignment files in FASTA format containing only the sequences with a 3D structure available in RNANet, but which have been aligned using all the homologous sequences of this family from Rfam or SILVA,
* Additional statistics files about nucleotide frequencies, modified bases, basepair types within each chain or by RNA family.

For now, we do not provide as public downloads the set of cleaned 3D structures nor the full alignments with Rfam sequences. If you need them, [recompute them](doc/INSTALL.md) or ask us.

## Updates
RNANet is updated monthly to take into account new structures proposed in the [BGSU Non-redundant lists](http://rna.bgsu.edu/rna3dhub/nrlist/). The monthly runs realign previous alignments with the new sequences using `esl-alimerge` from Infernal.

It is updated yearly from scratch to take into account new Rfam sequences or updates in the covariance models, and updates in the PDB 3D files.

For now, the SILVA releases used are fixed (LSU and SSU releases 138.1) and not automatically updated. SILVA authors if you reach this : please provide a "latest" download link to ease automatic retrieval of the latest version.

See what's new in the latest version of RNANet [in the CHANGELOG](CHANGELOG).

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

# Troubleshooting

Check if your problem is listed in the [known issues](doc/KnownIssues.md).

### Warning and Errors
If you ran RNANet and got an error or a warning that you do not fully understand, check the [Error documentation](doc/Errors.md).

### Not enough memory
If you run out of memory (job killed), you may want to reduce the number of jobs run in parallel. Use the `--maxcores` option with a small number to ask RNANet to limit the concurrency and the simultaneous need for a lot of RAM. The computation time will increase accordingly. If the blocking part is a `cmalign` alignment, use `--cmalign-opts="--cyk --nonbanded --notrunc --small"` to reduce alignment requirements.

### Not enough memory/too slow (developer trick)
If `--maxcores` is not enough, and that you identified the step which fails, you can try to edit the Python code. Look for the "coeff_ncores" argument of some functions calls. This is the coefficient applied to `--maxcores` for different steps of the pipeline. You can change it following your needs to reduce or increase concurrency (to use less memory, or compute faster, respectively).

# Contact
RNANet is still in beta, this means we are truly open (and enjoying) all the feedback we can get from interested users.

Please send all your questions, feature requests, bug reports or angry reacts to
louis.becquey(a)univ-evry.fr 
