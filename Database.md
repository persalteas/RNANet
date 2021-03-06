
# More about the database structure
To help you design your own SQL requests, we provide a description of the database tables and fields.

## Table `family`, for Rfam families and their properties
* `rfam_acc`: The family codename, from Rfam's numbering (Rfam accession number)
* `description`: What RNAs fit in this family
* `nb_homologs`: The number of hits known to be homologous downloaded from Rfam to compute nucleotide frequencies
* `nb_3d_chains`: The number of 3D RNA chains mapped to the family (from Rfam-PDB mappings, or inferred using the redundancy list)
* `nb_total_homol`: Sum of the two previous fields, the number of sequences in the multiple sequence alignment, used to compute nucleotide frequencies
* `max_len`: The longest RNA sequence among the homologs (in bases, unaligned)
* `ali_len`: The aligned sequences length (in bases, aligned)
* `ali_filtered_len`: The aligned sequences length when we filter the alignment to keep only the RNANet chains (which have a 3D structure) and some gap-only columns.
* `comput_time`: Time required to compute the family's multiple sequence alignment in seconds,
* `comput_peak_mem`: RAM (or swap) required to compute the family's multiple sequence alignment in megabytes,
* `idty_percent`: Average identity percentage over pairs of the 3D chains' sequences from the family

## Table `structure`, for 3D structures of the PDB
* `pdb_id`: The 4-char PDB identifier
* `pdb_model`: The model used in the PDB file
* `date`: The first submission date of the 3D structure to a public database
* `exp_method`: A string to know wether the structure as been obtained by X-ray crystallography ('X-RAY DIFFRACTION'), electron microscopy ('ELECTRON MICROSCOPY'), or NMR (not seen yet)
* `resolution`: Resolution of the structure, in Angströms

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
* `rfam_acc`: The family's MSA the column belongs to
* `index_ali`: Position of the column in the alignment (starts at 1)
* `cm_coord`: Position of the column in the Rfam covariance model of the family (starts at 1). The value is NULL in portions that are insertions compared to the model.
* `freq_A`, `freq_C`, `freq_G`, `freq_U`, `freq_other`: Nucleotide frequencies in the alignment at this position
* `gap_percent`: The frequencies of gaps at this position in the alignment (between 0.0 and 1.0)
* `consensus`: A consensus character (ACGUN or '-') summarizing the column, if we can. If >75% of the sequences are gaps at this position, the gap is picked as consensus. Otherwise, A/C/G/U is chosen if >50% of the non-gap positions are A/C/G/U. Otherwise, N is the consensus.
* `cons_sec_struct`: A consensus secondary structure for the RNAs of the family, obtained from the Infernal alignement. The structure is well-nested (no pseudoknots) and the possible symbols are '.' (unpaired) or '(' and ')' (paired). The field is NULL in portions that are insertions compared to the Rfam model of the family, meaning that their is no consensus on the structure.

There always is an entry, for each family (rfam_acc), with index_ali = 0; gap_percent = 1.0; and nucleotide frequencies set to 0.0. This entry is used when the nucleotide frequencies cannot be determined because of local alignment issues.

## Table `re_mapping`, to map a nucleotide to an alignment column
* `remapping_id`: A unique identifier
* `chain_id`: The chain which is mapped to an alignment
* `index_chain`: The absolute position of the nucleotide in the chain (from 1 to X)
* `index_ali` The position of that nucleotide in its family alignment
