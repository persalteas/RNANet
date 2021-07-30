
# FAQ

* **What is the difference between . and - in alignments ?**

In `cmalign` alignments, - means a nucleotide is missing compared to the covariance model. It represents a deletion. The dot '.' indicates that another chain has an insertion compared to the covariance model. The current chains does not lack anything, it's another which has more.

In the final filtered alignment that we provide for download, the same rule applies, but on top of that, some '.' are replaced by '-' when a gap in the 3D structure (a missing, unresolved nucleotide) is mapped to an insertion gap.

* **What are the cmalign options for ?**

From Infernal's user guide, we can quote that Infernal uses an HMM banding technique to accelerate alignment by default. It also takes care of 3' or 5' truncated sequences to be aligned correctly (and we have some).
First, one can choose an algorithm, between `--optacc` (maximizing posterior probabilities, the default) and `--cyk` (maximizing likelihood).

Then, the use of bands allows faster and more memory efficient computation, at the price of the guarantee of determining the optimal alignment. Bands can be disabled using the `--nonbanded` option. A best idea would be to control the threshold of probability mass to be considered negligible during HMM band calculation with the `--tau` parameter. Higher values of Tau yield greater speedups and lower memory usage, but a greater chance to miss the optimal alignment. In practice, the algorithm explores several Tau values (increasing it by a factor 2.0 from the original `--tau` value) until the DP matrix size falls below the threshold given by `--mxsize` (default 1028 Mb) or the value of `--maxtau` is reached (in this case, the program fails). One can disable this exploration with option `--fixedtau`. The default value of `--tau` is 1e-7, the default `--maxtau` is 0.05. Basically, you may decide on a value of `--mxsize` by dividing your available RAM by the number of cores used with cmalign. If necessary, you may use less cores than you have, using option `--cpu`.

Finally, if using `--cyk --nonbanded --notrunc --noprob`, one can use the `--small` option to align using the divide-and-conquer CYK algorithm from Eddy 2002, requiring a very few memory but a lot of time. The major drawback of this is that it requires `--notrunc` and `--noprob`, so we give up on the correct alignment of truncated sequences, and the computation of posterior probabilities.

* **Why are there some gap-only columns in the alignment ?**

These columns are not completely gap-only, they contain at least one dash-gap '-'. This means an actual, physical nucleotide which should exist in the 3D structure should be located there. The previous and following nucleotides are **not** contiguous in space in 3D.

* **Why is the numbering of residues in my 3D chain weird ?**

Probably because the numbering in the original chain already was a mess, and the RNANet re-numbering process failed to understand it correctly. If you ran RNANet yourself, check the `logs/` folder and find your chain's log. It will explain you how it was re-numbered.

* **What is your standardized way to re-number residues ?**

We first remove the nucleotides whose number is outside the family mapping (if any). Then, we renumber the following way:

    0) For truncated chains, we shift the numbering of every nucleotide so that the first nucleotide is 1.
    1) We identify duplicate residue numbers and increase by 1 the numbering of all nucleotides starting at the duplicate, recursively, and until we find a gap in the numbering suite. If no gap is found, residue numbers are shifted until the end of the chain.
    2) We proceed the similar way for nucleotides with letter numbering (e.g. 17, 17A and 17B will be renumbered to 17, 18 and 19, and the following nucleotides in the chain are also shifted).
    3) Nucleotides with partial numbering and a letter are hopefully detected and processed with their correct numbering (e.g. in ...1629, 1630, 163B, 1631, ... the residue 163B has nothing to do with number 163 or 164, the series will be renumbered 1629, 1630, 1631, 1632 and the following will be shifted).
    4) Nucleotides numbered -1 at the begining of a chain are shifted (with the following ones) to 1.
    5) Ligands at the end of the chain are removed. Is detected as ligand any residue which is not A/C/G/U and has no defined puckering or no defined torsion angles. Residues are also considered to be ligands if they are at the end of the chain with a residue number which is more than 50 than the previous residue (ligands are sometimes numbered 1000 or 9999). Finally, residues "GNG", "E2C", "OHX", "IRI", "MPD", "8UZ" at then end of a chain are removed.
    6) Ligands at the begining of a chain are removed. DSSR annotates them with index_chain 1, 2, 3..., so we can detect that there is a redundancy with the real nucleotides 1, 2, 3. We keep only the first, which hopefully is the real nucleotide. We also remove the ones that have a negative number (since we renumbered the truncated chain to 1, some became negative).
    7) Nucleotides with creative, disruptive numbering are attempted to be detected and renumbered, even if the numbers fell out of the family mapping interval. For example, the suite ... 1003, 2003, 3003, 1004... will be renumbered ...1003, 1004, 1005, 1006 ... and the following accordingly.
    8) Nucleotides missing from portions not resolved in 3D are created as gaps, with correct numbering, to fill the portion between the previous and the following resolved ones.

* **What are the versions of the dependencies you use ?**

`cmalign` is v1.1.4, `sina` is v1.6.0, `x3dna-dssr` is v2.3.2-2021jun29, Biopython is v1.78.
    