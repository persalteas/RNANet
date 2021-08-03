# Known Issues

## Annotation and numbering issues
* Some GDPs that are listed as HETATMs in the mmCIF files are not detected correctly to be real nucleotides. (e.g. 1e8o-E)
* Some chains are truncated in different pieces with different chain names. Reason unknown (e.g. 6ztp-AX)
* Some chains are not correctly renamed A in the produced separate files (e.g. 1d4r-B)

## Alignment issues
* Chain names appear in triple in the FASTA header (e.g. 1d4r[1]-B 1d4r[1]-B 1d4r[1]-B)

# Known feature requests
* Automated annotation of detected Recurrent Interaction Networks (RINs), see http://carnaval.lri.fr/ .
* Possibly, automated detection of HLs and ILs from the 3D Motif Atlas (BGSU). Maybe. Their own website already does the job.
* Weight sequences in alignment to give more importance to rarer sequences 
* Give both gap_percent and insertion_gap_percent
* A field estimating the quality of the sequence alignment in table family.
* Possibly, more metrics about the alignments coming from Infernal.
* Run cmscan ourselves from the NDB instead of using Rfam-PDB mappings ? (Iff this actually makes a real difference, untested yet)
* Use and save Infernal alignment bounds and truncation information
* Save if a chain is a representative in BGSU list
* Annotate unstructured regions (on a nucleotide basis)

## Technical to-do list
* `cmalign --merge` is now deprecated, we use `esl-alimerge` instead. But, esl is a single-core process. We should run the merges of alignements of different families in parallel to save some time [TODO]. 
