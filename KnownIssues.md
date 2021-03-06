# Known Issues

## Annotation and numbering issues
* Some GDPs that are listed as HETATMs in the mmCIF files are not detected correctly to be real nucleotides. (e.g. 1e8o-E)
* Some chains are truncated in different pieces with different chain names. Reason unknown (e.g. 6ztp-AX)
* Some chains are not correctly renamed A in the produced separate files (e.g. 1d4r-B)

## Alignment issues
* [SOLVED] Filtered alignments are shorter than the number of alignment columns saved to the SQL table `align_column`
* Chain names appear in triple in the FASTA header (e.g. 1d4r[1]-B 1d4r[1]-B 1d4r[1]-B)

## Technical running issues
* [SOLVED] Files produced by Docker containers are owned by root and require root permissions to be read 
* [SOLVED] SQLite WAL files are not deleted properly

# Known feature requests
* [DONE] Get filtered versions of the sequence alignments containing the 3D chains, publicly available for download
* [DONE] Get a consensus residue for each alignement column
* [DONE] Get an option to limit the number of cores 
* [UPCOMING] Automated annotation of detected Recurrent Interaction Networks (RINs), see http://carnaval.lri.fr/ .
* [UPCOMING] Possibly, automated detection of HLs and ILs from the 3D Motif Atlas (BGSU). Maybe. Their own website already does the job.
* A field estimating the quality of the sequence alignment in table family.
* Possibly, more metrics about the alignments coming from Infernal.
* Move to SILVA LSU release 138