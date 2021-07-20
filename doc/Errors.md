# Warnings and errors in RNANet

Use Ctrl + F on this page to look for your error message in the list.

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

* **Something's wrong with the SQL database. Check mysql-rfam-public.ebi.ac.uk status and try again later. Not printing statistics.** : We cannot retrieve family statistics from Rfam public server. Check if you can connect to it by hand : `mysql -u rfamro -P 4497 -D Rfam -h mysql-rfam-public.ebi.ac.uk`. if not, check that the port 4497 is opened on your network.

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