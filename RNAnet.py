#!/usr/bin/python3

# check Python version before everything
import platform
a = ["3.8", platform.python_version()]
a.sort()
if a[0] != "3.8":
    print(f"Python is too old: {platform.python_version()}")
    print("Please use version 3.8 or newer.")
    exit(1)

import Bio.PDB as pdb
import concurrent.futures
import getopt
import gzip
import io
import json
import numpy as np
import os
import pandas as pd
import pickle
import psutil
import re
import requests
import signal
import sqlalchemy
import sqlite3
import subprocess
import sys
import time
import traceback
import warnings
from functools import partial, wraps
from multiprocessing import Pool, Manager
from time import sleep
from tqdm import tqdm
from setproctitle import setproctitle
from Bio import AlignIO, SeqIO
from Bio.SeqIO.FastaIO import FastaIterator, SimpleFastaParser
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

runDir = os.getcwd()

def trace_unhandled_exceptions(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            s = traceback.format_exc()
            with open(runDir + "/errors.txt", "a") as f:
                f.write("Exception in "+func.__name__+"\n")
                f.write(s)
                f.write("\n\n")

            warn('Exception in '+func.__name__, error=True)
            print(s)
    return wrapped_func


pd.set_option('display.max_rows', None)
sqlite3.enable_callback_tracebacks(True)
sqlite3.register_adapter(np.int64, lambda val: int(val))        # Tell Sqlite what to do with <class numpy.int64> objects ---> convert to int
sqlite3.register_adapter(np.float64, lambda val: float(val))    # Tell Sqlite what to do with <class numpy.float64> objects ---> convert to float

m = Manager()
running_stats = m.list()
running_stats.append(0)  # n_launched
running_stats.append(0)  # n_finished
running_stats.append(0)  # n_skipped
path_to_3D_data = "tobedefinedbyoptions"
path_to_seq_data = "tobedefinedbyoptions"
python_executable = "python"+".".join(platform.python_version().split('.')[:2])  # Cuts python3.8.1 into python3.8 for example.
validsymb = '\U00002705'
warnsymb = '\U000026A0'
errsymb = '\U0000274C'

LSU_set = {"RF00002", "RF02540", "RF02541",
           "RF02543", "RF02546"}   # From Rfam CLAN 00112
SSU_set = {"RF00177", "RF02542",  "RF02545",
           "RF01959", "RF01960"}  # From Rfam CLAN 00111
no_nts_set = set()
weird_mappings = set()


class MutableFastaIterator(FastaIterator):
    """
    Same as Biopython's FastaIterator, but uses Bio.Seq.MutableSeq objects instead of Bio.Seq.Seq.
    """
    def iterate(self, handle):
        """Parse the file and generate SeqRecord objects."""
        title2ids = self.title2ids
        if title2ids:
            for title, sequence in SimpleFastaParser(handle):
                id, name, descr = title2ids(title)
                yield SeqRecord(MutableSeq(sequence), id=id, name=name, description=descr)
        else:
            for title, sequence in SimpleFastaParser(handle):
                try:
                    first_word = title.split(None, 1)[0]
                except IndexError:
                    assert not title, repr(title)
                    # Should we use SeqRecord default for no ID?
                    first_word = ""
                yield SeqRecord(
                    MutableSeq(sequence), id=first_word, name=first_word, description=title,
                )


class SelectivePortionSelector(object):
    """Class passed to MMCIFIO to select some chain portions in an MMCIF file.

    Validates every chain, residue, nucleotide, to say if it is in the selection or not.
    """

    def __init__(self, model_id, chain_id, valid_resnums, khetatm):
        self.chain_id = chain_id
        self.resnums = valid_resnums  # list of strings, that are mostly ints
        self.pdb_model_id = model_id
        self.hydrogen_regex = re.compile("[123 ]*H.*")
        self.keep_hetatm = khetatm

    def accept_model(self, model):
        return int(model.get_id() == self.pdb_model_id)

    def accept_chain(self, chain):
        return int(chain.get_id() == self.chain_id)

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()

        # Refuse waters and magnesium ions
        if hetatm_flag in ["W", "H_MG"]:
            return int(self.keep_hetatm)

        # Accept the residue if it is in the right interval:
        if icode == " " and len(self.resnums):
            return int(str(resseq) in self.resnums)
        elif icode != " " and len(self.resnums):
            return int(str(resseq)+icode in self.resnums)
        else:  # len(resnum) == 0, we don't use mappings (--no-homology option)
            return 1

    def accept_atom(self, atom):

        # Refuse hydrogens
        if self.hydrogen_regex.match(atom.get_id()):
            return 0

        # Accept all atoms otherwise.
        return 1


class Chain:
    """ 
    The object which stores all our data and the methods to process it.

    Chains accumulate information through this scipt, and are saved to files at the end of major steps.
    """

    def __init__(self, pdb_id, pdb_model, pdb_chain_id, chain_label, eq_class, rfam="", inferred=False, pdb_start=None, pdb_end=None):
        self.pdb_id = pdb_id                    # PDB ID
        self.pdb_model = int(pdb_model)         # model ID, starting at 1
        self.pdb_chain_id = pdb_chain_id        # chain ID (mmCIF), multiple letters
        if len(rfam):
            self.mapping = Mapping(chain_label, rfam, pdb_start, pdb_end, inferred)
        else:
            self.mapping = None
        self.eq_class = eq_class                # BGSU NR list class id
        self.chain_label = chain_label          # chain pretty name
        self.file = ""                          # path to the 3D PDB file
        self.seq = ""                           # sequence with modified nts
        self.seq_to_align = ""                  # sequence with modified nts replaced by N, but gaps can exist
        self.length = -1                        # length of the sequence (missing residues are not counted)
        self.full_length = -1                   # length of the chain extracted from source structure ([start; stop] interval, or a subset for inferred mappings)
        self.delete_me = False                  # an error occured during production/parsing
        self.error_messages = ""                # Error message(s) if any
        self.db_chain_id = -1                   # index of the RNA chain in the SQL database, table chain

    def __str__(self):
        return self.pdb_id + '[' + str(self.pdb_model) + "]-" + self.pdb_chain_id

    def __eq__(self, other):
        return self.chain_label == other.chain_label and str(self) == str(other)

    def __hash__(self):
        return hash((self.pdb_id, self.pdb_model, self.pdb_chain_id, self.chain_label))

    def extract(self, df, khetatm) -> None:
        """ Extract the part which is mapped to Rfam from the main CIF file and save it to another file.
        """
        setproctitle(f"RNANet.py {self.chain_label} extract()")

        if self.mapping is not None:
            status = f"Extract {self.mapping.nt_start}-{self.mapping.nt_end} atoms from {self.pdb_id}-{self.pdb_chain_id}"
            self.file = path_to_3D_data+"rna_mapped_to_Rfam/"+self.chain_label+".cif"
        else:
            status = f"Extract {self.pdb_id}-{self.pdb_chain_id}"
            self.file = path_to_3D_data+"rna_only/"+self.chain_label+".cif"

        # Check if file exists, if yes, abort (do not recompute)
        if os.path.exists(self.file):
            notify(status, "using previous file")
            return

        model_idx = self.pdb_model - (self.pdb_model > 0)   # because arrays start at 0, models start at 1

        with warnings.catch_warnings():
            # Ignore the PDB problems. This mostly warns that some chain is discontinuous.
            warnings.simplefilter('ignore', pdb.PDBExceptions.PDBConstructionWarning)
            warnings.simplefilter('ignore', pdb.PDBExceptions.BiopythonWarning)

            # Load the whole mmCIF into a Biopython structure object:
            mmcif_parser = pdb.MMCIFParser()
            try:
                s = mmcif_parser.get_structure(self.pdb_id, path_to_3D_data + "RNAcifs/"+self.pdb_id+".cif")
            except ValueError as e:
                warn(f"ValueError in {self.chain_label} CIF file: {e}")
                self.delete_me = True
                return
            except IndexError as e:
                warn(f"IndexError in {self.chain_label} CIF file: {e}")
                self.delete_me = True
                return

            if self.mapping is not None:
                valid_set = set(df.old_nt_resnum)
            else:
                valid_set = set()

            # Define a selection
            sel = SelectivePortionSelector(model_idx, self.pdb_chain_id, valid_set, khetatm)

            # Save that selection on the mmCIF object s to file
            ioobj = pdb.MMCIFIO()
            ioobj.set_structure(s)
            ioobj.save(self.file, sel)

        notify(status)

    @trace_unhandled_exceptions
    def extract_3D_data(self, save_logs=True):
        """ Maps DSSR annotations to the chain. """

        setproctitle(f"RNANet.py {self.chain_label} extract_3D_data()")

        ############################################
        # Load the mmCIF annotations from file
        ############################################
        try:
            with open(path_to_3D_data + "annotations/" + self.pdb_id + ".json", 'r') as json_file:
                json_object = json.load(json_file)
            notify(f"Read {self.pdb_id} DSSR annotations")
        except json.decoder.JSONDecodeError as e:
            warn("Could not load "+self.pdb_id+f".json with JSON package: {e}", error=True)
            self.delete_me = True
            self.error_messages = f"Could not load existing {self.pdb_id}.json file: {e}"
            return None

        # Print eventual warnings given by DSSR, and abort if there are some
        if "warning" in json_object.keys():
            warn(f"found DSSR warning in annotation {self.pdb_id}.json: {json_object['warning']}. Ignoring {self.chain_label}.")
            if "no nucleotides" in json_object['warning']:
                no_nts_set.add(self.pdb_id)
            self.delete_me = True
            self.error_messages = f"DSSR warning {self.pdb_id}.json: {json_object['warning']}. Ignoring {self.chain_label}."
            return None

        ############################################
        # Create the data-frame
        ############################################
        try:
            # Create the Pandas DataFrame for the nucleotides of the right chain
            nts = json_object["nts"]                        # sub-json-object
            df = pd.DataFrame(nts)                          # conversion to dataframe
            df = df[df.chain_name == self.pdb_chain_id]     # keeping only this chain's nucleotides

            # Assert nucleotides of the chain are found
            if df.empty:
                warn(f"Could not find nucleotides of chain {self.pdb_chain_id} in annotation {self.pdb_id}.json. Ignoring chain {self.chain_label}.")
                no_nts_set.add(self.pdb_id)
                self.delete_me = True
                self.error_messages = f"Could not find nucleotides of chain {self.pdb_chain_id} in annotation {self.pdb_id}.json. Either there is a problem with {self.pdb_id} mmCIF download, or the bases are not resolved in the structure. Delete it and retry."
                return None

            # Remove low pertinence or undocumented descriptors, convert angles values
            cols_we_keep = ["index_chain", "nt_resnum", "nt_name", "nt_code", "nt_id", "dbn", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                            "epsilon_zeta", "bb_type", "chi", "glyco_bond", "form", "ssZp", "Dp", "eta", "theta", "eta_prime", "theta_prime", "eta_base", "theta_base",
                            "v0", "v1", "v2", "v3", "v4", "amplitude", "phase_angle", "puckering"]
            df = df[cols_we_keep]
            df.loc[:, ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'epsilon_zeta', 'chi', 'v0', 'v1', 'v2', 'v3', 'v4',  # Conversion to radians
                       'eta', 'theta', 'eta_prime', 'theta_prime', 'eta_base', 'theta_base', 'phase_angle']] *= np.pi/180.0
            df.loc[:, ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'epsilon_zeta', 'chi', 'v0', 'v1', 'v2', 'v3', 'v4',  # mapping [-pi, pi] into [0, 2pi]
                       'eta', 'theta', 'eta_prime', 'theta_prime', 'eta_base', 'theta_base', 'phase_angle']] %= (2.0*np.pi)

        except KeyError as e:
            warn(f"Error while parsing DSSR {self.pdb_id}.json output:{e}", error=True)
            self.delete_me = True
            self.error_messages = f"Error while parsing DSSR's json output:\n{e}"
            return None

        #############################################
        # Select the nucleotides we need
        #############################################

        # Remove nucleotides of the chain that are outside the Rfam mapping, if any
        if self.mapping is not None:
            if self.mapping.nt_start > self.mapping.nt_end:
                warn(f"Mapping is reversed, this case is not supported (yet). Ignoring chain {self.chain_label}.")
                self.delete_me = True
                self.error_messages = f"Mapping is reversed, this case is not supported (yet)."
                return None
            df = self.mapping.filter_df(df)

        # Duplicate residue numbers : shift numbering
        while True in df.duplicated(['nt_resnum']).values:
            i = df.duplicated(['nt_resnum']).values.tolist().index(True)
            duplicates = df[df.nt_resnum == df.iloc[i, 1]]
            n_dup = len(duplicates.nt_resnum)
            index_last_dup = duplicates.index_chain.iloc[-1] - 1
            if self.mapping is not None:
                self.mapping.log(f"Shifting nt_resnum numbering because of {n_dup} duplicate residues {df.iloc[i,1]}")

            try:
                if i > 0 and index_last_dup + 1 < len(df.index) and df.iloc[i, 1] == df.iloc[i-1, 1] and df.iloc[index_last_dup + 1, 1] - 1 > df.iloc[index_last_dup, 1]:
                    # The redundant nts are consecutive in the chain (at the begining at least), and there is a gap at the end

                    if duplicates.iloc[n_dup-1, 0] - duplicates.iloc[0, 0] + 1 == n_dup:
                        # They are all contiguous in the chain
                        # 4v9n-DA case (and similar ones) : 610-611-611A-611B-611C-611D-611E-611F-611G-617-618...
                        # there is a redundancy (611) followed by a gap (611-617).
                        # We want the redundancy to fill the gap.
                        df.iloc[i:i+n_dup-1, 1] += 1
                    else:
                        # We solve the problem continous component by continuous component
                        for j in range(1, n_dup+1):
                            if duplicates.iloc[j, 0] == 1 + duplicates.iloc[j-1, 0]:  # continuous
                                df.iloc[i+j-1, 1] += 1
                            else:
                                break
                elif df.iloc[i, 1] == df.iloc[i-1, 1]:
                    # Common 4v9q-DV case (and similar ones) : e.g. chains contains 17 and 17A which are both read 17 by DSSR.
                    # Solution : we shift the numbering of 17A (to 18) and the following residues.
                    df.iloc[i:, 1] += 1
                elif duplicates.iloc[0, 0] == 1 and df.iloc[i, 0] == 3:
                    # 4wzo_1_1J case, there is a residue numbered -1 and read as 1 before the number 0.
                    df.iloc[1:, 1] += 1
                    df.iloc[0, 1] = 0
                else:
                    # 4v9k-DA case (and similar ones) : the nt_id is not the full nt_resnum: ... 1629 > 1630 > 163B > 1631 > ...
                    # Here the 163B is read 163 by DSSR, but there already is a residue 163.
                    # Solution : set nt_resnum[i] to nt_resnum[i-1] + 1, and shift the following by 1.
                    df.iloc[i, 1] = 1 + df.iloc[i-1, 1]
                    df.iloc[i+1:, 1] += 1
            except:
                warn(f"Error with parsing of {self.chain_label} duplicate residue numbers. Ignoring it.")
                self.delete_me = True
                self.error_messages = f"Error with parsing of duplicate residues numbers."
                return None

        # Search for ligands at the end of the selection
        # Drop ligands detected as residues by DSSR, by detecting several markers
        while ( 
                len(df.index_chain) and df.iloc[-1, 2] not in ["A", "C", "G", "U"] 
                and (
                    (df.iloc[[-1]][["alpha", "beta", "gamma", "delta", "epsilon",
                                    "zeta", "v0", "v1", "v2", "v3", "v4"]].isna().values).all()
                    or (df.iloc[[-1]].puckering == '').any()
                )
                # large nt_resnum gap between the two last residues
                or (len(df.index_chain) >= 2 and df.iloc[-1, 1] > 50 + df.iloc[-2, 1])
                or (len(df.index_chain) and df.iloc[-1, 2] in ["GNG", "E2C", "OHX", "IRI", "MPD", "8UZ"])
        ):
            if self.mapping is not None:
                self.mapping.log("Droping ligand:")
                self.mapping.log(df.tail(1))
            df = df.head(-1)

        # Duplicates in index_chain : drop, they are ligands
        # e.g. 3iwn_1_B_1-91, ligand C2E has index_chain 1 (and nt_resnum 601)
        duplicates = [ index for index, element in enumerate(df.duplicated(['index_chain']).values) if element ]
        if len(duplicates):
            for i in duplicates:
                warn(f"Found duplicated index_chain {df.iloc[i,0]} in {self.chain_label}. Keeping only the first.")
                if self.mapping is not None:
                    self.mapping.log(f"Found duplicated index_chain {df.iloc[i,0]}. Keeping only the first.")
            df = df.drop_duplicates("index_chain", keep="first")    # drop doublons in index_chain

        # drop eventual nts with index_chain < the first residue,
        # now negative because we renumber to 1 (usually, ligands)
        ligands = df[df.index_chain < 0]
        if len(ligands.index_chain):
            if self.mapping is not None:
                for line in ligands.iterrows():
                    self.mapping.log("Droping ligand:")
                    self.mapping.log(line)
            df = df.drop(ligands.index)

        # Find missing index_chain values
        # This happens because of resolved nucleotides that have a
        # strange nt_resnum value. Thanks, biologists ! :@ :(
        # e.g. 4v49-AA, position 5'- 1003 -> 2003 -> 1004 - 3'
        diff = set(range(df.shape[0])).difference(df['index_chain'] - 1)
        if len(diff) and self.mapping is not None:
            # warn(f"Missing residues in chain numbering: {[1+i for i in sorted(diff)]}")
            for i in sorted(diff):
                # check if a nucleotide with the correct index_chain exists in the nts object
                found = None
                for nt in nts:  # nts is the object from the loaded JSON and contains all nts
                    if nt['chain_name'] != self.pdb_chain_id:
                        continue
                    if nt['index_chain'] == i + 1 + self.mapping.st:
                        found = nt
                        break
                if found:
                    self.mapping.log(f"Residue {i+1+self.mapping.st}-{self.mapping.st} = {i+1} has been saved and renumbered {df.iloc[i,1]} instead of {found['nt_id'].replace(found['chain_name']+ '.' + found['nt_name'], '').replace('^','')}")
                    df_row = pd.DataFrame([found], index=[i])[
                        df.columns.values]
                    df_row.iloc[0, 0] = i+1          # index_chain
                    df_row.iloc[0, 1] = df.iloc[i, 1]  # nt_resnum
                    df = pd.concat([df.iloc[:i], df_row, df.iloc[i:]])
                    df.iloc[i+1:, 1] += 1
                else:
                    warn(f"Missing index_chain {i} in {self.chain_label} !")

        # Assert some nucleotides still exist
        try:
            # update length of chain from nt_resnum point of view
            l = df.iloc[-1, 1] - df.iloc[0, 1] + 1
        except IndexError:
            warn(f"Could not find real nucleotides of chain {self.pdb_chain_id} between {self.mapping.nt_start} and "
                 f"{self.mapping.nt_end} ({'not ' if not self.mapping.inferred else ''}inferred). Ignoring chain {self.chain_label}.")
            no_nts_set.add(self.pdb_id)
            self.delete_me = True
            self.error_messages = f"Could not find nucleotides of chain {self.pdb_chain_id} in annotation {self.pdb_id}.json. Either there is a problem with {self.pdb_id} mmCIF download, or the bases are not resolved in the structure. Delete it and retry."
            return None

        # Add eventual missing rows because of unsolved residues in the chain.
        # Sometimes, the 3D structure is REALLY shorter than the family it's mapped to,
        # especially with inferred mappings (e.g. 6hcf chain 82 to RF02543)
        #
        # There are several numbering scales in use here:
        # nt_numbering: the residue numbers in the RNA molecule. It can be any range. Unresolved residues count for 1.
        # index_chain and self.length: the nucleotides positions within the 3D chain. It starts at 1, and unresolved residues are skipped.
        # pdb_start/pdb_end: the RNA molecule portion to extract and map to Rfam. it is related to the index_chain scale.
        #
        # example on 6hcf chain 82:
        # RNA molecule          1 |------------------------------------------- ... ----------| theoretic length of a large subunit.
        # portion solved in 3D  1 |--------------|79 85|------------| 156
        # Rfam mapping           3 |------------------------------------------ ... -------| 3353 (yes larger, 'cause it could be inferred)
        # nt resnum              3 |--------------------------------|  156
        # index_chain            1 |-------------|77 83|------------|  154
        # expected data point    1 |--------------------------------|  154
        #
        
        if l != len(df['index_chain']): # if some residues are missing, len(df['index_chain']) < l
            resnum_start = df.iloc[0, 1]
            # the rowIDs the missing nucleotides would have (rowID = index_chain - 1 = nt_resnum - resnum_start)
            diff = set(range(l)).difference(df['nt_resnum'] - resnum_start)
            for i in sorted(diff):
                # Add a row at position i
                df = pd.concat([df.iloc[:i],
                                pd.DataFrame({"index_chain": i+1, "nt_resnum": i+resnum_start,
                                              "nt_id": "not resolved", "nt_code": '-', "nt_name": '-'}, index=[i]),
                                df.iloc[i:]])
                # Increase the index_chain of all following lines
                df.iloc[i+1:, 0] += 1
            df = df.reset_index(drop=True)
        self.full_length = len(df.index_chain)

        #######################################
        # Compute new features
        #######################################

        # Add a sequence column just for the alignments
        df['nt_align_code'] = [str(x).upper()
                               .replace('NAN', '-') # Unresolved nucleotides are gaps
                               .replace('?', 'N')   # Unidentified residues, let's delete them
                               .replace('T', 'U')   # 5MU are modified to t by DSSR, which gives T
                               .replace('P', 'U')   # Pseudo-uridines, but it is not really right to change them to U, see DSSR paper, Fig 2
                               for x in df['nt_code']]
        df['nt_align_code'] = [ x if x in "ACGU-" else 'N' for x in df['nt_align_code'] ] # All other modified nucleotides are transformed to N

        # One-hot encoding sequence
        df["is_A"] = [1 if x == "A" else 0 for x in df["nt_code"]]
        df["is_C"] = [1 if x == "C" else 0 for x in df["nt_code"]]
        df["is_G"] = [1 if x == "G" else 0 for x in df["nt_code"]]
        df["is_U"] = [1 if x == "U" else 0 for x in df["nt_code"]]
        df["is_other"] = [0 if x in "ACGU" else 1 for x in df["nt_code"]]
        df["nt_position"] = [ float(i+1)/self.full_length for i in range(self.full_length) ]

        # Iterate over pairs to identify base-base interactions
        res_ids = list(df['nt_id'])  # things like "chainID.C4, chainID.U5"
        paired = [''] * self.full_length
        pair_type_LW = [''] * self.full_length
        pair_type_DSSR = [''] * self.full_length
        interacts = [0] * self.full_length
        if "pairs" in json_object.keys():
            pairs = json_object["pairs"]
            for p in pairs:
                nt1 = p["nt1"]
                nt2 = p["nt2"]
                lw_pair = p["LW"]
                dssr_pair = p["DSSR"]
                if nt1 in res_ids:
                    nt1_idx = res_ids.index(nt1)
                else:
                    nt1_idx = -1
                if nt2 in res_ids:
                    nt2_idx = res_ids.index(nt2)
                else:
                    nt2_idx = -1

                # set nucleotide 1
                if nt1 in res_ids:
                    interacts[nt1_idx] += 1
                    if paired[nt1_idx] == "":
                        pair_type_LW[nt1_idx] = lw_pair
                        pair_type_DSSR[nt1_idx] = dssr_pair
                        paired[nt1_idx] = str(nt2_idx + 1)  # index + 1 is actually index_chain.
                    else:
                        pair_type_LW[nt1_idx] += ',' + lw_pair
                        pair_type_DSSR[nt1_idx] += ',' + dssr_pair
                        paired[nt1_idx] += ',' + str(nt2_idx + 1)   # index + 1 is actually index_chain.

                # set nucleotide 2 with the opposite base-pair
                if nt2 in res_ids:
                    interacts[nt2_idx] += 1
                    if paired[nt2_idx] == "":
                        pair_type_LW[nt2_idx] = lw_pair[0] + lw_pair[2] + lw_pair[1]
                        pair_type_DSSR[nt2_idx] = dssr_pair[0] + dssr_pair[3] + dssr_pair[2] + dssr_pair[1]
                        paired[nt2_idx] = str(nt1_idx + 1)
                    else:
                        pair_type_LW[nt2_idx] += ',' + lw_pair[0] + lw_pair[2] + lw_pair[1]
                        pair_type_DSSR[nt2_idx] += ',' + dssr_pair[0] + dssr_pair[3] + dssr_pair[2] + dssr_pair[1]
                        paired[nt2_idx] += ',' + str(nt1_idx + 1)

        # transform nt_id to shorter values
        df['old_nt_resnum'] = [ n.replace(self.pdb_chain_id+'.'+name, '').replace('^', '').replace('/', '') for n, name in zip(df.nt_id, df.nt_name) ]

        df['paired'] = paired
        df['pair_type_LW'] = pair_type_LW
        df['pair_type_DSSR'] = pair_type_DSSR
        df['nb_interact'] = interacts

        # remove now useless descriptors
        df = df.drop(['nt_id', 'nt_resnum'], axis=1)

        self.seq = "".join(df.nt_code)
        self.seq_to_align = "".join(df.nt_align_code)
        self.length = len([x for x in self.seq_to_align if x != "-"])

        # Remove too short chains
        if self.length < 5:
            warn(f"{self.chain_label} sequence is too short, let's ignore it.\t")
            self.delete_me = True
            self.error_messages = "Sequence is too short. (< 5 resolved nts)"
            return None

        # Log chain info to file
        if save_logs and self.mapping is not None:
            self.mapping.to_file(self.chain_label+".log")

        return df

    def register_chain(self, df):
        """Saves the extracted 3D data to the database.
        """

        setproctitle(f"RNANet.py {self.chain_label} register_chain()")

        with sqlite3.connect(runDir+"/results/RNANet.db", timeout=10.0) as conn:
            conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
            # Register the chain in table chain
            if self.mapping is not None:
                sql_execute(conn, f"""  INSERT INTO chain 
                                        (structure_id, chain_name, pdb_start, pdb_end, rfam_acc, eq_class, inferred, issue)
                                        VALUES 
                                        (?, ?, ?, ?, ?, ?, ?, ?)
                                        ON CONFLICT(structure_id, chain_name, rfam_acc) DO
                                        UPDATE SET  pdb_start=excluded.pdb_start, 
                                                    pdb_end=excluded.pdb_end, 
                                                    eq_class=excluded.eq_class,
                                                    inferred=excluded.inferred, 
                                                    issue=excluded.issue;""",
                            data=(str(self.pdb_id), str(self.pdb_chain_id),
                                  int(self.mapping.nt_start), int(self.mapping.nt_end),
                                  str(self.mapping.rfam_acc), str(self.eq_class),
                                  int(self.mapping.inferred), int(self.delete_me)))
                # get the chain id
                self.db_chain_id = sql_ask_database(conn, f"""SELECT (chain_id) FROM chain 
                                                                WHERE structure_id='{self.pdb_id}' 
                                                                AND chain_name='{self.pdb_chain_id}' 
                                                                AND rfam_acc='{self.mapping.rfam_acc}'
                                                                AND eq_class='{self.eq_class}';"""
                                                    )[0][0]
            else:
                sql_execute(conn, """INSERT INTO chain (structure_id, chain_name, rfam_acc, eq_class, issue) VALUES (?, ?, 'unmappd', ?, ?) 
                                   ON CONFLICT(structure_id, chain_name, rfam_acc) DO UPDATE SET issue=excluded.issue, eq_class=excluded.eq_class;""",
                            data=(str(self.pdb_id), str(self.pdb_chain_id), str(self.eq_class), int(self.delete_me)))
                self.db_chain_id = sql_ask_database(conn, f"""SELECT (chain_id) FROM chain 
                                                                WHERE structure_id='{self.pdb_id}' 
                                                                AND chain_name='{self.pdb_chain_id}' 
                                                                AND eq_class='{self.eq_class}'
                                                                AND rfam_acc = 'unmappd';"""
                                                    )[0][0]

            # Add the nucleotides if the chain is not an issue
            if df is not None and not self.delete_me:   # double condition is theoretically redundant here, but you never know
                sql_execute(conn, f"""INSERT OR IGNORE INTO nucleotide 
                                        (chain_id, index_chain, nt_name, nt_code, dbn, alpha, beta, gamma, delta, epsilon, zeta,
                                        epsilon_zeta, bb_type, chi, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
                                        v0, v1, v2, v3, v4, amplitude, phase_angle, puckering, nt_align_code, is_A, is_C, is_G, is_U, is_other, nt_position, 
                                        old_nt_resnum, paired, pair_type_LW, pair_type_DSSR, nb_interact)
                                        VALUES ({self.db_chain_id}, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                            many=True, data=list(df.to_records(index=False)), warn_every=10)


class Job:
    """ This class contains information about a task to run later.

    This could be a system command or the execution of a Python function.
    Time and memory usage of a job can be monitored.
    """

    def __init__(self, results="", command=[], function=None, args=[], how_many_in_parallel=0, priority=1, timeout=None, checkFunc=None, checkArgs=[], label=""):
        self.cmd_ = command             # A system command to run
        self.func_ = function           # A python function to run
        self.args_ = args               # The args tuple of the function to run
        self.checkFunc_ = checkFunc     # A function to check if the Job as already been executed before (and abort execution if yes)
        self.checkArgs_ = checkArgs     # Arguments for the checkFunc
        self.results_file = results     # A filename where the job stores its results, to check for existence before execution
        self.priority_ = priority       # Priority of the job in a list of jobs (Jobs with priority 1 are processed first, then priority 2, etc. Unrelated to processes priority.)
        self.timeout_ = timeout         # Abort the job if taking too long
        self.comp_time = -1             # Time to completion of the job. -1 means 'not executed yet'
        self.max_mem = -1               # Peak RAM+Swap usage of the job. -1 means 'not executed yet'
        self.label = label              # Title

        # Deploy the job on a Pool() started using 'how_many_in_parallel' CPUs.
        if not how_many_in_parallel:
            self.nthreads = read_cpu_number()
        elif how_many_in_parallel == -1:
            self.nthreads = read_cpu_number() - 1
        else:
            self.nthreads = how_many_in_parallel

    def __str__(self):
        if self.func_ is None:
            s = f"{self.priority_}({self.nthreads}) [{self.comp_time}]\t{self.label:25}" + " ".join(self.cmd_)
        else:
            s = f"{self.priority_}({self.nthreads}) [{self.comp_time}]\t{self.label:25}{self.func_.__name__}(" \
                + " ".join([ str(a) for a in self.args_ ]) + ")"
        return s


class Monitor:
    """ A job that simply watches the memory usage of another process. 

    Checks the RAM+Swap usage of monitored process and its children every 0.1 sec.
    Returns the peak value at the end.
    """

    def __init__(self, pid):
        self.keep_watching = True
        self.target_pid = pid

    def check_mem_usage(self):
        # Get the process object
        target_process = psutil.Process(self.target_pid)

        # Start watching
        max_mem = -1
        while self.keep_watching:
            try:
                # read memory usage
                info = target_process.memory_full_info()
                mem = info.rss + info.swap

                # Do the same for every child process
                for p in target_process.children(recursive=True):
                    info = p.memory_full_info()
                    mem += info.rss + info.swap

            except psutil.NoSuchProcess:
                # The process that we watch is finished, dead, or killed.
                self.keep_watching = False
            finally:
                # Update the peak value
                if mem > max_mem:
                    max_mem = mem
            # Wait 100 ms and loop
            sleep(0.1)

        # The watch has ended
        return max_mem


class Downloader:
    def download_Rfam_PDB_mappings(self):
        """Query the Rfam public MySQL database for mappings between their RNA families and PDB structures.

        """

        setproctitle(f"RNANet.py download_Rfam_PDB_mappings()")

        # Download PDB mappings to Rfam family
        print("> Fetching latest PDB mappings from Rfam..." + " " * 29, end='', flush=True)
        try:
            db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
            mappings = pd.read_sql('SELECT rfam_acc, pdb_id, chain, pdb_start, pdb_end, bit_score, evalue_score, cm_start, cm_end, hex_colour FROM pdb_full_region WHERE is_significant=1;',
                con=db_connection)
            mappings.to_csv(runDir + "/data/Rfam-PDB-mappings.csv")
            print(f"\t{validsymb}")
        except sqlalchemy.exc.OperationalError:  # Cannot connect :'(
            print(f"\t{errsymb}")
            # Check if a previous run succeeded (if file exists, use it)
            if os.path.isfile(runDir + "/data/Rfam-PDB-mappings.csv"):
                print("\t> Using previous version.")
                mappings = pd.read_csv(runDir + "/data/Rfam-PDB-mappings.csv")
            else:  # otherwise, abort.
                print("Can't do anything without data. Exiting.")
                raise Exception("Can't reach mysql-rfam-public.ebi.ac.uk on port 4497. Is it open on your system ?")

        return mappings

    def download_Rfam_cm(self):
        """ Download the covariance models from Rfam.

        Does not download if already there.
        """

        setproctitle(f"RNANet.py download_Rfam_cm()")

        print(f"\t> Download Rfam.cm.gz from Rfam..." + " " * 37, end='', flush=True)
        if not os.path.isfile(path_to_seq_data + "Rfam.cm"):
            try:
                subprocess.run(["wget", "ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz", "-O", path_to_seq_data + "Rfam.cm.gz"])
                print(f"\t{validsymb}", flush=True)
                print(f"\t\t> Uncompressing Rfam.cm...", end='', flush=True)
                subprocess.run(["gunzip", path_to_seq_data + "Rfam.cm.gz"], stdout=subprocess.DEVNULL)
                print(f"\t{validsymb}", flush=True)
            except:
                warn(f"Error downloading and/or extracting Rfam.cm !\t", error=True)
        else:
            print(f"{validsymb}\t(no need)", flush=True)

    def download_Rfam_family_stats(self, list_of_families):
        """Query the Rfam public MySQL database for statistics about their RNA families.

        Family ID, number of sequences identified, maximum length of those sequences.
        SETS family in the database (partially)
        """

        setproctitle(f"RNANet.py download_Rfam_family_stats()")

        try:
            db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')

            # Prepare the SQL query. It computes the length of the chains and gets the maximum length by family.
            q = """SELECT stats.rfam_acc, k.description, stats.maxlength FROM
                    (SELECT fr.rfam_acc, MAX(    
                                                (CASE WHEN fr.seq_start > fr.seq_end THEN fr.seq_start
                                                                                    ELSE fr.seq_end
                                                END)
                                                -
                                                (CASE WHEN fr.seq_start > fr.seq_end THEN fr.seq_end
                                                                                    ELSE fr.seq_start
                                                END) + 1  
                                            ) AS 'maxlength'
                        FROM full_region fr
                        GROUP BY fr.rfam_acc
                    ) as stats
                    NATURAL JOIN
                    (SELECT rfam_acc, description FROM keywords) as k;
                    """

            # Query the public database
            d = pd.read_sql(q, con=db_connection)

            # filter the results to families we are interested in
            d = d[d["rfam_acc"].isin(list_of_families)]

            print(d)

            with sqlite3.connect(runDir + "/results/RNANet.db", timeout=20.0) as conn:
                conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
                # We use the REPLACE keyword to get the latest information
                sql_execute(conn, """INSERT OR REPLACE INTO family (rfam_acc, description, max_len)
                                     VALUES (?, ?, ?);""", 
                            many=True, 
                            data=list(d.to_records(index=False))
                            )  

        except sqlalchemy.exc.OperationalError:
            warn("Something's wrong with the SQL database. Check mysql-rfam-public.ebi.ac.uk status and try again later. Not printing statistics.")

    def download_Rfam_sequences(self, rfam_acc):
        """ Downloads the unaligned sequences known related to a given RNA family.

        Actually gets a FASTA archive from the public Rfam FTP. Does not download if already there."""

        setproctitle(f"RNANet.py download_Rfam_sequences({rfam_acc})")

        if not os.path.isfile(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz"):
            for _ in range(10):  # retry 100 times if it fails
                try:
                    subprocess.run(["wget", f'ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/{rfam_acc}.fa.gz', "-O",
                                    path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    notify(f"Downloaded {rfam_acc}.fa.gz from Rfam")
                    return          # if it worked, no need to retry
                except Exception as e:
                    warn(f"Error downloading {rfam_acc}.fa.gz: {e}")
                    warn("retrying in 0.2s (worker " + str(os.getpid()) + f', try {_+1}/100)')
                    time.sleep(0.2)
            warn("Tried to reach Rfam FTP 100 times and failed. Aborting.", error=True)
        else:
            notify(f"Downloaded {rfam_acc}.fa.gz from Rfam", "already there")

    def download_BGSU_NR_list(self, res):
        """ Downloads a list of RNA 3D structures proposed by Bowling Green State University RNA research group.
        The chosen list is the one with resolution threshold just above the desired one.

        Does not remove structural redundancy.
        """

        setproctitle(f"RNANet.py download_BGSU_NR_list({res})")

        nr_code = min([i for i in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 20.0] if i >= res])
        print(f"> Fetching latest list of RNA files at {nr_code} A resolution from BGSU website...", end='', flush=True)

        # Download latest BGSU non-redundant list
        try:
            s = requests.get(f"http://rna.bgsu.edu/rna3dhub/nrlist/download/current/{nr_code}A/csv").content
            nr = open(path_to_3D_data + f"latest_nr_list_{nr_code}A.csv", 'w')
            nr.write("class,representative,class_members\n")
            nr.write(io.StringIO(s.decode('utf-8')).getvalue())
            nr.close()
        except:
            warn("Error downloading NR list !\t", error=True)

            # Try to read previous file
            if os.path.isfile(path_to_3D_data + f"latest_nr_list_{nr_code}A.csv"):
                print("\t> Use of the previous version.\t", end="", flush=True)
            else:
                return pd.DataFrame([], columns=["class", "class_members"])

        nrlist = pd.read_csv(path_to_3D_data + f"latest_nr_list_{nr_code}A.csv")
        full_structures_list = [ tuple(i[1]) for i in nrlist[['class', 'class_members']].iterrows() ]
        print(f"\t{validsymb}", flush=True)

        # The beginning of an adventure.
        return full_structures_list     # list of ( str (class), str (class_members) )

    def download_from_SILVA(self, unit):

        setproctitle(f"RNANet.py download_from_SILVA({unit})")

        if not os.path.isfile(path_to_seq_data + f"realigned/{unit}.arb"):
            try:
                print(f"Downloading {unit} from SILVA...", end='', flush=True)
                if unit == "LSU":
                    subprocess.run(["wget", "-nv", "http://www.arb-silva.de/fileadmin/arb_web_db/release_132/ARB_files/SILVA_132_LSURef_07_12_17_opt.arb.gz",
                                    "-O", path_to_seq_data + "realigned/LSU.arb.gz"])
                else:
                    subprocess.run(["wget", "-nv", "http://www.arb-silva.de/fileadmin/silva_databases/release_138/ARB_files/SILVA_138_SSURef_05_01_20_opt.arb.gz",
                                    "-O", path_to_seq_data + "realigned/SSU.arb.gz"])
            except:
                warn(f"Error downloading the {unit} database from SILVA", error=True)
                exit(1)
            subprocess.run(["gunzip", path_to_seq_data + f"realigned/{unit}.arb.gz"], stdout=subprocess.DEVNULL)
            print('\t'+validsymb)
        else:
            notify(f"Downloaded and extracted {unit} database from SILVA", "used previous file")


class Mapping:
    """
    A custom class to store more information about nucleotide mappings.
    """

    def __init__(self, chain_label, rfam_acc, pdb_start, pdb_end, inferred):
        """
        Arguments:
        rfam_acc : Rfam family accession number of the mapping
        pdb_start/pdb_end : nt_resnum start and end values in the 3D data that are mapped to the family
        inferred : wether the mapping has been inferred using BGSU's NR list
        """
        self.chain_label = chain_label
        self.rfam_acc = rfam_acc
        self.nt_start = pdb_start  # nt_resnum numbering
        self.nt_end = pdb_end  # nt_resnum numbering
        self.inferred = inferred

        self.logs = []  # Events are logged when modifying the mapping

    def filter_df(self, df):

        newdf = df.drop(df[(df.nt_resnum < self.nt_start) |
                           (df.nt_resnum > self.nt_end)].index)

        if len(newdf.index_chain) > 0:
            # everything's okay
            df = newdf
        else:
            # There were nucleotides in this chain but we removed them all while
            # filtering the ones outside the Rfam mapping.
            # This probably means that, for this chain, the mapping is relative to
            # index_chain and not nt_resnum.
            warn(f"Assuming mapping to {self.rfam_acc} is an absolute position interval.")
            weird_mappings.add(self.chain_label + "." + self.rfam_acc)
            df = df.drop(df[(df.index_chain < self.nt_start) |
                            (df.index_chain > self.nt_end)].index)

        # If, for some reason, index_chain does not start at one (e.g. 6boh, chain GB), make it start at one
        self.st = 0
        if len(df.index_chain) and df.iloc[0, 0] != 1:
            self.st = df.iloc[0, 0] - 1
            df.iloc[:, 0] -= self.st
            self.log(f"Shifting index_chain of {self.st}")

        # Check that some residues are not included by mistake:
        # e.g. 4v4t-AA.RF00382-20-55 contains 4 residues numbered 30 but actually far beyond the mapped part,
        # because the icode are not read by DSSR.
        toremove = df[df.index_chain > self.nt_end]
        if not toremove.empty:
            df = df.drop(toremove.index)
            self.log(f"Some nt_resnum values are likely to be wrong, not considering residues:")
            self.log(str(toremove))

        return df

    def log(self, message):
        if isinstance(message, str):
            self.logs.append(message+'\n')
        else:
            self.logs.append(str(message))

    def to_file(self, filename):
        if self.logs == []:
            return  # Do not create a log file if there is nothing to log

        if not os.path.exists(runDir+"/logs"):
            os.makedirs(runDir+"/logs", exist_ok=True)
        with open(runDir+"/logs/"+filename, "w") as f:
            f.writelines(self.logs)


class Pipeline:
    def __init__(self):
        self.dl = Downloader()
        self.known_issues = []  # list of chain_labels to ignore
        self.update = []        # list of Chain() objects we need to extract 3D information from
        self.n_chains = 0       # len(self.update)
        self.retry = []         # list of Chain() objects which we failed to extract information from
        self.loaded_chains = [] # list of Chain() objects we successfully extracted information from
        self.fam_list = []      # Rfam families of the above chains

        # Default options:
        self.CRYSTAL_RES = 4.0
        self.KEEP_HETATM = False
        self.HOMOLOGY = True
        self.USE_KNOWN_ISSUES = True
        self.RUN_STATS = False
        self.EXTRACT_CHAINS = False
        self.REUSE_ALL = False
        self.SELECT_ONLY = None
        self.ARCHIVE = False
        self.SAVELOGS = True
        self.FULLINFERENCE = False

    def process_options(self):
        """Sets the paths and options of the pipeline
        """
        
        global path_to_3D_data
        global path_to_seq_data

        setproctitle("RNANet.py process_options()")

        try:
            opts, _ = getopt.getopt(sys.argv[1:], "r:fhs", ["help", "resolution=", "3d-folder=", "seq-folder=", "keep-hetatm=",  "only=", "maxcores=",
                                                            "from-scratch", "full-inference", "no-homology", "ignore-issues", "extract", 
                                                            "all", "no-logs", "archive", "update-homologous", "version"])
        except getopt.GetoptError as err:
            print(err)
            sys.exit(2)

        for opt, arg in opts:

            if opt in ["--from-scratch", "--update-homologous"] and "tobedefinedbyoptions" in [path_to_3D_data, path_to_seq_data]:
                print("Please provide --3d-folder and --seq-folder first, so that we know what to delete and update.")
                exit()

            if opt == "-h" or opt == "--help":
                print("RNANet, a script to build a multiscale RNA dataset from public data\n"
                      "Developped by Louis Becquey (louis.becquey@univ-evry.fr), 2020")
                print()
                print("Options:")
                print("-h [ --help ]\t\t\tPrint this help message")
                print("--version\t\t\tPrint the program version")
                print()
                print("Select what to do:")
                print("--------------------------------------------------------------------------------------------------------------")
                print("-f [ --full-inference ]\t\tInfer new mappings even if Rfam already provides some. Yields more copies of"
                      "\n\t\t\t\t chains mapped to different families.")
                print("-s\t\t\t\tRun statistics computations after completion")
                print("--extract\t\t\tExtract the portions of 3D RNA chains to individual mmCIF files.")
                print("--keep-hetatm=False\t\t(True | False) Keep ions, waters and ligands in produced mmCIF files. "
                      "\n\t\t\t\t Does not affect the descriptors.")
                print("--no-homology\t\t\tDo not try to compute PSSMs and do not align sequences."
                      "\n\t\t\t\t Allows to yield more 3D data (consider chains without a Rfam mapping).")
                print()
                print("Select how to do it:")
                print("--------------------------------------------------------------------------------------------------------------")
                print("--3d-folder=…\t\t\tPath to a folder to store the 3D data files. Subfolders will contain:"
                      "\n\t\t\t\t\tRNAcifs/\t\tFull structures containing RNA, in mmCIF format"
                      "\n\t\t\t\t\trna_mapped_to_Rfam/\tExtracted 'pure' RNA chains"
                      "\n\t\t\t\t\tdatapoints/\t\tFinal results in CSV file format.")
                print("--seq-folder=…\t\t\tPath to a folder to store the sequence and alignment files. Subfolders will be:"
                      "\n\t\t\t\t\trfam_sequences/fasta/\tCompressed hits to Rfam families"
                      "\n\t\t\t\t\trealigned/\t\tSequences, covariance models, and alignments by family")
                print("--maxcores=…\t\t\tLimit the number of cores to use in parallel portions to reduce the simultaneous"
                      "\n\t\t\t\t need of RAM. Should be a number between 1 and your number of CPUs. Note that portions"
                      "\n\t\t\t\t of the pipeline already limit themselves to 50% or 70% of that number by default.")
                print("--archive\t\t\tCreate tar.gz archives of the datapoints text files and the alignments,"
                      "\n\t\t\t\t and update the link to the latest archive. ")
                print("--no-logs\t\t\tDo not save per-chain logs of the numbering modifications")
                print()
                print("Select which data we are interested in:")
                print("--------------------------------------------------------------------------------------------------------------")
                print("-r 4.0 [ --resolution=4.0 ]\tMaximum 3D structure resolution to consider a RNA chain.")
                print("--all\t\t\t\tBuild chains even if they already are in the database.")
                print("--only\t\t\t\tAsk to process a specific chain label only")
                print("--ignore-issues\t\t\tDo not ignore already known issues and attempt to compute them")
                print("--update-homologous\t\tRe-download Rfam and SILVA databases, realign all families, and recompute all CSV files")
                print("--from-scratch\t\t\tDelete database, local 3D and sequence files, and known issues, and recompute.")
                print()
                print("Typical usage:")
                print(f"nohup bash -c 'time {fileDir}/RNAnet.py --3d-folder ~/Data/RNA/3D/ --seq-folder ~/Data/RNA/sequences -s --no-logs' &")
                sys.exit()
            elif opt == '--version':
                print("RNANet v1.3 beta, parallelized, Dockerized")
                print("Last revision : Jan 2021")
                sys.exit()
            elif opt == "-r" or opt == "--resolution":
                assert float(arg) > 0.0 and float(arg) <= 20.0
                self.CRYSTAL_RES = float(arg)
            elif opt == "-s":
                self.RUN_STATS = True
            elif opt == "--keep-hetatm":
                assert arg in ["True", "False"]
                self.KEEP_HETATM = (arg == "True")
            elif opt == "--no-homology":
                self.HOMOLOGY = False
            elif opt == '--3d-folder':
                path_to_3D_data = os.path.abspath(arg)
                if path_to_3D_data[-1] != '/':
                    path_to_3D_data += '/'
                print("> Storing 3D data into", path_to_3D_data)
            elif opt == '--seq-folder':
                path_to_seq_data = os.path.abspath(arg)
                if path_to_seq_data[-1] != '/':
                    path_to_seq_data += '/'
                print("> Storing sequences into", path_to_seq_data)
            elif opt == "--ignore-issues":
                self.USE_KNOWN_ISSUES = False
            elif opt == "--only":
                self.USE_KNOWN_ISSUES = False
                self.REUSE_ALL = True
                self.SELECT_ONLY = arg
            elif opt == "--from-scratch":
                warn("Deleting previous database and recomputing from scratch.")
                subprocess.run(["rm", "-rf",
                                path_to_3D_data + "annotations",
                                path_to_3D_data + "RNAcifs",
                                path_to_3D_data + "rna_mapped_to_Rfam",
                                path_to_3D_data + "rnaonly",
                                path_to_seq_data + "realigned",
                                path_to_seq_data + "rfam_sequences",
                                runDir + "/known_issues.txt",
                                runDir + "/known_issues_reasons.txt",
                                runDir + "/results/RNANet.db"])
            elif opt == "--update-homologous":
                warn("Deleting previous sequence files and recomputing alignments.")
                subprocess.run(["rm", "-rf",
                                path_to_seq_data + "realigned",
                                path_to_seq_data + "rfam_sequences"])
                self.REUSE_ALL = True
            elif opt == "--all":
                self.REUSE_ALL = True
                self.USE_KNOWN_ISSUES = False
            elif opt == "--extract":
                self.EXTRACT_CHAINS = True
            elif opt == "--archive":
                self.ARCHIVE = True
            elif opt == "--no-logs":
                self.SAVELOGS = False
            elif opt == "--maxcores":
                global ncores
                ncores = min(ncores, int(arg))
            elif opt == "-f" or opt == "--full-inference":
                self.FULLINFERENCE = True

        if self.HOMOLOGY and "tobedefinedbyoptions" in [path_to_3D_data, path_to_seq_data] or path_to_3D_data == "tobedefinedbyoptions":
            print("usage: RNANet.py --3d-folder path/where/to/store/chains --seq-folder path/where/to/store/alignments")
            print("See RNANet.py --help for more information.")
            exit(1)

    @trace_unhandled_exceptions
    def list_available_mappings(self) -> None:
        """List 3D chains with available Rfam mappings.

        Return a list of Chain() objects with the mappings set up.        
        If self.HOMOLOGY is set to False, simply returns a list of Chain() objects with available 3D chains.
        """

        setproctitle("RNANet.py list_available_mappings()")

        # List all 3D RNA chains below given resolution
        full_structures_list = self.dl.download_BGSU_NR_list(self.CRYSTAL_RES)  # list of tuples ( class, class_members )

        # Check for a list of known problems:
        if os.path.isfile(runDir + "/known_issues.txt"):
            with open(runDir + "/known_issues.txt", 'r') as issues:
                if self.HOMOLOGY:
                    self.known_issues = [ x[:-1] for x in issues.readlines() if '-' in x ]
                else:
                    self.known_issues = [ x[:-1] for x in issues.readlines() if not '-' in x ]
            if self.USE_KNOWN_ISSUES:
                print("\t> Ignoring known issues:")
                print(" ".join(self.known_issues))

        if self.HOMOLOGY:
            # Ask Rfam their mappings between PDB structures and Rfam families
            allmappings = self.dl.download_Rfam_PDB_mappings()

            # Compute the extended list of mappable structures using NR-list and Rfam-PDB mappings
            # And get Chain() objects.
            # If self.FULLINFERENCE is False, the extended list is already filtered to remove
            # the chains which already are in the database.
            print("> Building list of structures...", flush=True)
            p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=ncores)
            try:

                pbar = tqdm(full_structures_list, maxinterval=1.0, miniters=1,
                            desc="Eq. classes", bar_format="{desc}:{percentage:3.0f}%|{bar}|")
                for _, newchains in enumerate(p.imap_unordered(partial(
                                                                    work_infer_mappings, 
                                                                    not self.REUSE_ALL, 
                                                                    allmappings, 
                                                                    self.FULLINFERENCE
                                                              ), 
                                                              full_structures_list, 
                                                              chunksize=1)):
                    self.update += newchains
                    
                    pbar.update(1)  # Everytime the iteration finishes, update the global progress bar

                pbar.close()
                p.close()
                p.join()
            except KeyboardInterrupt:
                warn("KeyboardInterrupt, terminating workers.", error=True)
                pbar.close()
                p.terminate()
                p.join()
                exit(1)
        else:
            conn = sqlite3.connect(runDir+"/results/RNANet.db", timeout=10.0)
            conn.execute('pragma journal_mode=wal')
            for eq_class, codelist in tqdm(full_structures_list, desc="Eq. classes"):
                codes = codelist.replace('+', ',').split(',')

                # Simply convert the list of codes to Chain() objects
                for c in codes:
                    nr = c.split('|')
                    pdb_id = nr[0].lower()
                    pdb_model = int(nr[1])
                    pdb_chain_id = nr[2].upper()
                    chain_label = f"{pdb_id}_{str(pdb_model)}_{pdb_chain_id}"
                    res = sql_ask_database(conn, f"""SELECT chain_id from chain 
                                                        WHERE structure_id='{pdb_id}' 
                                                        AND chain_name='{pdb_chain_id}' 
                                                        AND rfam_acc = 'unmappd' 
                                                        AND issue=0""")
                    if not len(res) or self.REUSE_ALL:  # the chain is NOT yet in the database, or this is a known issue
                        self.update.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label, eq_class))
            conn.close()

        if self.SELECT_ONLY is not None:
            self.update = [ c for c in self.update if c.chain_label == self.SELECT_ONLY ]

        self.n_chains = len(self.update)
        print(str(self.n_chains) + " RNA chains of interest.")

    @trace_unhandled_exceptions
    def dl_and_annotate(self, retry=False, coeff_ncores=0.75):
        """
        Gets mmCIF files from the PDB, and runs DSSR on them.
        Ignores a structure if the file already exists (not if we are retrying).

        REQUIRES the previous definition of self.update, so call list_available_mappings() before.
        SETS table structure
        """

        setproctitle(f"RNANet.py dl_and_annotate(retry={retry})")

        # Prepare the results folders
        if not os.path.isdir(path_to_3D_data + "RNAcifs"):
            # for the whole structures
            os.makedirs(path_to_3D_data + "RNAcifs")
        if not os.path.isdir(path_to_3D_data + "annotations"):
            # for DSSR analysis of the whole structures
            os.makedirs(path_to_3D_data + "annotations")

        # Download and annotate
        print("> Downloading and annotating structures (or checking previous results if they exist)...", flush=True)
        if retry:
            mmcif_list = sorted(set([c.pdb_id for c in self.retry]))
        else:
            mmcif_list = sorted(set([c.pdb_id for c in self.update]))
        try:
            p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=int(coeff_ncores*ncores))
            pbar = tqdm(mmcif_list, maxinterval=1.0, miniters=1, desc="mmCIF files")
            for _ in p.imap_unordered(work_mmcif, mmcif_list, chunksize=1):
                pbar.update(1)  # Everytime the iteration finishes, update the global progress bar
            pbar.close()
            p.close()
            p.join()
        except KeyboardInterrupt:
            warn("KeyboardInterrupt, terminating workers.", error=True)
            pbar.close()
            p.terminate()
            p.join()
            exit(1)

    def build_chains(self, retry=False, coeff_ncores=1.0):
        """ Extract the desired chain portions if asked,
        and extract their informations from the JSON files to the database.

        REQUIRES the previous definition of self.update, so call list_available_mappings() before.
        SETS self.loaded_chains
        """

        setproctitle(f"RNANet.py build_chains(retry={retry})")

        # Prepare folders
        if self.EXTRACT_CHAINS:
            if self.HOMOLOGY and not os.path.isdir(path_to_3D_data + "rna_mapped_to_Rfam"):
                # for the portions mapped to Rfam
                os.makedirs(path_to_3D_data + "rna_mapped_to_Rfam")
            if (not self.HOMOLOGY) and not os.path.isdir(path_to_3D_data + "rna_only"):
                # extract chains of pure RNA
                os.makedirs(path_to_3D_data + "rna_only")

        # define and run jobs
        joblist = []
        if retry:
            clist = self.retry
        else:
            clist = self.update
        for c in clist:
            if retry:
                c.delete_me = False  # give a second chance
            if (c.chain_label not in self.known_issues) or not self.USE_KNOWN_ISSUES:
                joblist.append(Job(function=work_build_chain, how_many_in_parallel=int(coeff_ncores*ncores),
                                   args=[c, self.EXTRACT_CHAINS, self.KEEP_HETATM, retry, self.SAVELOGS]))
        try:
            results = execute_joblist(joblist)
        except:
            print("Exiting", flush=True)
            exit(1)

        # If there were newly discovered problems, add this chain to the known issues
        issues = 0
        issues_names = []
        ki = open(runDir + "/known_issues.txt", 'a')
        kir = open(runDir + "/known_issues_reasons.txt", 'a')
        for c in results:
            if c[1].delete_me and c[1].chain_label not in self.known_issues:
                if retry or "Could not load existing" not in c[1].error_messages:
                    self.known_issues.append(c[1].chain_label)
                    issues += 1
                    issues_names.append(c[1].chain_label)
                    ki.write(c[1].chain_label + '\n')
                    kir.write(c[1].chain_label + '\n' +
                              c[1].error_messages + '\n\n')
                    with sqlite3.connect(runDir+"/results/RNANet.db") as conn:
                        conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
                        sql_execute(conn, f"UPDATE chain SET issue = 1 WHERE chain_id = ?;", data=(c[1].db_chain_id,))
        ki.close()
        kir.close()
        if issues:
            warn(f"Added {issues} newly discovered issues to known issues:")
            print("\033[33m" + " ".join(issues_names) + "\033[0m", flush=True)

        # Add successfully built chains to list
        self.loaded_chains += [c[1] for c in results if not c[1].delete_me]

        # Identify errors due to empty JSON files (this happen when RAM is full, we believe).
        # Retrying often solves the issue... so retry once with half the cores to limit the RAM usage.
        self.to_retry = [ c[1] for c in results if "Could not load existing" in c[1].error_messages ]

    def checkpoint_save_chains(self):
        """Saves self.loaded_chains to data/loaded_chains.picke
        """
        with open(runDir + "/data/loaded_chains.pickle", "wb") as pick:
            pickle.dump(self.loaded_chains, pick)

    def checkpoint_load_chains(self):
        """Load self.loaded_chains from data/loaded_chains.pickle
        """
        with open(runDir + "/data/loaded_chains.pickle", "rb") as pick:
            self.loaded_chains = pickle.load(pick)

    def prepare_sequences(self):
        """Downloads homologous sequences and covariance models required to compute MSAs.

        REQUIRES that self.loaded_chains is defined.
        SETS family (partially, through call)
        """

        setproctitle("RNANet.py prepare_sequences()")

        # Preparing a results folder
        if not os.access(path_to_seq_data + "realigned/", os.F_OK):
            os.makedirs(path_to_seq_data + "realigned/")
        if not os.path.isdir(path_to_seq_data + "rfam_sequences/fasta/"):
            os.makedirs(path_to_seq_data + "rfam_sequences/fasta/", exist_ok=True)

        # Update the family table (rfam_acc, description, max_len)
        self.dl.download_Rfam_family_stats(self.fam_list)

        # Download the covariance models for all families
        self.dl.download_Rfam_cm()

        joblist = []
        for f in self.fam_list:
            joblist.append(Job(function=work_prepare_sequences, how_many_in_parallel=ncores, args=[
                           self.dl, f, rfam_acc_to_download[f]]))
        try:
            execute_joblist(joblist)

            if len(set(self.fam_list).intersection(SSU_set)):
                self.dl.download_from_SILVA("SSU")
            if len(set(self.fam_list).intersection(LSU_set)):
                self.dl.download_from_SILVA("LSU")
        except KeyboardInterrupt:
            print("Exiting")
            exit(1)

    def realign(self):
        """Perform multiple sequence alignments.

        REQUIRES self.fam_list to be defined
        SETS family (partially)
        """

        setproctitle("RNANet.py realign()")

        # Prepare the job list
        joblist = []
        for f in self.fam_list:
            # the function already uses all CPUs so launch them one by one (how_many_in_parallel=1)
            joblist.append(Job(function=work_realign, args=[f], how_many_in_parallel=1, label=f))

        # Execute the jobs
        try:
            results = execute_joblist(joblist)
        except:
            print("Exiting")
            exit(1)

        # Update the database
        data = []
        for r in results:
            align = AlignIO.read(path_to_seq_data + "realigned/" + r[0] + "++.afa", "fasta")
            nb_3d_chains = len([1 for r in align if '[' in r.id])
            if r[0] in SSU_set:  # SSU v138 is used
                nb_homologs = 2225272       # source: https://www.arb-silva.de/documentation/release-138/
                nb_total_homol = nb_homologs + nb_3d_chains
            elif r[0] in LSU_set:  # LSU v132 is used
                nb_homologs = 198843        # source: https://www.arb-silva.de/documentation/release-132/
                nb_total_homol = nb_homologs + nb_3d_chains
            else:
                nb_total_homol = len(align)
                nb_homologs = nb_total_homol - nb_3d_chains
            data.append((nb_homologs, nb_3d_chains, nb_total_homol, align.get_alignment_length(), r[2], r[3], r[0]))

        with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
            sql_execute(conn, """UPDATE family SET nb_homologs = ?, nb_3d_chains = ?, nb_total_homol = ?, ali_len = ?, comput_time = ?, comput_peak_mem = ? 
                                 WHERE rfam_acc = ?;""", many=True, data=data)

    def remap(self):
        """Compute nucleotide frequencies of some alignments and save them in the database

        REQUIRES self.fam_list to be defined
        """

        setproctitle("RNANet.py remap()")

        print("Computing nucleotide frequencies in alignments...\nThis can be very long on slow storage devices (Hard-drive...)")
        print("Check your CPU and disk I/O activity before deciding if the job failed.")
        nworkers = max(min(ncores, len(self.fam_list)), 1)

        # Prepare the architecture of a shiny multi-progress-bars design
                                                # Push the number of workers to a queue.
        global idxQueue                         # ... Then each Pool worker will
        for i in range(nworkers):               # ... pick a number from the queue when starting the computation for one family,
            idxQueue.put(i)                     # ... and replace it when the computation has ended so it could be picked up later.

        # Start a process pool to dispatch the RNA families,
        # over multiple CPUs (one family by CPU)
        p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=nworkers)

        try:
            fam_pbar = tqdm(total=len(self.fam_list), desc="RNA families", position=0, leave=True)
            # Apply work_pssm_remap to each RNA family
            for i, _ in enumerate(p.imap_unordered(work_pssm_remap, self.fam_list, chunksize=1)):
                # Everytime the iteration finishes on a family, update the global progress bar over the RNA families
                fam_pbar.update(1)
            fam_pbar.close()
            p.close()
            p.join()
        except KeyboardInterrupt:
            warn("KeyboardInterrupt, terminating workers.", error=True)
            fam_pbar.close()
            p.terminate()
            p.join()
            exit(1)

    def output_results(self):
        """Produces CSV files, archive them, and additional metadata files

        REQUIRES self.loaded_chains (to output corresponding CSV files) and self.fam_list (for statistics)
        """

        setproctitle("RNANet.py output_results()")

        # Prepare folders:
        if not os.path.isdir(path_to_3D_data + "datapoints/"):
            os.makedirs(path_to_3D_data + "datapoints/")

        # Save to by-chain CSV files
        p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=3)
        try:
            pbar = tqdm(total=len(self.loaded_chains), desc="Saving chains to CSV", position=0, leave=True)
            for _, _2 in enumerate(p.imap_unordered(work_save, self.loaded_chains)):
                pbar.update(1)
            pbar.close()
            p.close()
            p.join()
        except KeyboardInterrupt:
            warn("KeyboardInterrupt, terminating workers.", error=True)
            pbar.close()
            p.terminate()
            p.join()
            exit(1)

        # Run statistics
        if self.RUN_STATS:
            # Remove previous precomputed data
            subprocess.run(["rm", "-f", runDir + f"/data/wadley_kernel_eta_{self.CRYSTAL_RES}.npz", 
                                        runDir + f"/data/wadley_kernel_eta_prime_{self.CRYSTAL_RES}.npz", 
                                        runDir + f"/data/pair_counts_{self.CRYSTAL_RES}.csv"])
            for f in self.fam_list:
                subprocess.run(["rm", "-f", runDir + f"/data/{f}.npy", 
                                            runDir + f"/data/{f}_pairs.csv", 
                                            runDir + f"/data/{f}_counts.csv"])

            # Run statistics files
            subprocess.run([python_executable, fileDir+"/scripts/regression.py", runDir + "/results/RNANet.db"])
            subprocess.run([python_executable, fileDir+"/statistics.py", "--3d-folder",  path_to_3D_data, 
                            "--seq-folder", path_to_seq_data, "-r", str(self.CRYSTAL_RES)])

        # Save additional informations
        with sqlite3.connect(runDir+"/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal')
            pd.read_sql_query("""SELECT rfam_acc, description, idty_percent, nb_homologs, nb_3d_chains, nb_total_homol, max_len, comput_time, comput_peak_mem 
                                 FROM family ORDER BY nb_3d_chains DESC;""",
                              conn).to_csv(runDir + f"/results/families.csv", float_format="%.2f", index=False)
            pd.read_sql_query("""SELECT eq_class, structure_id, chain_name, pdb_start, pdb_end, rfam_acc, inferred, date, exp_method, resolution, issue 
                                 FROM structure 
                                 JOIN chain ON structure.pdb_id = chain.structure_id
                                 ORDER BY structure_id, chain_name, rfam_acc ASC;""",
                              conn).to_csv(runDir + f"/results/summary.csv", float_format="%.2f", index=False)

        if self.ARCHIVE:
            os.makedirs(runDir + "/archive", exist_ok=True)
            datestr = time.strftime('%Y%m%d')
            subprocess.run(["rm", "-f", runDir + f"/archive/RNANET_datapoints_latest.tar.gz"])
            subprocess.run(["tar", "-C", path_to_3D_data + "/datapoints", "-czf", runDir + f"/archive/RNANET_datapoints_{datestr}.tar.gz", "."])
            subprocess.run(["ln", "-s", runDir + f"/archive/RNANET_datapoints_{datestr}.tar.gz", runDir + f"/archive/RNANET_datapoints_latest.tar.gz"])

            # gather the alignments
            os.makedirs(path_to_seq_data + "realigned/3d_only", exist_ok=True)
            subprocess.run(["cp", path_to_seq_data + "realigned/*_3d_only.afa", path_to_seq_data + "realigned/3d_only" ])
            subprocess.run(["rm", "-f", runDir + f"/archive/RNANET_alignments_latest.tar.gz"])
            subprocess.run(["tar", "-C", path_to_seq_data + "realigned/3d_only" , "-czf", runDir + f"/archive/RNANET_alignments_latest.tar.gz", "."])

    def sanitize_database(self):
        """Searches for issues in the database and correct them"""

        setproctitle("RNANet.py sanitize_database()")

        conn = sqlite3.connect(runDir + "/results/RNANet.db")
        conn.execute('pragma journal_mode=wal')

        # Assert every structure is used
        r = sql_ask_database(conn, """SELECT DISTINCT pdb_id FROM structure WHERE pdb_id NOT IN (SELECT DISTINCT structure_id FROM chain);""")
        if len(r) and r[0][0] is not None:
            warn("Structures without referenced chains have been detected.")
            print(" ".join([x[0] for x in r]))

        # Assert every chain is attached to a structure
        r = sql_ask_database(conn, """SELECT DISTINCT chain_id, structure_id FROM chain WHERE structure_id NOT IN (SELECT DISTINCT pdb_id FROM structure);""")
        if len(r) and r[0][0] is not None:
            warn("Chains without referenced structures have been detected")
            print(" ".join([str(x[1])+'-'+str(x[0]) for x in r]))

        if self.HOMOLOGY:
            # check if chains have been re_mapped:
            r = sql_ask_database(conn, """SELECT COUNT(DISTINCT chain_id) AS Count, rfam_acc FROM chain 
                                          WHERE issue = 0 
                                                AND rfam_acc != 'unmappd'
                                                AND chain_id NOT IN (SELECT DISTINCT chain_id FROM re_mapping)
                                          GROUP BY rfam_acc;""")
            try:
                if len(r) and r[0][0] is not None:
                    warn("Chains were not remapped:")
                    for x in r:
                        print(str(x[0]) + " chains of family " + x[1])
            except TypeError as e:
                print(r)
                print(next(r))
                print(e)
                exit()
            # # TODO : Optimize this (too slow)
            # # check if some columns are missing in the remappings:
            # r = sql_ask_database(conn, """SELECT c.chain_id, c.structure_id, c.chain_name, c.rfam_acc, r.index_chain, r.index_ali
            #                                 FROM chain as c
            #                                 NATURAL JOIN re_mapping as r
            #                                 WHERE index_ali NOT IN (SELECT index_ali FROM align_column WHERE rfam_acc = c.rfam_acc);""")
            # if len(r) and r[0][0] is not None:
            #     warn("Missing positions in the re-mapping:")
            #     for x in r:
            #         print(x)

            # check that filtered alignment have the same length than the number of saved alignment columns for a family
            r = sql_ask_database(conn, """select family.rfam_acc, count, ali_filtered_len 
                                          FROM family 
                                          LEFT JOIN (
                                              SELECT rfam_acc, count(distinct index_ali) as count from align_column where index_ali>0 group by rfam_acc
                                          ) AS s ON family.rfam_acc=s.rfam_acc;""")
            for f in r:
                if f[1] is None or f[2] is None:
                    warn(f"{f[0]} has incomplete alignement data: {f[1]} alignement columns saved, filtered alignment is of length {f[2]}")
                    continue

                if f[1] != f[2]:
                    warn(f"{f[0]} has {f[1]} alignement columns saved, but its filtered alignment is of length {f[2]} !")

        conn.close()


def read_cpu_number():
    """This function reads the number of CPU cores available from /proc/cpuinfo.
    One shall not use os.cpu_count() on LXC containers,
    because it reads info from /sys wich is not the VM resources but the host resources.
    """
    p = subprocess.run(['grep', '-Ec', '(Intel|AMD)', '/proc/cpuinfo'], stdout=subprocess.PIPE)
    return int(int(p.stdout.decode('utf-8')[:-1])/2)

def init_worker(tqdm_lock=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)

def warn(message, error=False):
    """Pretty-print warnings and error messages.
    """
    # Cut if too long
    if len(message) > 66:
        x = message.find(' ', 50, 66)
        if x != -1:
            warn(message[:x], error=error)
            warn(message[x+1:], error=error)
        else:
            warn(message[:x], error=error)
        return

    if error:
        print(f"\t> \033[31mERR: {message:65s}\033[0m\t{errsymb}", flush=True)
    else:
        print(f"\t> \033[33mWARN: {message:64s}\033[0m\t{warnsymb}", flush=True)

def notify(message, post=''):
    if len(post):
        post = '(' + post + ')'
    print(f"\t> {message:70s}\t{validsymb}\t{post}", flush=True)

def _mutable_SeqIO_to_alignment_iterator(handle):
    records = list(MutableFastaIterator(handle))
    if records:
        yield MultipleSeqAlignment(records)

def parse(handle):
    with open(handle, 'r') as fp:
        yield from _mutable_SeqIO_to_alignment_iterator(fp)

def read(handle):
    iterator = parse(handle)
    try:
        alignment = next(iterator)
    except StopIteration:
        raise ValueError("No records found in handle") from None
    try:
        next(iterator)
        raise ValueError("More than one record found in handle")
    except StopIteration:
        pass
    return alignment

def sql_define_tables(conn):
    conn.executescript(
        """ PRAGMA foreign_keys = on;
            CREATE TABLE IF NOT EXISTS structure (
                pdb_id         CHAR(4) PRIMARY KEY NOT NULL,
                pdb_model      CHAR(1) NOT NULL,
                date           DATE,
                exp_method     VARCHAR(50),
                resolution     REAL,
                UNIQUE (pdb_id, pdb_model)
            );
            CREATE TABLE IF NOT EXISTS chain (
                chain_id        INTEGER PRIMARY KEY NOT NULL,
                structure_id    CHAR(4) NOT NULL,
                chain_name      VARCHAR(2) NOT NULL,
                eq_class        VARCHAR(16),
                pdb_start       SMALLINT,
                pdb_end         SMALLINT,
                issue           TINYINT,
                rfam_acc        CHAR(7),
                inferred        TINYINT,
                chain_freq_A    REAL,
                chain_freq_C    REAL,
                chain_freq_G    REAL,
                chain_freq_U    REAL,
                chain_freq_other REAL,
                pair_count_cWW  SMALLINT,
                pair_count_cWH  SMALLINT,
                pair_count_cWS  SMALLINT,
                pair_count_cHH  SMALLINT,
                pair_count_cHS  SMALLINT,
                pair_count_cSS  SMALLINT,
                pair_count_tWW  SMALLINT,
                pair_count_tWH  SMALLINT,
                pair_count_tWS  SMALLINT,
                pair_count_tHH  SMALLINT,
                pair_count_tHS  SMALLINT,
                pair_count_tSS  SMALLINT,
                pair_count_other SMALLINT,
                UNIQUE (structure_id, chain_name, rfam_acc),
                FOREIGN KEY(structure_id) REFERENCES structure(pdb_id),
                FOREIGN KEY(rfam_acc) REFERENCES family(rfam_acc)
            );
            CREATE TABLE IF NOT EXISTS nucleotide (
                chain_id        INT,
                index_chain     SMALLINT,
                old_nt_resnum   VARCHAR(5),
                nt_position     SMALLINT,
                nt_name         VARCHAR(5),
                nt_code         CHAR(1),
                nt_align_code   CHAR(1),
                is_A TINYINT, is_C TINYINT, is_G TINYINT, is_U TINYINT, is_other TINYINT,
                dbn             CHAR(1),
                paired          VARCHAR(20),
                nb_interact     TINYINT,
                pair_type_LW    VARCHAR(20),
                pair_type_DSSR  VARCHAR(25),
                alpha REAL, beta REAL, gamma REAL, delta REAL, epsilon REAL, zeta REAL,
                epsilon_zeta    REAL,
                bb_type         VARCHAR(5),
                chi             REAL,
                glyco_bond      VARCHAR(3),
                v0 REAL, v1 REAL, v2 REAL, v3 REAL, v4 REAL,
                form            CHAR(1),
                ssZp            REAL,
                Dp              REAL,
                eta REAL, theta REAL, eta_prime REAL, theta_prime REAL, eta_base REAL, theta_base REAL,
                phase_angle     REAL,
                amplitude       REAL,
                puckering       VARCHAR(20),
                PRIMARY KEY (chain_id, index_chain),
                FOREIGN KEY(chain_id) REFERENCES chain(chain_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS re_mapping (
                chain_id        INT NOT NULL,
                index_chain     INT NOT NULL,
                index_ali       INT NOT NULL,
                PRIMARY KEY (chain_id, index_chain),
                FOREIGN KEY(chain_id) REFERENCES chain(chain_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS family (
                rfam_acc        CHAR(7) PRIMARY KEY NOT NULL,
                description     VARCHAR(100),
                nb_homologs     INT,
                nb_3d_chains    INT,
                nb_total_homol  INT,
                max_len         UNSIGNED SMALLINT,
                ali_len         UNSIGNED SMALLINT,
                ali_filtered_len UNSIGNED SMALLINT,
                comput_time     REAL,
                comput_peak_mem REAL,
                idty_percent    REAL
            );
            CREATE TABLE IF NOT EXISTS align_column (
                rfam_acc        CHAR(7) NOT NULL,
                index_ali       INT NOT NULL,
                freq_A          REAL,
                freq_C          REAL,
                freq_G          REAL,
                freq_U          REAL,
                freq_other      REAL,
                gap_percent     REAL,
                consensus       CHAR(1),
                PRIMARY KEY (rfam_acc, index_ali),
                FOREIGN KEY(rfam_acc) REFERENCES family(rfam_acc)
            );
         """)
    conn.commit()

@trace_unhandled_exceptions
def sql_ask_database(conn, sql, warn_every=10):
    """
    Reads the SQLite database.
    Returns a list of tuples.
    """
    cursor = conn.cursor()
    for _ in range(100):  # retry 100 times if it fails
        try:
            result = cursor.execute(sql).fetchall()
            cursor.close()
            return result         # if it worked, no need to retry
        except sqlite3.OperationalError as e:
            if warn_every and not (_+1) % warn_every:
                warn(str(e) + ", retrying in 0.2s (worker " +
                     str(os.getpid()) + f', try {_+1}/100)')
            time.sleep(0.2)
    cursor.close()
    warn("Tried to reach database 100 times and failed. Aborting.", error=True)
    return []

@trace_unhandled_exceptions
def sql_execute(conn, sql, many=False, data=None, warn_every=10):
    for _ in range(100):  # retry 100 times if it fails
        try:
            if many:
                conn.executemany(sql, data)
            else:
                cur = conn.cursor()
                if data is None:
                    cur.execute(sql)
                else:
                    cur.execute(sql, data)
                cur.close()
            conn.commit()   # Apply modifications
            return          # if it worked, no need to retry
        except sqlite3.OperationalError as e:
            if warn_every and not (_+1) % warn_every:
                warn(str(e) + ", retrying in 0.2s (worker " +
                     str(os.getpid()) + f', try {_+1}/100)')
            time.sleep(0.2)
    warn("Tried to reach database 100 times and failed. Aborting.", error=True)

@trace_unhandled_exceptions
def execute_job(j, jobcount):
    """Run a Job object.
    """
    # increase the counter of running jobs
    running_stats[0] += 1

    # Monitor this process
    m = -1
    monitor = Monitor(os.getpid())

    if len(j.cmd_):  # The job is a system command

        print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.label}")

        # Add the command to logfile
        os.makedirs(runDir+"/logs", exist_ok=True)
        logfile = open(runDir + "/logs/log_of_the_run.sh", 'a')
        logfile.write(" ".join(j.cmd_))
        logfile.write("\n")
        logfile.close()

        # Run it
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # put the monitor in a different thread
            assistant_future = executor.submit(monitor.check_mem_usage)

            # run the command. subprocess.run will be a child of this process, and stays monitored.
            start_time = time.time()
            r = subprocess.run(j.cmd_, timeout=j.timeout_,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end_time = time.time()

            # Stop the Monitor, then get its result
            monitor.keep_watching = False
            m = assistant_future.result()

    elif j.func_ is not None:

        print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.func_.__name__}({', '.join([str(a) for a in j.args_ if type(a) != list])})", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # put the monitor in a different thread
            assistant_future = executor.submit(monitor.check_mem_usage)

            # call the python function (in this process)
            start_time = time.time()
            r = j.func_(* j.args_)
            end_time = time.time()

            # Stop the Monitor, then get its result
            monitor.keep_watching = False
            m = assistant_future.result()

    # increase the counter of finished jobs
    running_stats[1] += 1

    # return time and memory statistics, plus the job results
    t = end_time - start_time
    return (t, m, r)

def execute_joblist(fulljoblist):
    """ Run a list of job objects.

    The jobs in the list can have differente priorities and/or different number of threads.

    Returns a tuple (label, actual_result, comp_time, peak_mem)
    """

    # Reset counters
    running_stats[0] = 0       # started
    running_stats[1] = 0       # finished
    running_stats[2] = 0       # failed

    # Sort jobs in a tree structure, first by priority, then by CPU numbers
    jobs = {}
    jobcount = len(fulljoblist)
    if not jobcount:
        warn("nothing to do !")
        return []
    for job in fulljoblist:
        if job.priority_ not in jobs.keys():
            jobs[job.priority_] = {}
        if job.nthreads not in jobs[job.priority_].keys():
            jobs[job.priority_][job.nthreads] = []
        jobs[job.priority_][job.nthreads].append(job)

    # number of different priorities in the list
    nprio = max(jobs.keys())

    # Process the jobs from priority 1 to nprio
    results = []
    for i in range(1, nprio+1):
        if i not in jobs.keys():
            continue  # no job has the priority level i

        print("processing jobs of priority", i)
        different_thread_numbers = sorted(jobs[i].keys())

        # jobs should be processed 1 by 1, 2 by 2, or n by n depending on their definition
        for n in different_thread_numbers:
            # get the bunch of jobs of same priority and thread number
            bunch = jobs[i][n]
            if not len(bunch):
                continue  # no jobs should be processed n by n

            print("using", n, "processes:")
            # execute jobs of priority i that should be processed n by n:
            p = Pool(processes=n, maxtasksperchild=1, initializer=init_worker)
            try:
                raw_results = p.map(partial(execute_job, jobcount=jobcount), bunch, chunksize=2)
                p.close()
                p.join()
            except KeyboardInterrupt as e:
                warn("KeyboardInterrupt, terminating workers.", error=True)
                p.terminate()
                p.join()
                raise e

            for j, r in zip(bunch, raw_results):
                j.comp_time = round(r[0], 2)  # seconds
                j.max_mem = int(r[1]/1000000)  # MB
                results.append((j.label, r[2], round(r[0], 2), int(r[1]/1000000)))

    # throw back the money
    return results

@trace_unhandled_exceptions
def work_infer_mappings(update_only, allmappings, fullinference, codelist) -> list:
    """Given a list of PDB chains corresponding to an equivalence class from BGSU's NR list, 
    build a list of Chain() objects mapped to Rfam families, by expanding available mappings 
    of any element of the list to all the list elements.
    """

    setproctitle("RNAnet.py work_infer_mappings()")

    newchains = []
    known_mappings = pd.DataFrame()

    # Split the comma-separated list of chain codes into chain codes:
    eq_class = codelist[0]
    codes = codelist[1].replace('+', ',').split(',')

    # Search for mappings that apply to an element of this PDB chains list:
    for c in codes:
        # search for Rfam mappings with this chain c:
        m_row_indices = allmappings.pdb_id + "|1|" + allmappings.chain == c[:4].lower()+c[4:]
        m = allmappings.loc[m_row_indices].drop(['bit_score', 'evalue_score', 'cm_start', 'cm_end', 'hex_colour'], axis=1)
        if len(m):
            # remove the found mappings from the dataframe
            allmappings = allmappings.loc[m_row_indices == False]
            # Add the found mappings to the list of found mappings for this class of equivalence
            known_mappings = pd.concat([known_mappings, m])

    # Now infer mappings for chains that are not explicitely listed in Rfam-PDB mappings:
    if len(known_mappings):

        families = set(known_mappings['rfam_acc'])

        # generalize
        inferred_mappings = known_mappings.drop(['pdb_id', 'chain'], axis=1).drop_duplicates()

        # check for approximative redundancy:
        if len(inferred_mappings) != len(inferred_mappings.drop_duplicates(subset="rfam_acc")):
            # Then, there exists some mapping variants onto the same Rfam family CM,
            # but varing in the start/end positions in the chain.
            # ==> Summarize them in one mapping but with the largest window.
            for rfam in families:
                sel_5_to_3 = (inferred_mappings['pdb_start'] < inferred_mappings['pdb_end'])
                thisfam_5_3 = (inferred_mappings['rfam_acc'] == rfam) & sel_5_to_3
                thisfam_3_5 = (inferred_mappings['rfam_acc'] == rfam) & (sel_5_to_3 == False)

                if (
                        len(inferred_mappings[thisfam_5_3]) != len(inferred_mappings[inferred_mappings['rfam_acc'] == rfam])
                    and len(inferred_mappings[thisfam_5_3]) > 0
                ):
                    # there are mappings in both directions... wtf Rfam ?!
                    if (len(inferred_mappings[thisfam_5_3]) == len(inferred_mappings[thisfam_3_5]) == 1
                                and int(inferred_mappings[thisfam_5_3].pdb_start) == int(inferred_mappings[thisfam_3_5].pdb_end)
                                and int(inferred_mappings[thisfam_5_3].pdb_end) == int(inferred_mappings[thisfam_3_5].pdb_start)
                            ):
                        # The two mappings are on the same nucleotide interval, but in each sense.
                        # e.g. RF00254 6v5b and 6v5c... maybe a bug on their side ?
                        # How can a chain match a CM in both senses ?
                        # We keep only the 5->3 sense.
                        inferred_mappings = inferred_mappings.drop(index=inferred_mappings.index[thisfam_3_5])
                        sel_5_to_3 = (inferred_mappings['pdb_start'] < inferred_mappings['pdb_end'])
                        thisfam_5_3 = (inferred_mappings['rfam_acc'] == rfam) & sel_5_to_3
                        thisfam_3_5 = (inferred_mappings['rfam_acc'] == rfam) & (sel_5_to_3 == False)
                        print()
                        warn(f"Found mappings to {rfam} in both directions on the same interval, keeping only the 5'->3' one.")
                    else:
                        warn(f"There are mappings for {rfam} in both directions:", error=True)
                        print(inferred_mappings)
                        exit(1)

                # Compute consensus for chains in 5' -> 3' sense
                if len(inferred_mappings[thisfam_5_3]):
                    pdb_start_min = min(inferred_mappings[thisfam_5_3]['pdb_start'])
                    pdb_end_max = max(inferred_mappings[thisfam_5_3]['pdb_end'])
                    pdb_start_max = max(inferred_mappings[thisfam_5_3]['pdb_start'])
                    pdb_end_min = min(inferred_mappings[thisfam_5_3]['pdb_end'])
                    if (pdb_start_max - pdb_start_min < 100) and (pdb_end_max - pdb_end_min < 100):
                        # the variation is only a few nucleotides, we take the largest window.
                        inferred_mappings.loc[thisfam_5_3, 'pdb_start'] = pdb_start_min
                        inferred_mappings.loc[thisfam_5_3, 'pdb_end'] = pdb_end_max
                    else:
                        # there probably is an outlier. We chose the median value in the whole list of known_mappings.
                        known_sel_5_to_3 = (known_mappings['rfam_acc'] == rfam) & (known_mappings['pdb_start'] < known_mappings['pdb_end'])
                        inferred_mappings.loc[thisfam_5_3, 'pdb_start'] = known_mappings.loc[known_sel_5_to_3, 'pdb_start'].median()
                        inferred_mappings.loc[thisfam_5_3, 'pdb_end'] = known_mappings.loc[known_sel_5_to_3, 'pdb_end'].median()

                #  Compute consensus for chains in 3' -> 5' sense
                if len(inferred_mappings[thisfam_3_5]):
                    pdb_start_min = min(inferred_mappings[thisfam_3_5]['pdb_start'])
                    pdb_end_max = max(inferred_mappings[thisfam_3_5]['pdb_end'])
                    pdb_start_max = max(inferred_mappings[thisfam_3_5]['pdb_start'])
                    pdb_end_min = min(inferred_mappings[thisfam_3_5]['pdb_end'])
                    if (pdb_start_max - pdb_start_min < 100) and (pdb_end_max - pdb_end_min < 100):
                        # the variation is only a few nucleotides, we take the largest window.
                        inferred_mappings.loc[thisfam_3_5, 'pdb_start'] = pdb_start_max
                        inferred_mappings.loc[thisfam_3_5, 'pdb_end'] = pdb_end_min
                    else:
                        # there probably is an outlier. We chose the median value in the whole list of known_mappings.
                        known_sel_3_to_5 = (known_mappings['rfam_acc'] == rfam) & (known_mappings['pdb_start'] > known_mappings['pdb_end'])
                        inferred_mappings.loc[thisfam_3_5, 'pdb_start'] = known_mappings.loc[known_sel_3_to_5, 'pdb_start'].median()
                        inferred_mappings.loc[thisfam_3_5, 'pdb_end'] = known_mappings.loc[known_sel_3_to_5, 'pdb_end'].median()
            inferred_mappings.drop_duplicates(inplace=True)

        # Now build Chain() objects for the mapped chains
        for c in codes:
            nr = c.split('|')
            pdb_id = nr[0].lower()
            pdb_model = int(nr[1])
            pdb_chain_id = nr[2]
            for rfam in families:
                # if a known mapping of this chain on this family exists, apply it
                this_chain_idxs = (known_mappings.pdb_id + "|1|" + known_mappings.chain == c[:4].lower()+c[4:])
                m = known_mappings.loc[this_chain_idxs & (known_mappings['rfam_acc'] == rfam)]
                if len(m) and len(m) < 2:
                    pdb_start = int(m.pdb_start)
                    pdb_end = int(m.pdb_end)
                    inferred = False
                elif len(m):
                    # two different parts of the same chain are mapped to the same family... (ex: 6ek0-L5)
                    # ==> map the whole chain to that family, not the parts
                    pdb_start = int(m.pdb_start.min())
                    pdb_end = int(m.pdb_end.max())
                    inferred = False
                elif (fullinference or not(this_chain_idxs.any())): 
                    # if no known mapping on another family, use the inferred mapping
                    # idem if the user said to do so with --full-inference
                    pdb_start = int(inferred_mappings.loc[(inferred_mappings['rfam_acc'] == rfam)].pdb_start)
                    pdb_end = int(inferred_mappings.loc[(inferred_mappings['rfam_acc'] == rfam)].pdb_end)
                    inferred = True
                else:
                    # skip this family, we cannot map this chain to it.
                    continue
                chain_label = f"{pdb_id}_{str(pdb_model)}_{pdb_chain_id}_{pdb_start}-{pdb_end}"

                # Check if the chain exists in the database
                if update_only:
                    with sqlite3.connect(runDir+"/results/RNANet.db", timeout=10.0) as conn:
                        conn.execute('pragma journal_mode=wal')
                        res = sql_ask_database(conn, f"""SELECT chain_id from chain 
                                                         WHERE structure_id='{pdb_id}' 
                                                         AND chain_name='{pdb_chain_id}' 
                                                         AND rfam_acc='{rfam}' 
                                                         AND issue=0""")
                    if not len(res):  # the chain is NOT yet in the database, or this is a known issue
                        newchains.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label, eq_class,
                                               rfam=rfam, inferred=inferred, pdb_start=pdb_start, pdb_end=pdb_end))
                else:
                    newchains.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label, eq_class,
                                           rfam=rfam, inferred=inferred, pdb_start=pdb_start, pdb_end=pdb_end))

    return newchains

@trace_unhandled_exceptions
def work_mmcif(pdb_id):
    """ Look for a CIF file (with all chains) from RCSB

    SETS table structure
    """

    setproctitle(f"RNAnet.py work_mmcif({pdb_id})")

    final_filepath = path_to_3D_data+"RNAcifs/"+pdb_id+".cif"

    # Attempt to download it if not present
    try:
        if not os.path.isfile(final_filepath):
            subprocess.run(
                ["wget", f'http://files.rcsb.org/download/{pdb_id}.cif', "-O", final_filepath],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except:
        warn(f"Unable to download {pdb_id}.cif. Ignoring it.", error=True)
        return

    # check if it exists in database
    with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
        conn.execute('pragma journal_mode=wal')
        r = sql_ask_database(conn, f"""SELECT * from structure where pdb_id = '{pdb_id}';""")

    # if not, read the CIF header and register the structure
    if not len(r):
        # Load the MMCIF file with Biopython
        mmCif_info = pdb.MMCIF2Dict.MMCIF2Dict(final_filepath)

        # Get info about that structure
        try:
            exp_meth = mmCif_info["_exptl.method"][0]
        except KeyError:
            warn(f"Wtf, {pdb_id}.cif has no _exptl.method ? Assuming X-ray.")
            warn(f"Check https://files.rcsb.org/header/{pdb_id}.cif to figure it out.")
            exp_meth = "X-RAY DIFFRACTION"
        date = mmCif_info["_pdbx_database_status.recvd_initial_deposition_date"][0]
        if "_refine.ls_d_res_high" in mmCif_info.keys() and mmCif_info["_refine.ls_d_res_high"][0] not in ['.', '?']:
            reso = float(mmCif_info["_refine.ls_d_res_high"][0])
        elif "_refine.ls_d_res_low" in mmCif_info.keys() and mmCif_info["_refine.ls_d_res_low"][0] not in ['.', '?']:
            reso = float(mmCif_info["_refine.ls_d_res_low"][0])
        elif "_em_3d_reconstruction.resolution" in mmCif_info.keys() and mmCif_info["_em_3d_reconstruction.resolution"][0] not in ['.', '?']:
            reso = float(mmCif_info["_em_3d_reconstruction.resolution"][0])
        else:
            warn(f"Wtf, structure {pdb_id} has no resolution ?")
            warn(f"Check https://files.rcsb.org/header/{pdb_id}.cif to figure it out.")
            reso = 0.0

        # Save into the database
        with sqlite3.connect(runDir + "/results/RNANet.db") as conn:
            conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
            sql_execute(conn, """INSERT OR REPLACE INTO structure (pdb_id, pdb_model, date, exp_method, resolution)
                                VALUES (?, ?, DATE(?), ?, ?);""", data=(pdb_id, 1, date, exp_meth, reso))

    if not os.path.isfile(path_to_3D_data + "annotations/" + pdb_id + ".json"):

        # run DSSR (you need to have it in your $PATH, follow x3dna installation instructions)
        output = subprocess.run(["x3dna-dssr", f"-i={final_filepath}", "--json", "--auxfile=no"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = output.stdout.decode('utf-8')
        stderr = output.stderr.decode('utf-8')

        if "exception" in stderr:
            # DSSR is unable to parse the chain.
            warn(f"Exception while running DSSR, ignoring {pdb_id}.", error=True)
            return 1

        # save the analysis to file only if we can load it :/
        json_file = open(path_to_3D_data + "annotations/" +
                         pdb_id + ".json", "w")
        json_file.write(stdout)
        json_file.close()

    return 0

@trace_unhandled_exceptions
def work_build_chain(c, extract, khetatm, retrying=False, save_logs=True):
    """Reads information from JSON and save it to database.
    If asked, also extracts the 3D chains from their original structure files.
    """

    setproctitle(f"RNAnet.py work_build_chain({c.chain_label})")

    if not os.path.isfile(path_to_3D_data + "annotations/" + c.pdb_id + ".json"):
        warn(f"Could not find annotations for {c.chain_label}, ignoring it.", error=True)
        c.delete_me = True
        c.error_messages += f"Could not download and/or find annotations for {c.chain_label}."

    # extract the 3D descriptors
    if not c.delete_me:
        df = c.extract_3D_data(save_logs)
        c.register_chain(df)

    # Small check that all nucleotides of a chain have an entry in nucleotide table
    if not c.delete_me:
        with sqlite3.connect(runDir+"/results/RNANet.db", timeout=10.0) as conn:
            conn.execute('pragma journal_mode=wal')
            nnts = sql_ask_database(conn, f"SELECT COUNT(index_chain) FROM nucleotide WHERE chain_id={c.db_chain_id};", warn_every=10)[0][0]
        if not(nnts):
            warn(f"Nucleotides not inserted: {c.error_messages}")
            c.delete_me = True
            c.error_messages = "Nucleotides not inserted !"
        else:
            notify(f"Inserted {nnts} nucleotides to chain {c.chain_label}")

    # extract the portion we want
    if extract and not c.delete_me:
        c.extract(df, khetatm)

    return c

@trace_unhandled_exceptions
def work_prepare_sequences(dl, rfam_acc, chains):
    """Prepares FASTA files of homologous sequences to realign with cmalign or SINA.
    """

    setproctitle("RNAnet.py work_prepare_sequences()")

    if rfam_acc in LSU_set | SSU_set:  # rRNA
        if os.path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.afa"):
            # Detect doublons and remove them
            existing_afa = AlignIO.read(path_to_seq_data + f"realigned/{rfam_acc}++.afa", "fasta")
            existing_ids = [r.id for r in existing_afa]
            del existing_afa
            new_ids = [str(c) for c in chains]
            doublons = [i for i in existing_ids if i in new_ids]
            del existing_ids, new_ids
            if len(doublons):
                warn(f"Removing {len(doublons)} doublons from existing {rfam_acc}++.fa and using their newest version")
                fasta = path_to_seq_data + f"realigned/{rfam_acc}++.fa"
                seqfile = SeqIO.parse(fasta, "fasta")
                # remove it and rewrite it with its own content filtered
                os.remove(fasta)
                with open(fasta, 'w') as f:
                    for rec in seqfile:
                        if rec.id not in doublons:
                            f.write(format(rec, "fasta"))

        # Add the new sequences with previous ones, if any
        with open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "a") as f:
            for c in chains:
                if len(c.seq_to_align):
                    f.write(f"> {str(c)}\n"+c.seq_to_align.replace('-', '').replace('U', 'T')+'\n')
        status = f"{rfam_acc}: {len(chains)} new PDB sequences to align (with SINA)"

    elif not os.path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.stk"):
        # there was no previous aligned sequences, and we use cmalign.
        # So, we need to download homologous sequences from Rfam.

        # Extracting covariance model for this family
        if not os.path.isfile(path_to_seq_data + f"realigned/{rfam_acc}.cm"):
            with open(path_to_seq_data + f"realigned/{rfam_acc}.cm", "w") as f:
                subprocess.run(["cmfetch", path_to_seq_data + "Rfam.cm", rfam_acc], stdout=f)
            notify(f"Extracted {rfam_acc} covariance model (cmfetch)")

        # Download homologous sequences
        dl.download_Rfam_sequences(rfam_acc)

        # Prepare a FASTA file containing Rfamseq hits for that family
        if os.path.isfile(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz"): # test if download succeeded

            # gunzip the file
            with gzip.open(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz", 'rb') as gz:
                file_content = gz.read()
            with open(path_to_seq_data + f"realigned/{rfam_acc}.fa", "wb") as plusplus:
                plusplus.write(file_content)

            # Write the combined fasta file
            with open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "w") as plusplus:
                ids = set()
                # Remove doublons from the Rfam hits
                for r in SeqIO.parse(path_to_seq_data + f"realigned/{rfam_acc}.fa", "fasta"):
                    if r.id not in ids:
                        ids.add(r.id)
                        plusplus.write('> '+r.description+'\n'+str(r.seq)+'\n')
                # Add the 3D chains sequences
                for c in chains:
                    if len(c.seq_to_align):
                        plusplus.write(f"> {str(c)}\n"+c.seq_to_align.replace('-', '').replace('U', 'T')+'\n')

            del file_content
            # os.remove(path_to_seq_data + f"realigned/{rfam_acc}.fa")

        else:
            raise Exception(rfam_acc + "sequences download failed !")

        status = f"{rfam_acc}: {len(ids)} hits + {len(chains)} PDB sequences to align (with cmalign)"

    else:  # We are using cmalign and a previous alignment exists
        # Add the new sequences to a separate FASTA file
        with open(path_to_seq_data + f"realigned/{rfam_acc}_new.fa", "w") as f:
            for c in chains:
                if len(c.seq_to_align):
                    f.write(f"> {str(c)}\n"+c.seq_to_align.replace('-', '').replace('U', 'T')+'\n')
        status = f"{rfam_acc}: {len(chains)} new PDB sequences to realign (with existing cmalign alignment)"

    # print some stats
    notify(status)

@trace_unhandled_exceptions
def work_realign(rfam_acc):
    """ Runs multiple sequence alignements by RNA family.

    It aligns the Rfam hits from a RNA family with the sequences from the list of chains. 
    Rfam covariance models are used with Infernal tools, except for rRNAs. 
    cmalign requires too much RAM for them, so we use SINA, a specifically designed tool for rRNAs.
    """

    setproctitle(f"RNAnet.py work_realign({rfam_acc})")

    if rfam_acc in LSU_set | SSU_set:
        # Ribosomal subunits deserve a special treatment.
        # They require too much RAM to be aligned with Infernal.
        # Then we will use SINA instead.
        if rfam_acc in ["RF00177", "RF01960"]:
            arbfile = "realigned/SSU.arb"
        else:
            arbfile = "realigned/LSU.arb"

        # Run alignment
        p = subprocess.run(["sina", "-i", path_to_seq_data + f"realigned/{rfam_acc}++.fa",
                                    "-o", path_to_seq_data + f"realigned/{rfam_acc}++.afa",
                                    "-r", path_to_seq_data + arbfile,
                                    "--meta-fmt=csv"])
    else:
        # Align using Infernal for most RNA families

        if os.path.isfile(path_to_seq_data + "realigned/" + rfam_acc + "++.stk"):
            # Alignment exists. We just want to add new sequences into it.

            if not os.path.isfile(path_to_seq_data + f"realigned/{rfam_acc}_new.fa"):
                # there are no new sequences to align...
                return

            existing_ali_path = path_to_seq_data + f"realigned/{rfam_acc}++.stk"
            new_ali_path = path_to_seq_data + f"realigned/{rfam_acc}_new.stk"

            # Align the new sequences
            with open(new_ali_path, 'w') as o:
                p1 = subprocess.run(["cmalign", path_to_seq_data + f"realigned/{rfam_acc}.cm",
                                     path_to_seq_data + f"realigned/{rfam_acc}_new.fa"],
                                    stdout=o, stderr=subprocess.PIPE)
            notify("Aligned new sequences together")

            # Detect doublons and remove them
            existing_stk = AlignIO.read(existing_ali_path, "stockholm")
            existing_ids = [r.id for r in existing_stk]
            del existing_stk
            new_stk = AlignIO.read(new_ali_path, "stockholm")
            new_ids = [r.id for r in new_stk]
            del new_stk
            doublons = [i for i in existing_ids if i in new_ids]
            del existing_ids, new_ids
            if len(doublons):
                warn(f"Removing {len(doublons)} doublons from existing {rfam_acc}++.stk and using their newest version")
                with open(path_to_seq_data + "realigned/toremove.txt", "w") as toremove:
                    toremove.write('\n'.join(doublons)+'\n')
                p = subprocess.run(["esl-alimanip", "--seq-r", path_to_seq_data + "realigned/toremove.txt", "-o", existing_ali_path+"2", existing_ali_path],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                p = subprocess.run(["mv", existing_ali_path+"2", existing_ali_path],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                os.remove(path_to_seq_data + "realigned/toremove.txt")

            # And we merge the two alignments
            p2 = subprocess.run(["esl-alimerge", "-o", path_to_seq_data + f"realigned/{rfam_acc}_merged.stk",
                                 "--rna", existing_ali_path, new_ali_path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            stderr = p1.stderr.decode('utf-8') + p2.stderr.decode('utf-8')
            subprocess.run(["mv", path_to_seq_data + f"realigned/{rfam_acc}_merged.stk", existing_ali_path])
            notify("Merged alignments into one")

            # remove the partial files
            os.remove(new_ali_path)
            os.remove(path_to_seq_data + f"realigned/{rfam_acc}_new.fa")

        else:
            # Alignment does not exist yet. We need to compute it from scratch.
            print(f"\t> Aligning {rfam_acc} sequences together (cmalign) ...", end='', flush=True)

            p = subprocess.run(["cmalign", "--small", "--cyk", "--noprob", "--nonbanded", "--notrunc",
                                '-o', path_to_seq_data + f"realigned/{rfam_acc}++.stk",
                                path_to_seq_data + f"realigned/{rfam_acc}.cm",
                                path_to_seq_data + f"realigned/{rfam_acc}++.fa"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            stderr = p.stderr.decode("utf-8")

        if len(stderr):
            print('', flush=True)
            warn(f"Error during sequence alignment: {stderr}", error=True)
            with open(runDir + "/errors.txt", "a") as er:
                er.write(f"Attempting to realign {rfam_acc}:\n" + stderr + '\n')
            return 1
        else:
            print('\t'+validsymb, flush=True)

        # Convert Stockholm to aligned FASTA
        subprocess.run(["esl-reformat", "-o", path_to_seq_data + f"realigned/{rfam_acc}++.afa", 
                        "--informat", "stockholm", 
                        "afa", path_to_seq_data + f"realigned/{rfam_acc}++.stk"])
        subprocess.run(["rm", "-f", "esltmp*"]) # We can, because we are not running in parallel for this part.

    # Assert everything worked, or save an error
    with open(path_to_seq_data + f"realigned/{rfam_acc}++.afa", 'r') as output:
        if not len(output.readline()):
            # The process crashed, probably because of RAM overflow
            warn(f"Failed to realign {rfam_acc} (killed)", error=True)
            with open(runDir + "/errors.txt", "a") as er:
                er.write(f"Failed to realign {rfam_acc} (killed)")

@trace_unhandled_exceptions
def work_pssm_remap(f):
    """Computes Position-Specific-Scoring-Matrices given the multiple sequence alignment of the RNA family.
    This also remaps the 3D object sequence with the aligned sequence in the MSA.
    If asked, the 3D object sequence is completed by the consensus nucleotide when one of them is missing.

    Uses only 1 core, so this function can be called in parallel.

    """

    # Get a worker number to position the progress bar
    global idxQueue
    thr_idx = idxQueue.get()

    ##########################################################################################
    #                           Compute frequencies in the alignment
    ##########################################################################################

    setproctitle(f"RNAnet.py work_pssm_remap({f}) compute PSSMs")

    # Open the alignment
    try:
        align = read(path_to_seq_data + f"realigned/{f}++.afa") # This is our custom AlignIO overload which uses MutableSeq instead of Seq
    except:
        warn(f"{f}'s alignment is wrong. Recompute it and retry.", error=True)
        with open(runDir + "/errors.txt", "a") as errf:
            errf.write(f"{f}'s alignment is wrong. Recompute it and retry.\n")
        return 1
    nseqs = len(align)
    ncols = align.get_alignment_length()

    # Compute statistics per column
    pssm_info = np.zeros((6, ncols))
    res_index = {'A':0, 'C':1, 'G':2, 'U':3, 'N':4, '-':5}
    letters = "ACGUN"
    consensus = []
    
    for residue_num in tqdm(range(ncols), position=thr_idx+1, desc=f"Worker {thr_idx+1}: Count bases in fam {f}", leave=False):
        
        # Count the bases (iterate lines)
        for record in align:
            letter = record.seq[residue_num].upper().replace('.','-')
            try:
                idx = res_index[letter]
            except KeyError:
                # warn(f"Unknown residue found in {family} family: {letter}", error=True)
                # These are K, R, etc from Rfam. The RNANet sequences provided are pure ACGUN, but not the Rfam ones.
                idx = 4 # consider it is N
            pssm_info[idx,residue_num] += 1.0
        
        # Get the number of non-gap nucleotides
        N = 0
        for i in range(5):
            N += pssm_info[i,residue_num]

        if N>0:
            # Divide base counts by number of non-gaps
            for i in range(5):
                pssm_info[i,residue_num] /= N
        
        # last line is for the gap percentage (Ngaps/Nlines)
        pssm_info[5,residue_num] /= nseqs

        # Define consensus base for this position:
        if pssm_info[5,residue_num] > 0.7:
            # gaps are in majority if over 75% (that's my definition)
            consensus.append('-')
        else:
            idx = np.argmax(pssm_info[0:5,residue_num])
            if pssm_info[idx, residue_num] > 0.5:
                consensus.append(letters[idx])
            else:
                consensus.append('N')

    # At this point, pssm_info is a numpy array containing the PSSM and consensus a list of consensus chars.


    ##########################################################################################
    #           Remap sequences of the 3D chains with sequences in the alignment
    ##########################################################################################

    setproctitle(f"RNAnet.py work_pssm_remap({f}) remap")

    # For each sequence, remap chain residues with sequence alignment 
    columns_to_save = set()
    re_mappings = []
    pbar = tqdm(total=nseqs, position=thr_idx+1, desc=f"Worker {thr_idx+1}: Remap {f} chains", leave=False)
    pbar.update(0)
    for s in align:
        # skip Rfamseq entries
        if not '[' in s.id:  
            continue

        # Get the chain id in the database
        conn = sqlite3.connect(runDir + '/results/RNANet.db', timeout=10.0)
        conn.execute('pragma journal_mode=wal')
        db_id = sql_ask_database(conn, f"SELECT chain_id FROM chain WHERE structure_id = '{s.id.split('[')[0]}' AND chain_name = '{s.id.split('-')[1]}' AND rfam_acc = '{f}';")
        if len(db_id):
            db_id = db_id[0][0]
        else:
            conn.close()
            warn(f"Bizarre... sequence {s.id} is not found in the database ! Cannot remap it ! Ignoring...")
            pbar.update(1)
            continue
        seq_to_align = ''.join([ x[0] for x in sql_ask_database(conn, f"SELECT nt_align_code FROM nucleotide WHERE chain_id = {db_id} ORDER BY index_chain ASC;")])
        full_length = len(seq_to_align)
        conn.close()

        # Save colums in the appropriate positions
        i = 0   # to iterate the object sequence
        j = 0   # to iterate the alignment sequence
        while i < full_length and j < ncols:
            # Here we try to map seq_to_align (the sequence of the 3D chain, including gaps when residues are missing),
            # with s.seq, the sequence aligned in the MSA, containing any of ACGU and two types of gaps, - and .

            if seq_to_align[i] == s.seq[j].upper():      # alignment and sequence correspond (incl. gaps)
                re_mappings.append((db_id, i+1, j+1))    # because index_chain in table nucleotide is in [1,N], we use i+1 and j+1.
                columns_to_save.add(j+1)    # it's a set, doublons are automaticaly ignored
                i += 1
                j += 1
            elif seq_to_align[i] == '-':   # '-' in the chain, but '.' or letter in the aligned sequence
                # search for a gap to the consensus nearby
                k = 0  # Search must start at zero to assert the difference comes from '-' in front of '.'
                while j+k < ncols and s.seq[j+k] == '.':
                    k += 1

                # if found, set j to that position
                if j+k < ncols and s.seq[j+k] == '-':
                    re_mappings.append((db_id, i+1, j+k+1))
                    columns_to_save.add(j+k+1)
                    i += 1
                    j += k+1
                    continue

                # if not, take the insertion gap if this is one
                if j < ncols and s.seq[j] == '.':
                    re_mappings.append((db_id, i+1, j+1))
                    columns_to_save.add(j+1)
                    s.seq[j] = '-' # We replace the insertion gap by a real gap (thanks to MutableSeqs)
                    i += 1
                    j += 1
                    continue

                # else, just mark the gap as unknown (there is an alignment mismatch '-' in the 3D facing a letter in the alignment)
                re_mappings.append((db_id, i+1, 0))
                i += 1
            elif s.seq[j] in ['.', '-']:  # gap in the alignment, but not in the real chain
                j += 1  # ignore the column
            else:  # sequence mismatch which is not a gap...
                print(f"You are never supposed to reach this. Comparing {s.id} in {i} ({seq_to_align[i-1:i+2]}) with seq[{j}] ({s.seq[j-3:j+4]}).",
                      seq_to_align, s.seq, sep='\n', flush=True)
                raise Exception('Something is wrong with sequence alignment.')

        pbar.update(1)
    pbar.close()

    # Get a sorted list from the set
    columns = sorted(columns_to_save)

    ##########################################################################################
    #           Save the alignment columns and their mappings to the database
    ##########################################################################################

    setproctitle(f"RNAnet.py work_pssm_remap({f}) saving")

    # Save the re_mappings
    conn = sqlite3.connect(runDir + '/results/RNANet.db', timeout=20.0)
    conn.execute('pragma journal_mode=wal') # Allow multiple other readers to ask things while we execute this writing query
    sql_execute(conn, """INSERT INTO re_mapping (chain_id, index_chain, index_ali) 
                         VALUES (?, ?, ?) 
                         ON CONFLICT(chain_id, index_chain) DO UPDATE SET index_ali=excluded.index_ali;""",
                many=True, data=re_mappings)

    # Delete alignment columns that are not used anymore from the database
    current_family_columns = [ x[0] for x in sql_ask_database(conn, f"SELECT index_ali FROM align_column WHERE rfam_acc = '{f}';")]
    unused = []
    for col in current_family_columns:
        if col not in columns_to_save:
            unused.append((f, col))
    sql_execute(conn, """DELETE FROM align_column WHERE rfam_acc = ? AND index_ali = ?;""", many=True, data=unused)
    conn.commit()

    # Save the useful columns in the database
    data = [(f, j) + tuple(pssm_info[:,j-1]) + (consensus[j-1],) for j in sorted(columns_to_save)]
    sql_execute(conn, """INSERT INTO align_column (rfam_acc, index_ali, freq_A, freq_C, freq_G, freq_U, freq_other, gap_percent, consensus)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(rfam_acc, index_ali) DO 
                         UPDATE SET freq_A=excluded.freq_A, freq_C=excluded.freq_C, freq_G=excluded.freq_G, freq_U=excluded.freq_U, 
                                    freq_other=excluded.freq_other, gap_percent=excluded.gap_percent, consensus=excluded.consensus;""", many=True, data=data)
    # Add an unknown values column, with index_ali 0 (for nucleotides unsolved in 3D giving a gap '-' but found facing letter in the alignment)
    sql_execute(conn, f"""INSERT OR IGNORE INTO align_column (rfam_acc, index_ali, freq_A, freq_C, freq_G, freq_U, freq_other, gap_percent, consensus)
                          VALUES (?, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, '-');""", data=(f,))
    # Save the number of "used columns" to table family ( = the length of the alignment if it was composed only of the RNANet chains)
    sql_execute(conn, f"UPDATE family SET ali_filtered_len = ? WHERE rfam_acc = ?;", data=(len(columns_to_save), f))
    conn.close()

    ##########################################################################################
    #               Saving the filtered alignement with only the saved positinos
    ##########################################################################################

    setproctitle(f"RNAnet.py work_pssm_remap({f}) filtering alignment")

    # filter the alignment
    names = [ x.id for x in align if '[' in x.id ]
    align = align[-len(names):]
    filtered_alignment = align[:, 1:1] # all the lines, but no columns
    for p in columns: 
        filtered_alignment += align[:, p-1:p] # save columns one by one

    # write it to file in both STK and FASTA formats (STK required for distance matrices in statistics)
    with open(path_to_seq_data+f"/realigned/{f}_3d_only.stk", "w") as only_3d:
        try:
            only_3d.write(format(filtered_alignment, "stockholm"))
        except ValueError as e:
            warn(e)
    with open(path_to_seq_data+f"/realigned/{f}_3d_only.afa", "w") as only_3d:
        try:
            only_3d.write(format(filtered_alignment, "fasta"))
        except ValueError as e:
            warn(e)

    setproctitle(f"RNAnet.py work_pssm_remap({f}) finished")
    idxQueue.put(thr_idx) # replace the thread index in the queue
    return 0

@trace_unhandled_exceptions
def work_save(c, homology=True):

    setproctitle(f"RNAnet.py work_save({c.chain_label})")

    conn = sqlite3.connect(runDir + "/results/RNANet.db", timeout=15.0)
    conn.execute('pragma journal_mode=wal')
    if homology:
        df = pd.read_sql_query(f"""
                SELECT index_chain, old_nt_resnum, nt_position, nt_name, nt_code, nt_align_code, 
                is_A, is_C, is_G, is_U, is_other, freq_A, freq_C, freq_G, freq_U, freq_other, gap_percent, consensus, dbn,
                paired, nb_interact, pair_type_LW, pair_type_DSSR, alpha, beta, gamma, delta, epsilon, zeta, epsilon_zeta,
                chi, bb_type, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
                v0, v1, v2, v3, v4, amplitude, phase_angle, puckering FROM 
                (SELECT chain_id, rfam_acc from chain WHERE chain_id = {c.db_chain_id})
                NATURAL JOIN re_mapping
                NATURAL JOIN nucleotide
                NATURAL JOIN align_column;""",
             conn)
        filename = path_to_3D_data + "datapoints/" + c.chain_label + '.' + c.mapping.rfam_acc
    else:
        df = pd.read_sql_query(f"""
                SELECT index_chain, old_nt_resnum, nt_position, nt_name, nt_code, nt_align_code, 
                is_A, is_C, is_G, is_U, is_other, dbn,
                paired, nb_interact, pair_type_LW, pair_type_DSSR, alpha, beta, gamma, delta, epsilon, zeta, epsilon_zeta,
                chi, bb_type, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
                v0, v1, v2, v3, v4, amplitude, phase_angle, puckering FROM 
                nucleotide WHERE chain_id = {c.db_chain_id} ORDER BY index_chain ASC;""",
            conn)
        filename = path_to_3D_data + "datapoints/" + c.chain_label
    conn.close()

    df.to_csv(filename, float_format="%.2f", index=False)


if __name__ == "__main__":

    fileDir = os.path.dirname(os.path.realpath(__file__))
    ncores = read_cpu_number()
    pp = Pipeline()
    pp.process_options()
    print(f"> Running {python_executable} on {ncores} CPU cores in folder {runDir}.")

    # Prepare folders
    os.makedirs(runDir + "/results", exist_ok=True)
    os.makedirs(runDir + "/data", exist_ok=True)
    subprocess.run(["rm", "-f", runDir+"/errors.txt"])

    # Check existence of the database, or create it
    with sqlite3.connect(runDir + '/results/RNANet.db') as conn:
        sql_define_tables(conn)
    print("> Storing results into", runDir + "/results/RNANet.db")

    # compute an update compared to what is in the table "chain" (comparison on structure_id + chain_name + rfam_acc).
    # If --all was passed, all the structures are kept.
    # Fills pp.update with Chain() objects.
    pp.list_available_mappings()

    # ===========================================================================
    # 3D information
    # ===========================================================================

    # Download and annotate new RNA 3D chains (Chain objects in pp.update)
    # If the original cif file and/or the Json DSSR annotation file already exist, they are not redownloaded/recomputed.
    pp.dl_and_annotate(coeff_ncores=0.5)
    print("Here we go.")

    # At this point, the structure table is up to date.
    # Now save the DSSR annotations to the database.
    # Extract the 3D chains to separate structure files if asked with --extract.
    pp.build_chains(coeff_ncores=1.0)

    if len(pp.to_retry):
        # Redownload and re-annotate
        print("> Retrying to annotate some structures which just failed.", flush=True)
        pp.dl_and_annotate(retry=True, coeff_ncores=0.3)  #
        pp.build_chains(retry=True, coeff_ncores=0.5)     # Use half the cores to reduce required amount of memory
    print(f"> Loaded {len(pp.loaded_chains)} RNA chains ({len(pp.update) - len(pp.loaded_chains)} ignored/errors).")
    if len(no_nts_set):
        print(f"Among errors, {len(no_nts_set)} structures seem to contain RNA chains without defined nucleotides:", no_nts_set, flush=True)
    if len(weird_mappings):
        print(f"{len(weird_mappings)} mappings to Rfam were taken as absolute positions instead of residue numbers:", weird_mappings, flush=True)
    if pp.SELECT_ONLY is None:
        pp.checkpoint_save_chains()

    if not pp.HOMOLOGY:
        # Save chains to file
        for c in pp.loaded_chains:
            work_save(c, homology=False)
        print("Completed.")
        exit(0)

    # At this point, structure, chain and nucleotide tables of the database are up to date.
    # (Modulo some statistics computed by statistics.py)

    # ===========================================================================
    # Homology information
    # ===========================================================================

    if pp.SELECT_ONLY is None:
        # If your job failed, you can comment all the "3D information" part and start from here.
        pp.checkpoint_load_chains()

    # Get the list of Rfam families found in the update
    rfam_acc_to_download = {}
    for c in pp.loaded_chains:
        if c.mapping.rfam_acc not in rfam_acc_to_download.keys():
            rfam_acc_to_download[c.mapping.rfam_acc] = [c]
        else:
            rfam_acc_to_download[c.mapping.rfam_acc].append(c)

    print(f"> Identified {len(rfam_acc_to_download.keys())} families to update and re-align with the crystals' sequences")
    pp.fam_list = sorted(rfam_acc_to_download.keys())

    if len(pp.fam_list):
        pp.prepare_sequences()
        pp.realign()

        # At this point, the family table is almost up to date 
        # (lacking idty_percent and ali_filtered_length, both set in statistics.py)

        thr_idx_mgr = Manager()
        idxQueue = thr_idx_mgr.Queue()

        pp.remap()

    # At this point, the align_column and re_mapping tables are up-to-date.

    # ==========================================================================================
    # Prepare the results
    # ==========================================================================================

    pp.sanitize_database()
    pp.output_results()

    print("Completed.") # This part of the code is supposed to release some serotonin in the modeller's brain, do not remove

    # so i can sleep for the end of the night
    # subprocess.run(["poweroff"])
