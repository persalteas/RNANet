#!/usr/bin/python3.8
import numpy as np
import pandas as pd
import concurrent.futures, Bio.PDB.StructureBuilder, getopt, gzip, io, json, os, psutil, re, requests, signal, sqlalchemy, sqlite3, subprocess, sys, time, traceback, warnings
from Bio import AlignIO, SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict 
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio._py3k import urlretrieve as _urlretrieve
from Bio._py3k import urlcleanup as _urlcleanup
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment, AlignInfo
from collections import OrderedDict
from functools import partial, wraps
from os import path, makedirs
from multiprocessing import Pool, Manager, set_start_method
from time import sleep
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


pd.set_option('display.max_rows', None)
sqlite3.enable_callback_tracebacks(True)
m = Manager()
running_stats = m.list()
running_stats.append(0) # n_launched
running_stats.append(0) # n_finished
running_stats.append(0) # n_skipped
path_to_3D_data = "tobedefinedbyoptions"
path_to_seq_data = "tobedefinedbyoptions"
validsymb = '\U00002705'
warnsymb = '\U000026A0'
errsymb = '\U0000274C'

# Default options:
CRYSTAL_RES = 4.0
KEEP_HETATM = False
FILL_GAPS = True 
HOMOLOGY = True
USE_KNOWN_ISSUES = True
RUN_STATS = False
EXTRACT_CHAINS = False

class NtPortionSelector(object):
    """Class passed to MMCIFIO to select some chain portions in an MMCIF file.

    Validates every chain, residue, nucleotide, to say if it is in the selection or not.
    """

    def __init__(self, model_id, chain_id, start, end):
        self.chain_id = chain_id
        self.start = start
        self.end = end
        self.pdb_model_id = model_id
        self.hydrogen_regex = re.compile("[123 ]*H.*")

    def accept_model(self, model):
        return int(model.get_id() == self.pdb_model_id)

    def accept_chain(self, chain):
        return int(chain.get_id() == self.chain_id)

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()

        # Refuse waters and magnesium ions
        if hetatm_flag in ["W", "H_MG"]:
            return int(KEEP_HETATM)      

        # I don't really know what this is but the doc said to warn:          
        if icode != " ":
            pass
            # warn(f"icode {icode} at position {resseq}\t\t")

        # Accept the residue if it is in the right interval:
        return int(self.start <= resseq <= self.end)

    def accept_atom(self, atom):

        # Refuse hydrogens
        if self.hydrogen_regex.match(atom.get_id()):
            return 0 

        # Accept all atoms otherwise.
        return 1


class BufferingSummaryInfo(AlignInfo.SummaryInfo):

    def get_pssm(self, family, index):
        """Create a position specific score matrix object for the alignment. 
 
        This creates a position specific score matrix (pssm) which is an 
        alternative method to look at a consensus sequence. 
 
        Returns: 
         - A PSSM (position specific score matrix) object. 
        """ 

        pssm_info = [] 
        # now start looping through all of the sequences and getting info 
        for residue_num in tqdm(range(self.alignment.get_alignment_length()), position=index+1, desc=f"Worker {index+1}: {family}", leave=False): 
            score_dict = self._get_base_letters("ACGUN") 
            for record in self.alignment: 
                this_residue = record.seq[residue_num].upper() 
                if this_residue not in "-.": 
                    try:
                        score_dict[this_residue] += 1.0 
                    except KeyError:
                        # if this_residue in "acgun":
                        #     warn(f"Found {this_residue} in {family} alignment...")
                        score_dict[this_residue] = 1.0
            pssm_info.append(('*', score_dict))

        return AlignInfo.PSSM(pssm_info)


class Chain:
    """ The object which stores all our data and the methods to process it.

    Chains accumulate information through this scipt, and are saved to files at the end of major steps."""

    def __init__(self, pdb_id, pdb_model, pdb_chain_id, chain_label, rfam="", inferred=False, pdb_start=None, pdb_end=None):
        self.pdb_id = pdb_id                    # PDB ID
        self.pdb_model = int(pdb_model)         # model ID, starting at 1
        self.pdb_chain_id = pdb_chain_id        # chain ID (mmCIF), multiple letters
        self.pdb_start = pdb_start              # if portion of chain, the start number (relative to the chain, not residue numbers)
        self.pdb_end = pdb_end                  # if portion of chain, the start number (relative to the chain, not residue numbers)
        self.reversed = (pdb_start > pdb_end)   # wether pdb_start > pdb_end in the Rfam mapping
        self.chain_label = chain_label          # chain pretty name 
        self.file = ""                          # path to the 3D PDB file
        self.rfam_fam = rfam                    # mapping to an RNA family
        self.inferred = inferred                # Wether this mapping has been inferred from BGSU's NR list
        self.seq = ""                           # sequence with modified nts
        self.seq_to_align = ""                  # sequence with modified nts replaced, but gaps can exist
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

    def extract(self):
        """ Extract the part which is mapped to Rfam from the main CIF file and save it to another file.
        """
        
        if not HOMOLOGY:
            status = f"Extract {self.pdb_id}-{self.pdb_chain_id}"
            self.file = path_to_3D_data+"rna_only/"+self.chain_label+".cif"
        else:
            status = f"Extract {self.pdb_start}-{self.pdb_end} atoms from {self.pdb_id}-{self.pdb_chain_id}"
            self.file = path_to_3D_data+"rna_mapped_to_Rfam/"+self.chain_label+".cif"

        # Check if file exists, if yes, abort (do not recompute)
        if os.path.exists(self.file):
            notify(status, "using previous file")
            return

        model_idx = self.pdb_model - (self.pdb_model > 0) # because arrays start at 0, models start at 1
       
        with warnings.catch_warnings():
            # Ignore the PDB problems. This mostly warns that some chain is discontinuous.
            warnings.simplefilter('ignore', PDBConstructionWarning)  

            # Load the whole mmCIF into a Biopython structure object:
            mmcif_parser = MMCIFParser()
            s = mmcif_parser.get_structure(self.pdb_id, path_to_3D_data + "RNAcifs/"+self.pdb_id+".cif")

            # Extract the desired chain
            c = s[model_idx][self.pdb_chain_id]

            if not HOMOLOGY:
                # Define a selection
                start = c.child_list[0].get_id()[1]  # the chain's first residue is numbered 'first_number'
                end = c.child_list[-1].get_id()[1]   # the chain's last residue number
            else:
                # Pay attention to residue numbering
                first_number = c.child_list[0].get_id()[1]          # the chain's first residue is numbered 'first_number'
                if self.pdb_start < self.pdb_end:                             
                    start = self.pdb_start + first_number - 1       # shift our start_position by 'first_number'
                    end = self.pdb_end + first_number - 1           # same for the end position
                else:
                    self.reversed = True                            # the 3D chain is numbered backwards compared to the Rfam family
                    end = self.pdb_start + first_number - 1
                    start = self.pdb_end + first_number - 1

            # Define a selection
            sel = NtPortionSelector(model_idx, self.pdb_chain_id, start, end)

            # Save that selection on the mmCIF object s to file
            ioobj = MMCIFIO()
            ioobj.set_structure(s)
            ioobj.save(self.file, sel)

        notify(status)

    def extract_3D_data(self, conn):
        """ Maps DSSR annotations to the chain. """

        # Load the mmCIF annotations from file
        try:
            with open(path_to_3D_data + "annotations/" + self.pdb_id + ".json", 'r') as json_file:
                json_object = json.load(json_file)
            notify(f"Read {self.chain_label} DSSR annotations")
        except json.decoder.JSONDecodeError as e:
            warn("Could not load "+self.pdb_id+f".json with JSON package: {e}", error=True)
            self.delete_me = True
            self.error_messages = f"Could not load existing {self.pdb_id}.json file: {e}"
            return 1
                
        # Print eventual warnings given by DSSR, and abort if there are some
        if "warning" in json_object.keys():
            warn(f"Ignoring {self.chain_label} ({json_object['warning']})")
            self.delete_me = True
            self.error_messages = f"DSSR warning for {self.chain_label}: {json_object['warning']}"
            return 1

        try:
            # Prepare a data structure (Pandas DataFrame) for the nucleotides
            nts = json_object["nts"]                         # sub-json-object
            df = pd.DataFrame(nts)                           # conversion to dataframe
            df = df[ df.chain_name == self.pdb_chain_id ]    # keeping only this chain's nucleotides

            # remove low pertinence or undocumented descriptors
            cols_we_keep = ["index_chain", "nt_resnum", "nt_name", "nt_code", "nt_id", "dbn", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                "epsilon_zeta", "bb_type", "chi", "glyco_bond", "form", "ssZp", "Dp", "eta", "theta", "eta_prime", "theta_prime", "eta_base", "theta_base",
                "v0", "v1", "v2", "v3", "v4", "amplitude", "phase_angle", "puckering" ]
            df = df[cols_we_keep]

            # Convert angles to radians
            df.loc[:,['alpha', 'beta','gamma','delta','epsilon','zeta','epsilon_zeta','chi','v0', 'v1', 'v2', 'v3', 'v4',
                        'eta','theta','eta_prime','theta_prime','eta_base','theta_base', 'phase_angle']] *= np.pi/180.0
            # mapping [-pi, pi] into [0, 2pi]
            df.loc[:,['alpha', 'beta','gamma','delta','epsilon','zeta','epsilon_zeta','chi','v0', 'v1', 'v2', 'v3', 'v4',
                        'eta','theta','eta_prime','theta_prime','eta_base','theta_base', 'phase_angle']] %= (2.0*np.pi)
        except KeyError as e:
            warn(f"Error while parsing DSSR's {self.chain_label} json output:{e}", error=True)
            self.delete_me = True
            self.error_messages = f"Error while parsing DSSR's json output:\n{e}"
            return 1

        # Add a sequence column just for the alignments
        df['nt_align_code'] = [ str(x).upper()
                                        .replace('NAN', '-')      # Unresolved nucleotides are gaps
                                        .replace('?', '-')        # Unidentified residues, let's delete them
                                        .replace('T', 'U')        # 5MU are modified to t, which gives T
                                        .replace('P', 'U')        # Pseudo-uridines, but it is not really right to change them to U, see DSSR paper, Fig 2
                                for x in df['nt_code'] ]

        # Shift numbering when duplicate residue numbers are found.
        # Example: 4v9q-DV contains 17 and 17A which are both read 17 by DSSR.
        while True in df.duplicated(['nt_resnum']).values:
            i = df.duplicated(['nt_resnum']).values.tolist().index(True)
            df.iloc[i:, 1] += 1

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
        # Rfam mapping           3 |------------------------------------------ ... -------| 3353
        # nt resnum              3 |--------------------------------|  156
        # index_chain            1 |-------------|77 83|------------|  149
        # expected data point    1 |--------------------------------|  154
        #
        try:
            l = df.iloc[-1,1] - df.iloc[0,1] + 1    # length of chain from nt_resnum point of view
        except IndexError:
            warn(f"Error while parsing DSSR's annotation: No nucleotides are part of {self.chain_label}!", error=True)
            self.delete_me = True
            self.error_messages = f"Error while parsing DSSR's json output: No nucleotides from {self.chain_label}. We expect a problem with {self.pdb_id} mmCIF download. Delete it and retry."
            return 1

        if l != len(df['index_chain']):         # if some residues are missing, len(df['index_chain']) < l
            resnum_start = df.iloc[0,1]
            diff = set(range(l)).difference(df['nt_resnum'] - resnum_start)     # the rowIDs the missing nucleotides would have (rowID = index_chain - 1 = nt_resnum - resnum_start)
            for i in sorted(diff):
                # Add a row at position i
                df = pd.concat([    df.iloc[:i], 
                                    pd.DataFrame({"index_chain": i+1, "nt_resnum": i+resnum_start, 
                                                    "nt_code":'-', "nt_name":'-', 'nt_align_code':'-'}, index=[i]), 
                                    df.iloc[i:]
                                ])
                # Increase the index_chain of all following lines
                df.iloc[i+1:, 0] += 1
            df = df.reset_index(drop=True)
        self.full_length = len(df.index_chain)

        # One-hot encoding sequence
        df["is_A"] = [ 1 if x in "Aa" else 0 for x in df["nt_code"] ]
        df["is_C"] = [ 1 if x in "Cc" else 0 for x in df["nt_code"] ]
        df["is_G"] = [ 1 if x in "Gg" else 0 for x in df["nt_code"] ]
        df["is_U"] = [ 1 if x in "Uu" else 0 for x in df["nt_code"] ]
        df["is_other"] = [ 0 if x in "ACGUacgu" else 1 for x in df["nt_code"] ]
        df["nt_position"] = [ float(i+1)/self.full_length for i in range(self.full_length) ]

        # Iterate over pairs to identify base-base interactions
        res_ids = list(df['nt_id']) # things like "chainID.C4, chainID.U5"
        paired = [ '' ] * self.full_length
        pair_type_LW = [ '' ] * self.full_length
        pair_type_DSSR = [ '' ] * self.full_length
        interacts = [ 0 ] * self.full_length
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
                        paired[nt1_idx] = str(nt2_idx + 1)
                    else:
                        pair_type_LW[nt1_idx] += ',' + lw_pair
                        pair_type_DSSR[nt1_idx] += ',' + dssr_pair
                        paired[nt1_idx] += ',' + str(nt2_idx + 1)
                
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

        df['paired'] = paired
        df['pair_type_LW'] = pair_type_LW
        df['pair_type_DSSR'] = pair_type_DSSR
        df['nb_interact'] = interacts
        df = df.drop(['nt_id'], axis=1) # remove now useless descriptors

        if self.reversed:
            # The 3D structure is numbered from 3' to 5' instead of standard 5' to 3'
            # or the sequence that matches the Rfam family is 3' to 5' instead of standard 5' to 3'.
            # Anyways, you need to invert the angles.
            # TODO: angles alpha, beta, etc
            warn(f"Has {self.chain_label} been numbered from 3' to 5' ? Inverting pseudotorsions, other angle measures are not corrected.")
            df = df.reindex(index=df.index[::-1]).reset_index(drop=True)
            df['index_chain'] = 1 + df.index
            temp_eta = df['eta']
            df['eta'] = [ df['theta'][n] for n in range(l) ]              # eta(n)    = theta(l-n+1) forall n in ]1, l] 
            df['theta'] = [ temp_eta[n] for n in range(l) ]               # theta(n)  = eta(l-n+1)   forall n in [1, l[ 
            temp_eta = df['eta_prime']
            df['eta_prime'] = [ df['theta_prime'][n] for n in range(l) ]  # eta(n)    = theta(l-n+1) forall n in ]1, l] 
            df['theta_prime'] = [ temp_eta[n] for n in range(l) ]         # theta(n)  = eta(l-n+1)   forall n in [1, l[ 
            temp_eta = df['eta_base']
            df['eta_base'] = [ df['theta_base'][n] for n in range(l) ]    # eta(n)    = theta(l-n+1) forall n in ]1, l] 
            df['theta_base'] = [ temp_eta[n] for n in range(l) ]          # theta(n)  = eta(l-n+1)   forall n in [1, l[ 
            newpairs = []
            for v in df['paired']:
                if ',' in v:
                    temp_v = []
                    vs = v.split(',')
                    for _ in vs:
                        temp_v.append(str(l-int(_)+1) if int(_) else _ )
                    newpairs.append(','.join(temp_v))
                else:
                    if len(v): 
                        newpairs.append(str(l-int(v)+1) if int(v) else v )
                    else: # means unpaired
                        newpairs.append(v)
            df['paired'] = newpairs

        # Saving to database
        sql_execute(conn, f"""
        INSERT OR REPLACE INTO nucleotide 
        (chain_id, index_chain, nt_resnum, nt_name, nt_code, dbn, alpha, beta, gamma, delta, epsilon, zeta,
        epsilon_zeta, bb_type, chi, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
        v0, v1, v2, v3, v4, amplitude, phase_angle, puckering, nt_align_code, is_A, is_C, is_G, is_U, is_other, nt_position, 
        paired, pair_type_LW, pair_type_DSSR, nb_interact)
        VALUES ({self.db_chain_id}, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ;""", many=True, data=list(df.to_records(index=False))
        )

        notify(f"Saved {self.chain_label} annotations to database.")
            
        # Now load data from the database
        self.seq = "".join(df.nt_code)
        self.seq_to_align = "".join(df.nt_align_code)
        self.length = len([ x for x in self.seq_to_align if x != "-" ])

        # Remove too short chains
        if self.length < 5:
            warn(f"{self.chain_label} sequence is too short, let's ignore it.\t", error=True)
            self.delete_me = True
            self.error_messages = "Sequence is too short. (< 5 resolved nts)"
        return 0

    def remap_and_save(self, conn, columns_to_save, s_seq):
        """Maps the object's sequence to its version in a MSA, to compute nucleotide frequencies at every position.
        
        conn: a connection to the database
        columns_to_save: a set of indexes in the alignment that are mapped to previous sequences in the alignment
        s_seq: the aligned version of self.seq_to_align
        This also replaces gaps by the most common nucleotide.
        """

        def register_col(i, j, unknown=False):
            if not unknown:
                # because index_chain in table nucleotide is in [1,N], we use i+1 and j+1.
                columns_to_save.add(j) # it's a set, doublons are automaticaly ignored
                sql_execute(conn, f"""INSERT OR REPLACE INTO re_mapping
                (chain_id, index_chain, index_ali) VALUES ({self.db_chain_id}, {i+1}, {j+1});
                """)
            else:
                # Index 0 is kept for an "unknown" values column
                sql_execute(conn, f"""INSERT OR REPLACE INTO re_mapping
                (chain_id, index_chain, index_ali) VALUES ({self.db_chain_id}, {i+1}, 0);""")

        alilen = len(s_seq)

        # Save colums in the appropriate positions
        i = 0
        j = 0
        while i<self.full_length and j<alilen:
            # Here we try to map self.seq_to_align (the sequence of the 3D chain, including gaps when residues are missing), 
            # with s_seq, the sequence aligned in the MSA, containing any of ACGU and two types of gaps, - and .

            if self.seq_to_align[i] == s_seq[j].upper(): # alignment and sequence correspond (incl. gaps)
                register_col(i, j)
                i += 1
                j += 1
            elif self.seq_to_align[i] == '-': # gap in the chain, but not in the aligned sequence

                # search for a gap to the consensus nearby
                k = 0
                while j+k<alilen and s_seq[j+k] in ['.','-']:
                    if s_seq[j+k] == '-':
                        break
                    k += 1

                # if found, set j to that position
                if j+k<alilen and s_seq[j+k] == '-':
                    j = j + k
                    continue

                # if not, search for a insertion gap nearby
                if j<alilen and s_seq[j] == '.':
                    register_col(i, j)
                    i += 1
                    j += 1
                    continue

                # else, just ignore the gap.
                register_col(i, j, unknown=True)
                i += 1
            elif s_seq[j] in ['.', '-']: # gap in the alignment, but not in the real chain
                j += 1 # ignore the column
            else: # sequence mismatch which is not a gap...
                print(f"You are never supposed to reach this. Comparing {self.chain_label} in {i} ({self.seq_to_align[i-1:i+2]}) with seq[{j}] ({s_seq[j-3:j+4]}).\n", 
                        self.seq_to_align, 
                        sep='', flush=True)
                raise Exception('Something is wrong with sequence alignment.')
        return columns_to_save

    def replace_gaps(self, conn):
        """ Replace gapped positions by the consensus sequence. """

        homology_data = sql_ask_database(conn, f"""SELECT freq_A, freq_C, freq_G, freq_U, freq_other FROM
                                                    (SELECT chain_id, rfam_acc FROM chain WHERE chain_id='{self.db_chain_id}')
                                                    NATURAL JOIN re_mapping
                                                    NATURAL JOIN align_column;
                                                """)
        assert len(homology_data) == self.full_length
        c_seq_to_align = list(self.seq_to_align)
        c_seq = list(self.seq)
        letters = ['A', 'C', 'G', 'U', 'N']
        for i in range(self.full_length):
            if c_seq_to_align[i] == '-':      # (then c_seq[i] also is)
                freq = homology_data[i]
                l = letters[freq.index(max(freq))]
                c_seq_to_align[i] = l
                c_seq[i] = l
                sql_execute(conn, f"""UPDATE nucleotide SET nt_align_code = '{l}', is_{l if l in "ACGU" else "other"} = 1
                                      WHERE chain_id = {self.db_chain_id} AND index_chain = {i+1};""")
        self.seq_to_align = ''.join(c_seq_to_align)
        self.seq = ''.join(c_seq)
        
    def save(self, conn, fformat = "csv"):
        # save to file
        df = pd.read_sql_query(f"""
        SELECT (index_chain, nt_resnum, position, nt_name, nt_code, nt_align_code, 
        is_A, is_C, is_G, is_U, is_other, freq_A, freq_C, freq_G, freq_U, freq_other, dbn,
        paired, nb_interact, pair_type_LW, pair_type_DSSR, alpha, beta, gamma, delta, epsilon, zeta, epsilon_zeta,
        chi, bb_type, glyco_bond, form, ssZp, Dp, eta, theta, eta_prime, theta_prime, eta_base, theta_base,
        v0, v1, v2, v3, v4, amlitude, phase_angle, puckering) FROM (
            (SELECT (chain_id, rfam_acc) from chain WHERE chain_id = {self.db_chain_id})
            NATURAL JOIN re_mapping
            NATURAL JOIN nucleotide
            NATURAL JOIN align_column
        )
        ;""", conn)
        if fformat == "csv":
            df.to_csv(path_to_3D_data + "datapoints/" + self.chain_label + str('.'+self.rfam_fam if self.rfam_fam != '' else ''))

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
            s = f"{self.priority_}({self.nthreads}) [{self.comp_time}]\t{self.label:25}{self.func_.__name__}(" + " ".join([str(a) for a in self.args_]) + ")"
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
        # Download PDB mappings to Rfam family
        print("> Fetching latest PDB mappings from Rfam..." + " " * 29, end='', flush=True)
        try:
            db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
            mappings = pd.read_sql('SELECT rfam_acc, pdb_id, chain, pdb_start, pdb_end, bit_score, evalue_score, cm_start, cm_end, hex_colour FROM pdb_full_region WHERE is_significant=1;', con=db_connection)
            mappings.to_csv(runDir + "/data/Rfam-PDB-mappings.csv")
            print(f"\t{validsymb}")
        except sqlalchemy.exc.OperationalError:  # Cannot connect :'(
            print(f"\t{errsymb}")
            # Check if a previous run succeeded (if file exists, use it)
            if path.isfile(runDir + "/data/Rfam-PDB-mappings.csv"):
                print("\t> Using previous version.")
                mappings = pd.read_csv(runDir + "/data/Rfam-PDB-mappings.csv")
            else: # otherwise, abort.
                print("Can't do anything without data. Exiting.")
                raise Exception("Can't reach mysql-rfam-public.ebi.ac.uk on port 4497. Is it open on your system ?")

        return mappings

    def download_Rfam_cm(self):
        """ Download the covariance models from Rfam.
        
        Does not download if already there.
        """

        print(f"\t> Download Rfam.cm.gz from Rfam..." + " " * 37, end='', flush=True) 
        if not path.isfile(path_to_seq_data + "Rfam.cm"):
            try:
                _urlcleanup()
                _urlretrieve(f'ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz', path_to_seq_data + "Rfam.cm.gz")
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
        """
        try: 
            db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
            conn = sqlite3.connect(runDir + "/results/RNANet.db", timeout=20.0)

            # Prepare the SQL query. It computes the length of the chains and gets the maximum length by family.
            q = """SELECT fr.rfam_acc, COUNT(DISTINCT fr.rfamseq_acc) AS 'n_seq',
                        MAX(
                            (CASE WHEN fr.seq_start > fr.seq_end THEN fr.seq_start
                                                                ELSE fr.seq_end
                            END)
                            -
                            (CASE WHEN fr.seq_start > fr.seq_end THEN fr.seq_end
                                                                ELSE fr.seq_start
                            END)
                        ) AS 'maxlength'
                    FROM full_region fr
                    GROUP BY fr.rfam_acc"""

            # Query the public database
            d = pd.read_sql(q, con=db_connection)

            # filter the results to families we are interested in
            d = d[ d["rfam_acc"].isin(list_of_families) ]

            # save the statistics to local database
            n_pdb = [ len(rfam_acc_to_download[f]) for f in d["rfam_acc"] ]
            d["n_pdb_seqs"] = n_pdb
            d["total_seqs"] = d["n_seq"] + d["n_pdb_seqs"]
            sql_execute(conn, """
                INSERT OR REPLACE INTO family (rfam_acc, nb_homologs, max_len, nb_3d_chains, nb_total_homol)
                VALUES (?, ?, ?, ?, ?);""", many=True, data=list(d.to_records(index=False))
            ) # We use the replace keyword to get the latest information
            conn.close()
            
            # print the stats
            for _, l in d.iterrows():
                print(f"\t> {l['rfam_acc']}: {l['n_seq']} Rfam hits + {l['n_pdb_seqs']} PDB sequences to realign")

        except sqlalchemy.exc.OperationalError:
            warn("Something's wrong with the SQL database. Check mysql-rfam-public.ebi.ac.uk status and try again later. Not printing statistics.")

    def download_Rfam_sequences(self, rfam_acc):
        """ Downloads the unaligned sequences known related to a given RNA family.

        Actually gets a FASTA archive from the public Rfam FTP. Does not download if already there."""

        if not path.isfile(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz"):
            try:
                _urlcleanup()
                _urlretrieve(   f'ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/{rfam_acc}.fa.gz',
                                path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz")
                notify(f"Downloaded {rfam_acc}.fa.gz from Rfam")
            except:
                warn(f"Error downloading {rfam_acc}.fa.gz. Does it exist ?\t", error=True)
        else:
            notify(f"Downloaded {rfam_acc}.fa.gz from Rfam", "(already there)")

    def download_BGSU_NR_list(self):
        """ Downloads a list of RNA 3D structures proposed by Bowling Green State University RNA research group.
        The chosen list is the one with resolution threshold just above the desired one.

        Does not remove structural redundancy.
        """
        nr_code = min([ i for i in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 20.0] if i >= CRYSTAL_RES ]) 
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
            if path.isfile(path_to_3D_data + f"latest_nr_list_{nr_code}A.csv"):
                print("\t> Use of the previous version.\t", end = "", flush=True)
            else:
                return [], []

        nrlist = pd.read_csv(path_to_3D_data + f"latest_nr_list_{nr_code}A.csv")
        full_structures_list = nrlist['class_members'].tolist()
        print(f"\t{validsymb}", flush=True)

        # The beginning of an adventure.
        return full_structures_list

    def download_from_SILVA(self, unit):
        if not path.isfile(path_to_seq_data + f"realigned/{unit}.arb"):
            try:
                _urlcleanup()
                if unit=="LSU":
                    _urlretrieve('http://www.arb-silva.de/fileadmin/arb_web_db/release_132/ARB_files/SILVA_132_LSURef_07_12_17_opt.arb.gz',
                                  path_to_seq_data + "realigned/LSU.arb.gz")
                else:
                    _urlretrieve('http://www.arb-silva.de/fileadmin/silva_databases/release_138/ARB_files/SILVA_138_SSURef_05_01_20_opt.arb.gz', 
                                  path_to_seq_data + "realigned/SSU.arb.gz")
            except:
                warn(f"Error downloading the {unit} database from SILVA", error=True)
                exit(1)
            subprocess.run(["gunzip", path_to_seq_data + f"realigned/{unit}.arb.gz"], stdout=subprocess.DEVNULL)
            notify(f"Downloaded and extracted {unit} database from SILVA")
        else:
            notify(f"Downloaded and extracted {unit} database from SILVA", "(used previous file)")

def read_cpu_number():
    # As one shall not use os.cpu_count() on LXC containers,
    # because it reads info from /sys wich is not the VM resources but the host resources.
    # This function reads it from /proc/cpuinfo instead.
    p = subprocess.run(['grep', '-Ec', '(Intel|AMD)', '/proc/cpuinfo'], stdout=subprocess.PIPE)
    return int(int(p.stdout.decode('utf-8')[:-1])/2)

def init_worker(tqdm_lock=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)

def trace_unhandled_exceptions(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            s = traceback.format_exc()
            with open(runDir +  "/errors.txt", "a") as f:
                f.write("Exception in "+func.__name__+"\n")
                f.write(s)
                f.write("\n\n")

            warn('Exception in '+func.__name__, error=True)
            print(s)
    return wrapped_func

def warn(message, error=False):
    """Pretty-print warnings and error messages.
    """
    # Cut if too long
    if len(message)>66:
        x = message.find(' ', 50, 66)
        warn(message[:x], error=error)
        warn(message[x+1:], error=error)
        return

    if error:
        print(f"\t> \033[31mERR: {message:65s}\033[0m\t{errsymb}", flush=True)
    else:
        print(f"\t> \033[33mWARN: {message:64s}\033[0m\t{warnsymb}", flush=True)

def notify(message, post=''):
    if len(post):
        post = '(' + post + ')'
    print(f"\t> {message:70s}\t{validsymb}\t{post}", flush=True)

@trace_unhandled_exceptions
def sql_define_tables(conn):
    conn.executescript(
        """ CREATE TABLE IF NOT EXISTS structure (
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
                pdb_start       SMALLINT,
                pdb_end         SMALLINT,
                reversed        TINYINT,
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
                UNIQUE (structure_id, chain_name, rfam_acc)
            );
            CREATE TABLE IF NOT EXISTS nucleotide (
                nt_id           INTEGER PRIMARY KEY NOT NULL,
                chain_id        INT,
                index_chain     SMALLINT,
                nt_resnum       SMALLINT,
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
                UNIQUE (chain_id, index_chain)
            );
            CREATE TABLE IF NOT EXISTS re_mapping (
                remapping_id    INTEGER PRIMARY KEY NOT NULL,
                chain_id        INT NOT NULL,
                index_chain     INT NOT NULL,
                index_ali       INT NOT NULL,
                UNIQUE (chain_id, index_chain),
                UNIQUE (chain_id, index_ali)
            );
            CREATE TABLE IF NOT EXISTS family (
                rfam_acc        CHAR(7) PRIMARY KEY NOT NULL,
                nb_homologs     INT,
                nb_3d_chains    INT,
                nb_total_homol  INT,
                max_len         UNSIGNED SMALLINT,
                comput_time     REAL,
                comput_peak_mem REAL,
                idty_percent    REAL
            );
            CREATE TABLE IF NOT EXISTS align_column (
                column_id       INTEGER PRIMARY KEY NOT NULL,
                rfam_acc        CHAR(7) NOT NULL,
                index_ali       INT NOT NULL,
                freq_A          REAL,
                freq_C          REAL,
                freq_G          REAL,
                freq_U          REAL,
                freq_other      REAL,
                UNIQUE (rfam_acc, index_ali)
            );
         """)

@trace_unhandled_exceptions
def sql_ask_database(conn, sql):
    """
    Reads the SQLite database.
    Returns a list of tuples.
    """
    cursor = conn.cursor()
    result = cursor.execute(sql).fetchall()
    cursor.close()
    return result

@trace_unhandled_exceptions
def sql_execute(conn, sql, many=False, data=None):
    for _ in range(100): # retry 100 times if it fails
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
            warn(str(e) + ", retrying in 0.2s (worker " + str(os.getpid()) + f', try {_+1}/100)')
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

    if len(j.cmd_): # The job is a system command

        print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.label}")

        # Add the command to logfile
        logfile = open(runDir + "/log_of_the_run.sh", 'a')
        logfile.write(" ".join(j.cmd_))
        logfile.write("\n")
        logfile.close()

        # Run it
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # put the monitor in a different thread
            assistant_future = executor.submit(monitor.check_mem_usage)
            
            # run the command. subprocess.run will be a child of this process, and stays monitored.
            start_time = time.time()
            r = subprocess.run(j.cmd_, timeout=j.timeout_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end_time = time.time()

            # Stop the Monitor, then get its result
            monitor.keep_watching = False
            m = assistant_future.result()

    elif j.func_ is not None:

        print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.func_.__name__}({', '.join([str(a) for a in j.args_ if not ((type(a) == list) and len(a)>3)])})")

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
    return (t,m,r)

def execute_joblist(fulljoblist):
    """ Run a list of job objects.

    The jobs in the list can have differente priorities and/or different number of threads.
    """

    # Reset counters
    running_stats[0] = 0       # started
    running_stats[1] = 0       # finished
    running_stats[2] = 0       # failed

    # Sort jobs in a tree structure, first by priority, then by CPU numbers
    jobs = {}
    jobcount = len(fulljoblist)
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
    for i in range(1,nprio+1):
        if i not in jobs.keys(): continue # no job has the priority level i

        print("processing jobs of priority", i)
        different_thread_numbers = sorted(jobs[i].keys())
        
        # jobs should be processed 1 by 1, 2 by 2, or n by n depending on their definition
        for n in different_thread_numbers:
            # get the bunch of jobs of same priority and thread number
            bunch = jobs[i][n]
            if not len(bunch): continue # no jobs should be processed n by n

            print("using", n, "processes:")
            # execute jobs of priority i that should be processed n by n:
            p = Pool(processes=n, maxtasksperchild=5, initializer=init_worker)
            try:
                raw_results = p.map(partial(execute_job, jobcount=jobcount), bunch)
                p.close()
                p.join()
            except KeyboardInterrupt as e:
                warn("KeyboardInterrupt, terminating workers.", error=True)
                p.terminate()
                p.join()
                raise e

            for j, r in zip(bunch, raw_results):
                j.comp_time = round(r[0], 2) # seconds
                j.max_mem = int(r[1]/1000000) # MB
                results.append( (j.label, r[2], round(r[0], 2), int(r[1]/1000000)))
    
    # throw back the money
    return results

@trace_unhandled_exceptions
def work_infer_mappings(allmappings, codelist):
    """Given a list of PDB chains corresponding to an equivalence class from BGSU's NR list, 
    build a list of Chain() objects mapped to Rfam families, by expanding available mappings 
    of any element of the list to all the list elements.
    """
    newchains = []
    known_mappings = pd.DataFrame()

    # Split the comma-separated list of chain codes into chain codes:
    codes = str(codelist).replace('+',',').split(',')

    # Search for mappings that apply to an element of this PDB chains list:
    for c in codes:
        # search for Rfam mappings with this chain c:
        m_row_indices = allmappings.pdb_id + "|1|" + allmappings.chain == c[:4].lower()+c[4:]
        m = allmappings.loc[m_row_indices].drop(['bit_score','evalue_score','cm_start','cm_end','hex_colour'], axis=1)
        if len(m):
            # remove the found mappings from the dataframe
            allmappings = allmappings.loc[m_row_indices == False]
            # Add the found mappings to the list of found mappings for this class of equivalence
            known_mappings = pd.concat([known_mappings, m])
    
    # Now infer mappings for chains that are not explicitely listed in Rfam-PDB mappings:
    if len(known_mappings):
        families = set(known_mappings['rfam_acc'])

        # generalize
        inferred_mappings = known_mappings.drop(['pdb_id','chain'], axis=1).drop_duplicates()
        
        # check for approximative redundancy:
        if len(inferred_mappings) != len(inferred_mappings.drop_duplicates(subset="rfam_acc")):
            # Then, there exists some mapping variants onto the same Rfam family CM,
            # but varing in the start/end positions in the chain. 
            # ==> Summarize them in one mapping but with the largest window.
            for rfam in families:
                sel_5_to_3 = (inferred_mappings['pdb_start'] < inferred_mappings['pdb_end'])
                thisfam_5_3 =  (inferred_mappings['rfam_acc'] == rfam ) & sel_5_to_3
                thisfam_3_5 =  (inferred_mappings['rfam_acc'] == rfam ) & (sel_5_to_3 == False)

                if (
                        len(inferred_mappings[thisfam_5_3]) !=  len(inferred_mappings[ inferred_mappings['rfam_acc'] == rfam ])
                    and len(inferred_mappings[thisfam_5_3]) > 0
                ):
                    warn(f"There are mappings for {rfam} in both directions:", error=True)
                    print(inferred_mappings)
                    exit(1)

                # Compute consensus for chains in 5' -> 3' sense
                if len(inferred_mappings[thisfam_5_3]):
                    pdb_start_min = min(inferred_mappings[ thisfam_5_3]['pdb_start'])
                    pdb_end_max = max(inferred_mappings[ thisfam_5_3]['pdb_end']) 
                    pdb_start_max = max(inferred_mappings[ thisfam_5_3]['pdb_start'])
                    pdb_end_min = min(inferred_mappings[ thisfam_5_3]['pdb_end'])
                    if (pdb_start_max - pdb_start_min < 100) and (pdb_end_max - pdb_end_min < 100):
                        # the variation is only a few nucleotides, we take the largest window.
                        inferred_mappings.loc[ thisfam_5_3, 'pdb_start'] = pdb_start_min
                        inferred_mappings.loc[ thisfam_5_3, 'pdb_end'] = pdb_end_max
                    else:
                        # there probably is an outlier. We chose the median value in the whole list of known_mappings.
                        known_sel_5_to_3 = (known_mappings['rfam_acc'] == rfam ) & (known_mappings['pdb_start'] < known_mappings['pdb_end'])
                        inferred_mappings.loc[ thisfam_5_3, 'pdb_start'] = known_mappings.loc[known_sel_5_to_3, 'pdb_start'].median()
                        inferred_mappings.loc[ thisfam_5_3, 'pdb_end'] = known_mappings.loc[known_sel_5_to_3, 'pdb_end'].median()

                #  Compute consensus for chains in 3' -> 5' sense
                if len(inferred_mappings[thisfam_3_5]):
                    pdb_start_min = min(inferred_mappings[ thisfam_3_5]['pdb_start'])
                    pdb_end_max = max(inferred_mappings[ thisfam_3_5]['pdb_end']) 
                    pdb_start_max = max(inferred_mappings[ thisfam_3_5]['pdb_start'])
                    pdb_end_min = min(inferred_mappings[ thisfam_3_5]['pdb_end'])
                    if (pdb_start_max - pdb_start_min < 100) and (pdb_end_max - pdb_end_min < 100):
                        # the variation is only a few nucleotides, we take the largest window.
                        inferred_mappings.loc[ thisfam_3_5, 'pdb_start'] = pdb_start_max
                        inferred_mappings.loc[ thisfam_3_5, 'pdb_end'] = pdb_end_min
                    else:
                        # there probably is an outlier. We chose the median value in the whole list of known_mappings.
                        known_sel_3_to_5 = (known_mappings['rfam_acc'] == rfam ) & (known_mappings['pdb_start'] > known_mappings['pdb_end'])
                        inferred_mappings.loc[ thisfam_3_5, 'pdb_start'] = known_mappings.loc[known_sel_3_to_5, 'pdb_start'].median()
                        inferred_mappings.loc[ thisfam_3_5, 'pdb_end'] = known_mappings.loc[known_sel_3_to_5, 'pdb_end'].median()
            inferred_mappings.drop_duplicates(inplace=True)

        # Now build Chain() objects for the mapped chains
        for c in codes:
            nr = c.split('|')
            pdb_id = nr[0].lower()
            pdb_model = int(nr[1])
            pdb_chain_id = nr[2]
            for rfam in families:

                # if a known mapping of this chain on this family exists, apply it
                m = known_mappings.loc[ (known_mappings.pdb_id + "|1|" + known_mappings.chain == c[:4].lower()+c[4:]) & (known_mappings['rfam_acc'] == rfam ) ]
                if len(m):
                    pdb_start = int(m.pdb_start)
                    pdb_end = int(m.pdb_end)
                    inferred = False
                else: # otherwise, use the inferred mapping
                    pdb_start = int(inferred_mappings.loc[ (inferred_mappings['rfam_acc'] == rfam) ].pdb_start)
                    pdb_end = int(inferred_mappings.loc[ (inferred_mappings['rfam_acc'] == rfam) ].pdb_end)
                    inferred = True
                chain_label = f"{pdb_id}_{str(pdb_model)}_{pdb_chain_id}_{pdb_start}-{pdb_end}"

                # Check if the chain exists in the database
                # with sqlite3.connect(runDir+"/results/RNANet.db", timeout=10.0) as conn:
                #     res = sql_ask_database(conn, f"""SELECT chain_id from chain WHERE structure_id='{pdb_id}' AND chain_name='{pdb_chain_id}' AND rfam_acc='{rfam}' AND issue=0""")
                # if not len(res): # the chain is NOT yet in the database, or this is a known issue
                #     newchains.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label, rfam=rfam, inferred=inferred, pdb_start=pdb_start, pdb_end=pdb_end))
                newchains.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label, rfam=rfam, inferred=inferred, pdb_start=pdb_start, pdb_end=pdb_end))
    
    return newchains

@trace_unhandled_exceptions
def work_mmcif(pdb_id):
    """ Look for a CIF file (with all chains) from RCSB
    """

    url = 'http://files.rcsb.org/download/%s.cif' % (pdb_id)
    final_filepath = path_to_3D_data+"RNAcifs/"+pdb_id+".cif"

    # Attempt to download it
    try:
        _urlcleanup()
        _urlretrieve(url, final_filepath)
    except:
        warn(f"Unable to download {pdb_id}.cif. Ignoring it.", error=True)
        return

    # Load the MMCIF file with Biopython 
    mmCif_info = MMCIF2Dict(final_filepath)

    # Get info about that structure
    exp_meth = mmCif_info["_exptl.method"][0]
    date = mmCif_info["_pdbx_database_status.recvd_initial_deposition_date"][0]
    if "_refine.ls_d_res_high" in mmCif_info.keys() and mmCif_info["_refine.ls_d_res_high"][0] not in ['.', '?']:
        reso = float(mmCif_info["_refine.ls_d_res_high"][0])
    elif "_refine.ls_d_res_low" in mmCif_info.keys() and mmCif_info["_refine.ls_d_res_low"][0] not in ['.', '?']:
        reso = float(mmCif_info["_refine.ls_d_res_low"][0])
    elif "_em_3d_reconstruction.resolution" in mmCif_info.keys() and mmCif_info["_em_3d_reconstruction.resolution"][0] not in ['.', '?']:
        reso = float(mmCif_info["_em_3d_reconstruction.resolution"][0])
    else:
        warn(f"Wtf, structure {pdb_id} has no resolution ? Check https://files.rcsb.org/header/{pdb_id}.cif to figure it out.")
        reso = 0.0
    
    # Save into the database
    with sqlite3.connect(runDir + "/results/RNANet.db")as conn:
        sql_execute(conn, """INSERT OR REPLACE INTO structure (pdb_id, pdb_model, date, exp_method, resolution)
                             VALUES (?, ?, DATE(?), ?, ?);""", data = (pdb_id, 1, date, exp_meth, reso))
    
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
    json_file = open(path_to_3D_data + "annotations/" + pdb_id + ".json", "w")
    json_file.write(stdout)
    json_file.close()
    return 0

@trace_unhandled_exceptions
def work_build_chain(c, retrying=False):
    """ Additionally adds all the desired information to a Chain object.

    """

    if not path.isfile(path_to_3D_data + "annotations/" + c.pdb_id + ".json"):
        warn(f"Could not find annotations for {c.chain_label}, ignoring it.", error=True)
        c.delete_me = True
        c.error_messages += f"Could not download and/or find annotations for {c.chain_label}."

    # Register the chain, and get its chain_id
    conn = sqlite3.connect(runDir+"/results/RNANet.db", timeout=10.0)
    if c.pdb_start is not None:
        sql_execute(conn, f"""  INSERT OR REPLACE INTO chain 
                                (structure_id, chain_name, pdb_start, pdb_end, reversed, rfam_acc, inferred, issue)
                                VALUES 
                                (?, ?, ?, ?, ?, ?, ?, ?);""", 
                                data=(c.pdb_id, c.pdb_chain_id, c.pdb_start, c.pdb_end, int(c.reversed), c.rfam_fam, int(c.inferred), int(c.delete_me)))
    else:
        sql_execute(conn, "INSERT OR REPLACE INTO chain (structure_id, chain_name, issue) VALUES (?, ?, ?);", data=(c.pdb_id, c.pdb_chain_id, int(c.delete_me)))
    c.db_chain_id = sql_ask_database(conn, f"SELECT (chain_id) FROM chain WHERE structure_id='{c.pdb_id}' AND chain_name='{c.pdb_chain_id}';")[0][0]

    # extract the portion we want
    if EXTRACT_CHAINS and not c.delete_me:
        c.extract()

    # extract the 3D descriptors
    if not c.delete_me:
        c.extract_3D_data(conn)

    # If there were newly discovered problems, add this chain to the known issues
    if c.delete_me and c.chain_label not in known_issues and not (not retrying and "Could not load existing" in c.error_messages):
        known_issues.append(c.chain_label)
        warn(f"Adding {c.chain_label} to known issues.")
        f = open(runDir + "/known_issues.txt", 'a')
        f.write(c.chain_label + '\n')
        f.close()
        f = open(runDir + "/known_issues_reasons.txt", 'a')
        f.write(c.chain_label + '\n' + c.error_messages + '\n\n')
        f.close()
    # Register the issues in the database
    if c.delete_me and c.chain_label not in known_issues:
        sql_execute(conn, f"UPDATE chain SET issue = 1 WHERE chain_id = ?;", data=(c.db_chain_id,))
    
    conn.close()
    
    # The Chain object is ready
    return c

@trace_unhandled_exceptions
def work_prepare_sequences(rfam_acc):
    """Prepares FASTA files of homologous sequences to realign with cmalign.
    Prepares databases of SSUs and LSUs to realign with SINA."""

    if rfam_acc not in ["RF00177", "RF01960", "RF00002", "RF02540", "RF02541", "RF02543"]: # Ribosomal Subunits
        
        # Extracting sequences for this family
        if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.fa"):
            # Download homologous sequences
            dl.download_Rfam_sequences(rfam_acc)

            # # Prepare a FASTA file containing Rfamseq hits for that family + our chains sequences
            # f = open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "w")

            # # Read the FASTA archive of Rfamseq hits, and add sequences to the file
            # with gzip.open(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz", 'rt') as gz:
            #     ids = []
            #     for record in SeqIO.parse(gz, "fasta"):
            #         if record.id not in ids:
            #             # Note: here we copy the sequences without modification. 
            #             # But, sequences with non ACGU letters exit (W, R, M, Y for example)
            #             f.write(">"+record.description+'\n'+str(record.seq)+'\n')
            #             ids.append(record.id)

            # gunzip the file
            with gzip.open(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz", 'rb') as gz:
                file_content = gz.read()
            with open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "wb") as plusplus:
                plusplus.write(file_content)
            del file_content

            notify(f"Prepared FASTA file of {rfam_acc} homologous sequences")
        
        # Extracting covariance model for this family
        if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}.cm"):
            f = open(path_to_seq_data + f"realigned/{rfam_acc}.cm", "w")
            subprocess.run(["cmfetch", path_to_seq_data + "Rfam.cm", rfam_acc], stdout=f)
            f.close()
            notify(f"Extracted {rfam_acc} covariance model (cmfetch)")

@trace_unhandled_exceptions
def work_realign(rfam_acc, chains):
    """ Runs multiple sequence alignements by RNA family.

    It aligns the Rfam hits from a RNA family with the sequences from the list of chains. 
    Rfam covariance models are used with Infernal tools, except for rRNAs. 
    cmalign requires too much RAM for them, so we use SINA, a specifically designed tool for rRNAs.
    """

    # Add the new chains' sequences to the list of homologs
    f = open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "a")
    for c in chains:
        f.write(f"> {str(c)}\n"+c.seq_to_align.replace('-', '').replace('U','T')+'\n') 
    f.close()
    notify(f"Added new 3D chains sequences to {rfam_acc}++.fa")

    # remove eventual previous files
    if path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.stk"):
        os.remove(path_to_seq_data + f"realigned/{rfam_acc}++.stk")
    if path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.afa"):
        os.remove(path_to_seq_data + f"realigned/{rfam_acc}++.afa")

    if rfam_acc not in ["RF00177", "RF01960", "RF00002", "RF02540", "RF02541", "RF02543"]: # Ribosomal Subunits
        # Align using Infernal for most RNA families
        print(f"\t> Re-aligning {rfam_acc} with {len(chains)} chains (cmalign)...", flush=True)
        f = open(path_to_seq_data + f"realigned/{rfam_acc}++.stk", "w")
        subprocess.run(["cmalign", "--mxsize", "2048", path_to_seq_data + f"realigned/{rfam_acc}.cm", path_to_seq_data + f"realigned/{rfam_acc}++.fa"], stdout=f)
        f.close()

        # Converting to aligned Fasta
        print("\t> Converting to aligned FASTA (esl-reformat)...")
        f = open(path_to_seq_data + f"realigned/{rfam_acc}++.afa", "w")
        subprocess.run(["esl-reformat", "afa", path_to_seq_data + f"realigned/{rfam_acc}++.stk"], stdout=f)
        f.close()
    else:
        # Ribosomal subunits deserve a special treatment.
        # They require too much RAM to be aligned with Infernal.
        # Then we will use SINA instead.
        if rfam_acc in ["RF00177", "RF01960"]:
            arbfile = "realigned/SSU.arb"
        else:
            arbfile = "realigned/LSU.arb"

        # Run alignment
        print(f"\t> Re-aligning {rfam_acc} with {len(chains)} chains (SINA)...", flush=True)
        subprocess.run(["sina", "-i", path_to_seq_data + f"realigned/{rfam_acc}++.fa",
                               "-o", path_to_seq_data + f"realigned/{rfam_acc}++.afa",
                               "-r", path_to_seq_data + arbfile,
                               "--meta-fmt=csv"])
    return 0

def summarize_position(counts):
    """ Counts the number of nucleotides at a given position, given a "column" from a MSA.
    """

    # Count modified nucleotides
    chars = counts.keys()
    known_chars_count = 0
    N = 0
    for char in chars:
        if char in "ACGU":
            known_chars_count += counts[char]
        if char not in ".-":
            N += counts[char]  # number of ungapped residues

    if N: # prevent division by zero if the column is only gaps
        return ( counts['A']/N, counts['C']/N, counts['G']/N, counts['U']/N, (N - known_chars_count)/N) # other residues, or consensus (N, K, Y...)
    else:
        return (0, 0, 0, 0, 0)

@trace_unhandled_exceptions
def work_pssm(f):
    """ Computes Position-Specific-Scoring-Matrices given the multiple sequence alignment of the RNA family.
    
    Also saves every chain of the family to file.
    Uses only 1 core, so this function can be called in parallel.
    """
    conn = sqlite3.connect(runDir + '/results/RNANet.db', timeout=20.0)

    # Get a worker number to position the progress bar
    global idxQueue
    thr_idx = idxQueue.get()

    # get the chains of this family
    list_of_chains =  rfam_acc_to_download[f]
    chains_ids = [ str(c) for c in list_of_chains ]

    # Open the alignment
    try:
        align = AlignIO.read(path_to_seq_data + f"realigned/{f}++.afa", "fasta")
    except:
        warn(f"{f}'s alignment is wrong. Recompute it and retry.", error=True)
        return 1
        

    # Compute statistics per column
    pssm = BufferingSummaryInfo(align).get_pssm(f, thr_idx)
    frequencies = [ summarize_position(pssm[i]) for i in range(align.get_alignment_length()) ]
    del pssm

    # For each sequence, find the right chain and remap chain residues with alignment columns
    columns_to_save = set()
    pbar = tqdm(total=len(chains_ids), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} chains", leave=False)
    pbar.update(0)
    for s in align:
        if not '[' in s.id: # this is a Rfamseq entry, not a 3D chain
            continue
        # get the right 3D chain:
        idx = chains_ids.index(s.id)
        # call its remap method 
        columns_to_save = list_of_chains[idx].remap_and_save(conn, columns_to_save, s.seq)

        pbar.update(1)
    pbar.close()

    # Save the useful columns in the database
    data = [ (f, j+1) + frequencies[j] for j in sorted(columns_to_save) ]
    sql_execute(conn, """INSERT OR REPLACE INTO align_column
    (rfam_acc, index_ali, freq_A, freq_C, freq_G, freq_U, freq_other)
    VALUES (?, ?, ?, ?, ?, ?, ?);""", many=True, data=data)
    # Add an unknown values column, with index_ali 0
    sql_execute(conn, f"""INSERT OR REPLACE INTO align_column
    (rfam_acc, index_ali, freq_A, freq_C, freq_G, freq_U, freq_other)
    VALUES (?, 0, 0.0, 0.0, 0.0, 0.0, 1.0);""", data=(f,))

    # Save chains to CSV
    for s in align:
        if not '[' in s.id: # this is a Rfamseq entry, not a 3D chain
            continue
        idx = chains_ids.index(s.id)

        # Replace gaps by consensus sequence, if asked
        list_of_chains[idx].replace_gaps(conn)

        # Save to CSV
        list_of_chains[idx].save(conn)

    conn.close()
    del rfam_acc_to_download[f] # We won't need this family's chain objects anymore, free up
    idxQueue.put(thr_idx) # replace the thread index in the queue
    return 0

if __name__ == "__main__":

    runDir = path.dirname(path.realpath(__file__))
    ncores = read_cpu_number()

    # Parse options
    try:
        opts, args = getopt.getopt( sys.argv[1:], "r:hs", 
                                [   "help", "resolution=", "keep-hetatm=", "from-scratch",
                                    "fill-gaps=", "3d-folder=", "seq-folder=",
                                    "no-homology", "ignore-issues", "extract",
                                    "update-mmcifs", "update-homologous" ])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:

        if opt in ["--from-scratch", "--update-mmcifs", "--update-homolgous"] and "tobedefinedbyoptions" in [path_to_3D_data, path_to_seq_data]:
            print("Please provide --3d-folder and --seq-folder first, so that we know what to delete and update.")
            exit()

        if opt == "-h" or opt == "--help":
            print(  "RNANet, a script to build a multiscale RNA dataset from public data\n"
                    "Developped by Louis Becquey (louis.becquey@univ-evry.fr), 2020")
            print()
            print("Options:")
            print("-h [ --help ]\t\t\tPrint this help message")
            print("--version\t\t\tPrint the program version")
            print()
            print("-r 4.0 [ --resolution=4.0 ]\tMaximum 3D structure resolution to consider a RNA chain.")
            print("-s\t\t\t\tRun statistics computations after completion")
            print("--extract\t\t\tExtract the portions of 3D RNA chains to individual mmCIF files.")
            print("--keep-hetatm=False\t\t(True | False) Keep ions, waters and ligands in produced mmCIF files. "
                    "\n\t\t\t\tDoes not affect the descriptors.")
            print("--fill-gaps=True\t\t(True | False) Replace gaps in nt_align_code field due to unresolved residues"
                    "\n\t\t\t\tby the most common nucleotide at this position in the alignment.")
            print("--3d-folder=…\t\t\tPath to a folder to store the 3D data files. Subfolders will contain:"
                    "\n\t\t\t\t\tRNAcifs/\t\tFull structures containing RNA, in mmCIF format"
                    "\n\t\t\t\t\trna_mapped_to_Rfam/\tExtracted 'pure' RNA chains"
                    "\n\t\t\t\t\tdatapoints/\t\tFinal results in CSV file format.")
            print("--seq-folder=…\t\t\tPath to a folder to store the sequence and alignment files."
                    "\n\t\t\t\t\trfam_sequences/fasta/\tCompressed hits to Rfam families"
                    "\n\t\t\t\t\trealigned/\t\tSequences, covariance models, and alignments by family")
            print("--no-homology\t\t\tDo not try to compute PSSMs and do not align sequences."
                    "\n\t\t\t\tAllows to yield more 3D data (consider chains without a Rfam mapping).")
            print()
            print("--ignore-issues\t\t\tDo not ignore already known issues and attempt to compute them")
            print("--update-mmcifs\t\t\tRe-download and Re-annotate mmCIF files")
            print("--update-homologous\t\tRe-download Rfam sequences and SILVA arb databases, and realign all families")
            print("--from-scratch\t\t\tDelete database, local 3D and sequence files, and known issues, and recompute.")
            print()
            print("Typical usage:")
            print(f"nohup bash -c 'time {runDir}/RNAnet.py --3d-folder ~/Data/RNA/3D/ --seq-folder /Data/RNA/sequences' &") 
            sys.exit()

        elif opt == '--version':
            print("RNANet 1.0 alpha ")
            sys.exit()
        elif opt == "-r" or opt == "--resolution":
            assert float(arg) > 0.0 and float(arg) < 20.0 
            CRYSTAL_RES = float(arg)
        elif opt == "-s":
            RUN_STATS = True
        elif opt=="--keep-hetatm":
            assert arg in [ "True", "False" ]
            KEEP_HETATM = (arg == "True")
        elif opt=="--fill-gaps":
            assert arg in [ "True", "False" ]
            FILL_GAPS = (arg == "True")
        elif opt=="--no-homology":
            HOMOLOGY = False
        elif opt=='--3d-folder':
            path_to_3D_data = path.abspath(arg)
            if path_to_3D_data[-1] != '/':
                path_to_3D_data += '/'
            print("> Storing 3D data into", path_to_3D_data)
        elif opt=='--seq-folder':
            path_to_seq_data = path.abspath(arg)
            if path_to_seq_data[-1] != '/':
                path_to_seq_data += '/'
            print("> Storing sequences into", path_to_seq_data)
        elif opt == "--ignore-issues":
            USE_KNOWN_ISSUES = False
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
        elif opt == "--update-mmcifs":
            warn("Deleting previous 3D files and redownloading and recomputing them")
            subprocess.run(["rm", "-rf", 
                            path_to_3D_data + "annotations",
                            path_to_3D_data + "RNAcifs",
                            path_to_3D_data + "rna_mapped_to_Rfam",
                            path_to_3D_data + "rnaonly"])
        elif opt == "--update-homologous":
            warn("Deleting previous sequence files and recomputing alignments.")
            subprocess.run(["rm", "-rf", 
                            path_to_seq_data + "realigned",
                            path_to_seq_data + "rfam_sequences"])
        elif opt == "--extract":
            EXTRACT_CHAINS = True
    
    if "tobedefinedbyoptions" in [path_to_3D_data, path_to_seq_data]:
        print("usage: RNANet.py --3d-folder path/where/to/store/chains --seq-folder path/where/to/store/alignments")
        print("See RNANet.py --help for more information.")
        exit(1)

    os.makedirs(runDir + "/results", exist_ok=True)
    os.makedirs(runDir + "/data", exist_ok=True)
    subprocess.run(["rm", "-f", runDir+"/errors.txt"])

    # Check existence of the database, or create it
    conn = sqlite3.connect(runDir + '/results/RNANet.db')
    sql_define_tables(conn)
    conn.commit()
    conn.close()
    print("> Storing results into", runDir + "/results/RNANet.db")

    # ===========================================================================
    # List 3D chains with available Rfam mapping
    # ===========================================================================

    # List all 3D RNA chains below given resolution
    dl = Downloader()
    full_structures_list = dl.download_BGSU_NR_list()

    # Check for a list of known problems:
    known_issues = []
    if path.isfile(runDir + "/known_issues.txt"):
        f = open(runDir + "/known_issues.txt", 'r')
        known_issues = [ x[:-1] for x in f.readlines() ]
        f.close()
        if USE_KNOWN_ISSUES:
            print("\t> Ignoring known issues:")
            for x in known_issues:
                print("\t  ", x)

    update = []
    if HOMOLOGY:
        # Ask Rfam if some are mapped to Rfam families
        allmappings = dl.download_Rfam_PDB_mappings()

        # Compute the list of mappable structures using NR-list and Rfam-PDB mappings
        # And get Chain() objects
        print("> Building list of structures...", flush=True)
        p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=ncores)
        try:
            pbar = tqdm(full_structures_list, maxinterval=1.0, miniters=1, bar_format="{percentage:3.0f}%|{bar}|")
            for i, newchains in enumerate(p.imap_unordered(partial(work_infer_mappings, allmappings), full_structures_list)): 
                update += newchains
                pbar.update(1) # Everytime the iteration finishes, update the global progress bar

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
        for codelist in tqdm(full_structures_list):
            codes = str(codelist).replace('+',',').split(',')

            # Simply convert the list of codes to Chain() objects
            for c in codes:
                nr = c.split('|')
                pdb_id = nr[0].lower()
                pdb_model = int(nr[1])
                pdb_chain_id = nr[2].upper()
                chain_label = f"{pdb_id}_{str(pdb_model)}_{pdb_chain_id}"
                res = sql_ask_database(conn, f"""SELECT chain_id from chain WHERE structure_id='{pdb_id}' AND chain_name='{pdb_chain_id}' AND rfam_acc IS NULL AND issue=0""")
                if not len(res): # the chain is NOT yet in the database, or this is a known issue
                    update.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label))
        conn.close()

    del full_structures_list
    
    n_chains = len(update)
    print(str(n_chains) + " RNA chains of interest.")

    # ===========================================================================
    # Download 3D structures, and annotate them
    # ===========================================================================
    
    # # Prepare the results folders
    # if not path.isdir(path_to_3D_data + "RNAcifs"):
    #     os.makedirs(path_to_3D_data + "RNAcifs")        # for the whole structures
    # if not path.isdir(path_to_3D_data + "annotations"):
    #     os.makedirs(path_to_3D_data + "annotations")    # for DSSR analysis of the whole structures
    
    # print("> Downloading and annotating structures...", flush=True)
    # mmcif_list = sorted(set([ c.pdb_id for c in update ]))
    # try:
    #     p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=int(ncores*0.75))
    #     pbar = tqdm(mmcif_list, maxinterval=1.0, miniters=1, desc="mmCIF files")
    #     for _ in p.imap_unordered(work_mmcif, mmcif_list): 
    #         pbar.update(1) # Everytime the iteration finishes, update the global progress bar
    #     pbar.close()
    #     p.close()
    #     p.join()
    # except KeyboardInterrupt:
    #     warn("KeyboardInterrupt, terminating workers.", error=True)
    #     pbar.close()
    #     p.terminate()
    #     p.join()
    #     exit(1)

    # ===========================================================================
    # Extract the desired chain portions, 
    # and extract their informations
    # ===========================================================================

    if EXTRACT_CHAINS:
        if HOMOLOGY and not path.isdir(path_to_3D_data + "rna_mapped_to_Rfam"):
            os.makedirs(path_to_3D_data + "rna_mapped_to_Rfam") # for the portions mapped to Rfam
        if not HOMOLOGY and not path.isdir(path_to_3D_data + "rna_only"):
            os.makedirs(path_to_3D_data + "rna_only") # extract chains of pure RNA

    joblist = []
    for c in update:
        if (c.chain_label not in known_issues) or not USE_KNOWN_ISSUES:
            joblist.append(Job(function=work_build_chain, how_many_in_parallel=ncores, args=[c]))
    
    try:
        results = execute_joblist(joblist)
    except:
        print("Exiting")
        exit(1)

    # Remove the chains whose parsing resulted in errors
    loaded_chains = [ c[1] for c in results if not c[1].delete_me ]

    # Identify errors due to empty JSON files (this happen when RAM is full, we believe).
    # Retrying often solves the issue... so retry once with half the cores to limit the RAM usage.
    to_retry = [ c[1] for c in results if "Could not load existing" in c[1].error_messages ]
    if len(to_retry):

        # Redownload and re-annotate 
        print("> Retrying to annotate some structures which just failed.", flush=True)
        mmcif_list = sorted(set([ c.pdb_id for c in to_retry ]))
        for c in mmcif_list:
            os.remove(path_to_3D_data + "RNAcifs/" + c + ".cif")
            os.remove(path_to_3D_data + "annotations/" + c + ".json")
        try:
            p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=int(ncores*0.5))
            pbar = tqdm(mmcif_list, maxinterval=1.0, miniters=1, desc="mmCIF files")
            for _ in p.imap_unordered(work_mmcif, mmcif_list): 
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

        # redefine joblist
        joblist = []
        for c in to_retry:
            c.delete_me = 0
            c.error_messages = ""
            joblist.append(Job(function=work_build_chain, how_many_in_parallel=int(0.5*ncores), args=[c, True]))
        
        # Re run the build
        try:
            results = execute_joblist(joblist)
        except:
            print("Exiting")
            exit(1)
        
        # Add the saved chains to the list of successful builds
        loaded_chains += [ c[1] for c in results if not c[1].delete_me ]

    # At this point, structure, chain and nucleotide tables of the database are up to date.

    print(f"> Loaded {len(loaded_chains)} RNA chains ({len(update) - len(loaded_chains)} errors).")
    del update # Here ends its utility, so let's free some memory
    del joblist
    del results
 
    if not HOMOLOGY:
        # Save chains to file
        for c in loaded_chains:
            c.save()
        print("Completed.")
        exit()

    # ===========================================================================
    # Download RNA sequences of the corresponding Rfam families
    # ===========================================================================
    
    # Preparing a results folder
    if not os.access(path_to_seq_data + "realigned/", os.F_OK):
        os.makedirs(path_to_seq_data + "realigned/")
    if not path.isdir(path_to_seq_data + "rfam_sequences/fasta/"):
        os.makedirs(path_to_seq_data + "rfam_sequences/fasta/", exist_ok=True)

    # Get the list of Rfam families found
    rfam_acc_to_download = {}
    for c in loaded_chains:
        if c.rfam_fam not in rfam_acc_to_download:
            rfam_acc_to_download[c.rfam_fam] = [ c ]
        else:
            rfam_acc_to_download[c.rfam_fam].append(c)
    print(f"> Identified {len(rfam_acc_to_download.keys())} families to update and re-align with the crystals' sequences:")

    # Download the covariance models for all families
    dl.download_Rfam_cm()

    # Ask the SQL server how much we have to download for each family
    fam_list = sorted(rfam_acc_to_download.keys())
    dl.download_Rfam_family_stats(fam_list)

    # At this point, the family table is up to date

    joblist = []
    for f in fam_list:
        joblist.append(Job(function=work_prepare_sequences, how_many_in_parallel=ncores, args=[f]))
    try:
        results = execute_joblist(joblist)
        
        if len(set(fam_list).intersection({"RF00177", "RF01960"})):
            dl.download_from_SILVA("SSU")
        if len(set(fam_list).intersection({"RF00002", "RF02540", "RF02541", "RF02543"})):
            dl.download_from_SILVA("LSU")
    except KeyboardInterrupt:
        print("Exiting")
        exit(1)


    # ==========================================================================================
    # Realign sequences from 3D chains to Rfam's identified hits (--> extended full alignement)
    # ==========================================================================================   

    # Prepare the job list
    joblist = []
    for f in fam_list:
        joblist.append( Job( function=work_realign, args=[f, rfam_acc_to_download[f]],  # Apply work_realign to each RNA family
                                 how_many_in_parallel=1, label=f))  # the function already uses all CPUs so launch them one by one
    
    # Execute the jobs
    try:
        results = execute_joblist(joblist)
    except:
        print("Exiting")
        exit(1)
    conn = sqlite3.connect(runDir + "/results/RNANet.db")
    sql_execute(conn, """UPDATE family SET comput_time = ?, comput_peak_mem = ? 
                         WHERE rfam_acc = ?;""", many=True, data=[ (r[2], r[3], r[0]) for r in results ])
    conn.close()
    del joblist

    # ==========================================================================================
    # Now compute statistics on base variants at each position of every 3D chain
    # ==========================================================================================

    print("Computing nucleotide frequencies in alignments...")

    # Prepare the architecture of a shiny multi-progress-bars design
    thr_idx_mgr = Manager()                 # Push the number of workers to a queue. 
    idxQueue = thr_idx_mgr.Queue()          # ... Then each Pool worker will
    for i in range(ncores):                 # ... pick a number from the queue when starting the computation for one family, 
        idxQueue.put(i)                     # ... and replace it when the computation has ended so it could be picked up later.

    # Start a process pool to dispatch the RNA families,
    # over multiple CPUs (one family by CPU)
    p = Pool(initializer=init_worker, initargs=(tqdm.get_lock(),), processes=int(ncores*0.5), maxtasksperchild=5)

    try:
        fam_pbar = tqdm(total=len(fam_list), desc="RNA families", position=0, leave=True) 
        for i, _ in enumerate(p.imap_unordered(work_pssm, fam_list)): # Apply work_pssm to each RNA family
            fam_pbar.update(1) # Everytime the iteration finishes on a family, update the global progress bar over the RNA families
        fam_pbar.close()
        p.close()
        p.join()
    except KeyboardInterrupt:
        warn("KeyboardInterrupt, terminating workers.", error=True)
        fam_pbar.close()
        p.terminate()
        p.join()
        exit(1)
        
    # At this point, the align_column and re_mapping tables are up-to-date.

    # ==========================================================================================
    # Post computation tasks
    # ==========================================================================================

    # Archive the results
    if not path.isdir(path_to_3D_data + "datapoints/"):
        os.makedirs(path_to_3D_data + "datapoints/")
    os.makedirs("results/archive", exist_ok=True)
    time_str = time.strftime("%Y%m%d")
    subprocess.run(["tar","-C", path_to_3D_data + "/datapoints","-czf",f"results/archive/RNANET_datapoints_{time_str}.tar.gz","."])
    subprocess.run(["rm", "-f", runDir + "/results/RNANET_datapoints_latest.tar.gz"])
    subprocess.run(['ln',"-s", runDir +f"/results/archive/RNANET_datapoints_{time_str}.tar.gz", runDir + "/results/RNANET_datapoints_latest.tar.gz"])
    conn = sqlite3.connect(runDir+"/results/RNANet.db")
    pd.read_sql_query("SELECT rfam_acc, idty_percent, nb_homologs, nb_3d_chains, nb_total_homol, max_len, comput_time, comput_peak_mem from family", 
                      conn).to_csv(runDir + "/results/families.csv")
    pd.read_sql_query("""SELECT structure_id, chain_name, pdb_start, pdb_end, rfam_acc, inferred, 
    reversed, date, exp_method, resolution, issue FROM
    structure NATURAL JOIN chain
    ORDER BY structure_id, chain_name, rfam_acc ASC;
    """, conn).to_csv(runDir + "/results/summary.csv")
    conn.close()

    # Run statistics
    if RUN_STATS:
        # Remove previous precomputed data
        subprocess.run(["rm","-f", "data/wadley_kernel_eta.npz", "data/wadley_kernel_eta_prime.npz"])

        # Run statistics files
        os.chdir(runDir)
        subprocess.run(["python3", "regression.py"])
        subprocess.run(["python3", "statistics.py", path_to_3D_data, path_to_seq_data])

    print("Completed.")  # This part of the code is supposed to release some serotonin in the modeller's brain, do not remove

    # # so i can sleep for the end of the night
    # subprocess.run(["shutdown","now"]) 
