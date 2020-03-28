#!/usr/bin/python3.8
import numpy as np
import pandas as pd
import concurrent.futures, Bio.PDB.StructureBuilder, getopt, gzip, io, json, os, psutil, re, requests, sqlalchemy, subprocess, sys, time, warnings
from Bio import AlignIO, SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio._py3k import urlretrieve as _urlretrieve
from Bio._py3k import urlcleanup as _urlcleanup
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment, AlignInfo
from collections import OrderedDict
from functools import partial
from os import path, makedirs
from multiprocessing import Pool, Manager
from time import sleep
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


pd.set_option('display.max_rows', None)
m = Manager()
running_stats = m.list()
running_stats.append(0) # n_launched
running_stats.append(0) # n_finished
running_stats.append(0) # n_skipped
runDir = path.dirname(path.realpath(__file__))
path_to_3D_data = "tobedefinedbyoptions"
path_to_seq_data = "tobedefinedbyoptions"
validsymb = '\U00002705'
warnsymb = '\U000026A0'
errsymb = '\U0000274C'

# Default options:
CRYSTAL_RES = "4.0"
KEEP_HETATM = False
FILL_GAPS = True 
HOMOLOGY = True
USE_KNOWN_ISSUES = True

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

        # # I don't really know what this is but the doc said to warn:          
        # if icode != " ":
        #     warn(f"icode {icode} at position {resseq}\t\t")

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

    def __init__(self, pdb_id, pdb_model, pdb_chain_id, chain_label, rfam="", pdb_start=None, pdb_end=None):
        self.pdb_id = pdb_id                    # PDB ID
        self.pdb_model = int(pdb_model)         # model ID, starting at 1
        self.pdb_chain_id = pdb_chain_id        # chain ID (mmCIF), multiple letters
        self.pdb_start = pdb_start              # if portion of chain, the start number (relative to the chain, not residue numbers)
        self.pdb_end = pdb_end                  # if portion of chain, the start number (relative to the chain, not residue numbers)
        self.reversed = False                   # wether pdb_end > pdb_start in the Rfam mapping
        self.chain_label = chain_label          # chain pretty name 
        self.full_mmCIFpath = ""                # path to the source mmCIF structure
        self.file = ""                          # path to the 3D PDB file
        self.rfam_fam = rfam                    # mapping to an RNA family
        self.seq = ""                           # sequence with modified nts
        self.aligned_seq = ""                   # sequence with modified nts replaced, but gaps can exist
        self.length = -1                        # length of the sequence (missing residues are not counted)
        self.full_length = -1                   # length of the chain extracted from source structure ([start; stop] interval)
        self.delete_me = False                  # an error occured during production/parsing
        self.error_messages = ""                # Error message(s) if any
        self.data = None                        # Pandas DataFrame with all the 3D data extracted by DSSR.

    def __str__(self):
        return self.pdb_id + '[' + str(self.pdb_model) + "]-" + self.pdb_chain_id
    
    def __eq__(self, other):
        return self.chain_label == other.chain_label and str(self) == str(other)

    def __hash__(self):
        return hash((self.pdb_id, self.pdb_model, self.pdb_chain_id, self.chain_label))

    def download_3D(self):
        """ Look for the main CIF file (with all chains) from RCSB
        """

        status = f"\t> Download {self.pdb_id}.cif\t\t\t"
        url = 'http://files.rcsb.org/download/%s.cif' % (self.pdb_id)
        final_filepath = path_to_3D_data+"RNAcifs/"+self.pdb_id+".cif"

        # Check if file already exists, if yes, abort
        if os.path.exists(final_filepath):
            print(status + f"\t{validsymb}\t(structure exists)")
            self.full_mmCIFpath = final_filepath
            return
        
        # Attempt to download it
        try:
            _urlcleanup()
            _urlretrieve(url, final_filepath)
            self.full_mmCIFpath = final_filepath
            print(status + f"\t{validsymb}")
        except IOError:
            print(status + f"\tERR \U0000274E\t\033[31mError downloading {url} !\033[0m")
            self.delete_me = True
            self.error_messages = f"Error downloading {url}"

    def extract_portion(self):
        """ Extract the part which is mapped to Rfam from the main CIF file and save it to another file.
        """
        
        status = f"\t> Extract {self.pdb_start}-{self.pdb_end} atoms from {self.pdb_id}-{self.pdb_chain_id}\t"
        self.file = path_to_3D_data+"rna_mapped_to_Rfam/"+self.chain_label+".cif"

        # Check if file exists, if yes, abort (do not recompute)
        if os.path.exists(self.file):
            print(status + f"\t{validsymb}\t(already done)", flush=True)
            return

        model_idx = self.pdb_model - (self.pdb_model > 0) # because arrays start at 0, models start at 1
        pdb_start = int(self.pdb_start)
        pdb_end = int(self.pdb_end)

       
        with warnings.catch_warnings():
            # Ignore the PDB problems. This mostly warns that some chain is discontinuous.
            warnings.simplefilter('ignore', PDBConstructionWarning)  

            # Check if the whole mmCIF file exists. If not, abort.
            if self.full_mmCIFpath == "":
                print(status + f"\t\U0000274E\t\033[31mError with CIF file of {self.pdb_id} !\033[0m", flush=True)
                self.delete_me = True
                self.error_messages = f"Error with CIF file of {self.pdb_id}"
                return

            # Load the whole mmCIF into a Biopython structure object:
            s = mmcif_parser.get_structure(self.pdb_id, self.full_mmCIFpath)

            # Extract the desired chain
            c = s[model_idx][self.pdb_chain_id]

            # Pay attention to residue numbering
            first_number = c.child_list[0].get_id()[1]          # the chain's first residue is numbered 'first_number'
            if pdb_start < pdb_end:                             
                start = pdb_start + first_number - 1            # shift our start_position by 'first_number'
                end = pdb_end + first_number - 1                # same for the end position
            else:
                self.reversed = True                            # the 3D chain is numbered backwards compared to the Rfam family
                end = pdb_start + first_number - 1
                start = pdb_end + first_number - 1

            # Define a selection
            sel = NtPortionSelector(model_idx, self.pdb_chain_id, start, end)

            # Save that selection on the mmCIF object s to file
            ioobj = MMCIFIO()
            ioobj.set_structure(s)
            ioobj.save(self.file, sel)

        print(status + f"\t{validsymb}")
    
    def extract_all(self):
        """ Extract the RNA chain from the main CIF file and save it to another file.
        """
        
        status = f"\t> Extract {self.pdb_id}-{self.pdb_chain_id}\t"
        self.file = path_to_3D_data+"rna_only/"+self.chain_label+".cif"

        # Check if file exists, if yes, abort (do not recompute)
        if os.path.exists(self.file):
            print(status + f"\t{validsymb}\t(already done)", flush=True)
            return

        model_idx = self.pdb_model - (self.pdb_model > 0) # because arrays start at 0, models start at 1
       
        with warnings.catch_warnings():
            # Ignore the PDB problems. This mostly warns that some chain is discontinuous.
            warnings.simplefilter('ignore', PDBConstructionWarning) # ignore the PDB problems 

            # Check if the whole mmCIF file exists. If not, abort.
            if self.full_mmCIFpath == "":
                print(status + f"\t\U0000274E\t\033[31mError with CIF file of {self.pdb_id} !\033[0m", flush=True)
                self.delete_me = True
                self.error_messages = f"Error with CIF file of {self.pdb_id}"
                return

            # Load the whole mmCIF into a Biopython structure object:
            s = mmcif_parser.get_structure(self.pdb_id, self.full_mmCIFpath)

            # Extract the desired chain
            c = s[model_idx][self.pdb_chain_id]

            # Define a selection
            first_number = c.child_list[0].get_id()[1]  # the chain's first residue is numbered 'first_number'
            last_number = c.child_list[-1].get_id()[1]  # the chain's last residue number
            sel = NtPortionSelector(model_idx, self.pdb_chain_id, first_number, last_number)

            # Save that selection on the mmCIF object s to file
            ioobj = MMCIFIO()
            ioobj.set_structure(s)
            ioobj.save(self.file, sel)

        print(status + f"\t{validsymb}")

    def set_rfam(self, rfam):
        """ Rember the Rfam mapping for this chain.
        """

        self.rfam_fam = rfam

    def extract_3D_data(self):
        """ Runs DSSR to annotate the 3D chain and get various information about it. """

        # Check if the file exists. If no, compute it.
        if not os.path.exists(path_to_3D_data+f"annotations/{self.chain_label}.{self.rfam_fam}.csv"):

            # run DSSR (you need to have it in your $PATH, follow x3dna installation instructions)
            output = subprocess.run(
                ["x3dna-dssr", f"-i={self.file}", "--json", "--auxfile=no"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = output.stdout.decode('utf-8') # this contains the results in JSON format, or is empty if there are errors
            stderr = output.stderr.decode('utf-8') # this contains the evenutal errors

            try:
                if "exception" in stderr:
                    # DSSR is unable to parse the chain.
                    warn(f"Exception while running DSSR: {stderr}\n\tIgnoring {self.chain_label}.\t\t\t", error=True)
                    self.delete_me = True
                    self.error_messages = f"Exception while running DSSR for {self.chain_label}:\n {stderr}"
                    return

                # Get the JSON from DSSR output 
                json_object = json.loads(stdout)

                # Print eventual warnings given by DSSR, and abort if there are some
                if "warning" in json_object.keys():
                    warn(f"Ignoring {self.chain_label} ({json_object['warning']})\t", error=True)
                    self.delete_me = True
                    self.error_messages = f"DSSR warning for {self.chain_label}: {json_object['warning']}"
                    return

                # Extract the interesting parts
                nts = json_object["nts"]

                # Prepare a data structure (Pandas DataFrame)
                resnum_start = int(nts[0]["nt_resnum"])
                df = pd.DataFrame(nts)
                # remove low pertinence or undocumented descriptors
                df = df.drop(['summary', 'chain_name', 'index', 'splay_angle',
                               'splay_distance', 'splay_ratio', 'sugar_class',
                               'bin', 'suiteness', 'cluster'], axis=1)
                # df['P_x'] = [ float(i[0]) if i[0] is not None else np.NaN for i in df['P_xyz'] ]                #
                # df['P_y'] = [ float(i[1]) if i[1] is not None else np.NaN for i in df['P_xyz'] ]                #
                # df['P_z'] = [ float(i[2]) if i[2] is not None else np.NaN for i in df['P_xyz'] ]                # Flatten the 
                # df['C5prime_x'] = [ float(i[0]) if i[0] is not None else np.NaN for i in df['C5prime_xyz'] ]    # Python dictionary
                # df['C5prime_y'] = [ float(i[1]) if i[1] is not None else np.NaN for i in df['C5prime_xyz'] ]    #
                # df['C5prime_z'] = [ float(i[2]) if i[2] is not None else np.NaN for i in df['C5prime_xyz'] ]    #

                # Convert angles to radians
                df.loc[:,['alpha', 'beta','gamma','delta','epsilon','zeta','epsilon_zeta','chi','v0', 'v1', 'v2', 'v3', 'v4',
                         'eta','theta','eta_prime','theta_prime','eta_base','theta_base', 'phase_angle']] *= np.pi/180.0
                # mapping [-pi, pi] into [0, 2pi]
                df.loc[:,['alpha', 'beta','gamma','delta','epsilon','zeta','epsilon_zeta','chi','v0', 'v1', 'v2', 'v3', 'v4',
                         'eta','theta','eta_prime','theta_prime','eta_base','theta_base', 'phase_angle']] %= (2.0*np.pi)

                # Add a sequence column just for the alignments
                df['nt_align_code'] = [ str(x).upper()
                                              .replace('NAN', '-')      # Unresolved nucleotides are gaps
                                              .replace('?', '-')         # Unidentified residues, let's delete them
                                              .replace('T', 'U')        # 5MU are modified to t, which gives T
                                              .replace('P', 'U')        # Pseudo-uridines, but it is not really right to change them to U, see DSSR paper, Fig 2
                                        for x in df['nt_code'] ]

                # Shift numbering when duplicate residue numbers are found.
                # Example: 4v9q-DV contains 17 and 17A which are both read 17 by DSSR.
                while True in df.duplicated(['nt_resnum']).values:
                    i = df.duplicated(['nt_resnum']).values.tolist().index(True)
                    df.iloc[i:, 1] += 1
                l = df.iloc[-1,1] - df.iloc[0,1] + 1

                # Add eventual missing rows because of unsolved residues in the chain:
                if l != len(df['index_chain']):
                    # We have some rows to add. First, identify them:
                    diff = set(range(l)).difference(df['nt_resnum'] - resnum_start)

                    for i in sorted(diff):
                        df = pd.concat([df.iloc[:i-1], pd.DataFrame({"index_chain": i, "nt_resnum": i+resnum_start-1, 
                                                                     "nt_code":'-', "nt_name":'-', 'nt_align_code':'-'}, index=[i-1]), df.iloc[i-1:]])
                        df.iloc[i:, 0] += 1
                    df = df.reset_index(drop=True)


                # Iterate over pairs to identify base-base interactions
                res_ids = list(df['nt_id']) # things like "chainID.C4, chainID.U5"
                paired = [ "0" ] * l
                pair_type_LW = [ '' ] * l
                pair_type_DSSR = [ '' ] * l
                interacts = [ 0 ] * l
                if "pairs" in json_object.keys():
                    pairs = json_object["pairs"]
                    for p in pairs:
                        nt1 = p["nt1"]
                        nt2 = p["nt2"]
                        if nt1 in res_ids and nt2 in res_ids:
                            nt1_idx = res_ids.index(nt1)
                            nt2_idx = res_ids.index(nt2)
                            if paired[nt1_idx] == "0":
                                paired[nt1_idx] = str(nt2_idx + 1)
                                pair_type_LW[nt1_idx] = p["LW"]
                                pair_type_DSSR[nt1_idx] = p["DSSR"]
                            else:
                                paired[nt1_idx] += ',' + str(nt2_idx + 1)
                                pair_type_LW[nt1_idx] += ',' + p["LW"]
                                pair_type_DSSR[nt1_idx] += ',' + p["DSSR"]
                            if paired[nt2_idx] == "0":
                                paired[nt2_idx] = str(nt1_idx + 1)
                                pair_type_LW[nt2_idx] = p["LW"]
                                pair_type_DSSR[nt2_idx] = p["DSSR"]
                            else:
                                paired[nt2_idx] += ',' + str(nt1_idx + 1)
                                pair_type_LW[nt2_idx] += ',' + p["LW"]
                                pair_type_DSSR[nt2_idx] += ',' + p["DSSR"]
                            interacts[nt1_idx] += 1
                            interacts[nt2_idx] += 1
                        elif nt1 in res_ids:
                            nt1_idx = res_ids.index(nt1)
                            interacts[nt1_idx] += 1
                        elif nt2 in res_ids:
                            nt2_idx = res_ids.index(nt2)
                            interacts[nt2_idx] += 1
                df['paired'] = paired
                df['pair_type_LW'] = pair_type_LW
                df['pair_type_DSSR'] = pair_type_DSSR
                df['nb_interact'] = interacts
                df = df.drop(['C5prime_xyz', 'P_xyz', 'nt_id'], axis=1) # remove now useless descriptors

                if self.reversed:
                    # The 3D structure is numbered from 3' to 5' instead of standard 5' to 3'
                    # or the sequence that matches the Rfam family is 3' to 5' instead of standard 5' to 3'.
                    # Anyways, you need to invert the angles.
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
                                temp_v.append(str(l-int(_)+1))
                            newpairs.append(','.join(temp_v))
                        else:
                            if int(v):
                                newpairs.append(str(l-int(v)+1))
                    df['paired'] = newpairs
            except KeyError as e:
                # Mostly, there are no part about nucleotides in the DSSR output. Abort.
                warn(f"Error while parsing DSSR's json output:\n{e}\n\tignoring {self.chain_label}\t\t\t\t", error=True)
                self.delete_me = True
                self.error_messages = f"Error while parsing DSSR's json output:\n{e}"
                return
            
            # Creating a df for easy saving to CSV
            df.to_csv(path_to_3D_data + f"annotations/{self.chain_label}.{self.rfam_fam}.csv")
            del df
            print("\t> Saved", self.chain_label, f"annotations to CSV.\t\t{validsymb}", flush=True)
        else:
            print("\t> Computing", self.chain_label, f"annotations...\t{validsymb}\t(already done)", flush=True)

        # Now load data from the CSV file
        d = pd.read_csv(path_to_3D_data+f"annotations/{self.chain_label}.{self.rfam_fam}.csv", index_col=0)
        self.seq = "".join(d.nt_code.values)
        self.aligned_seq = "".join(d.nt_align_code.values)
        self.length = len([ x for x in self.aligned_seq if x != "-" ])
        self.full_length = len(d.nt_code)
        self.data = d
        print(f"\t> Loaded data from CSV\t\t\t\t{validsymb}", flush=True)

        # Remove too short chains
        if self.length < 5:
            warn(f"{self.chain_label} sequence is too short, let's ignore it.\t", error=True)
            self.delete_me = True
            self.error_messages = "Sequence is too short. (< 5 resolved nts)"
        return

    def set_freqs_from_aln(self, s_seq, ali_freqs):
        """Maps the object's sequence to its version in a MSA, to compute nucleotide frequencies at every position.
        
        s_seq: the aligned version of self.aligned_seq
        ali_freqs: the nucleotide frequencies at every position of s_seq
        This also replaces gaps by the most common nucleotide.
        """
        alilen = len(s_seq)

        # Save colums in the appropriate positions
        i = 0
        j = 0
        temp_freqs = np.zeros((5,0))
        while i<self.full_length and j<alilen:
            # Here we try to map self.aligned_seq (the sequence of the 3D chain, including gaps when residues are missing), 
            # with s_seq, the sequence aligned in the MSA, containing any of ACGU and two types of gaps, - and .

            if self.aligned_seq[i] == s_seq[j].upper(): # alignment and sequence correspond (incl. gaps)
                temp_freqs = np.concatenate((temp_freqs, ali_freqs[:,j].reshape(-1,1)), axis=1)
                i += 1
                j += 1
            elif self.aligned_seq[i] == '-': # gap in the chain, but not in the aligned sequence

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
                    temp_freqs = np.concatenate((temp_freqs, ali_freqs[:,j].reshape(-1,1)), axis=1)
                    i += 1
                    j += 1
                    continue

                # else, just ignore the gap.
                temp_freqs = np.concatenate((temp_freqs, np.array([0.0,0.0,0.0,0.0,1.0]).reshape(-1,1)), axis=1)
                i += 1
            elif s_seq[j] in ['.', '-']: # gap in the alignment, but not in the real chain
                j += 1 # ignore the column
            else: # sequence mismatch which is not a gap...
                print(f"You are never supposed to reach this. Comparing {self.chain_label} in {i} ({self.aligned_seq[i-1:i+2]}) with seq[{j}] ({s_seq[j-3:j+4]}).\n", 
                        self.aligned_seq, 
                        sep='', flush=True)
                exit(1)

        # Replace gapped positions by the consensus sequence:
        if FILL_GAPS:
            c_aligned_seq = list(self.aligned_seq)
            c_seq = list(self.seq)
            letters = ['A', 'C', 'G', 'U', 'N']
            for i in range(self.full_length):
                if c_aligned_seq[i] == '-':      # (then c_seq[i] also is)
                    freq = temp_freqs[:,i]
                    l = letters[freq.tolist().index(max(freq))]
                    c_aligned_seq[i] = l
                    c_seq[i] = l
                    self.data.iloc[i,3] = l # self.data['nt_code'][i]
            self.aligned_seq = ''.join(c_aligned_seq)
            self.seq = ''.join(c_seq)

        # Temporary np array to store the computations
        point = np.zeros((11, self.full_length))
        for i in range(self.full_length):
            # normalized position in the chain
            point[0,i] = float(i+1)/self.full_length 

            # one-hot encoding of the actual sequence
            if self.seq[i] in letters[:4]:
                point[ 1 + letters[:4].index(self.seq[i]), i ] = 1
            else:
                point[5,i] = 1

            # PSSMs
            point[6,i] = temp_freqs[0, i]
            point[7,i] = temp_freqs[1, i]
            point[8,i] = temp_freqs[2, i]
            point[9,i] = temp_freqs[3, i]
            point[10,i] = temp_freqs[4, i]
        
        self.data = pd.concat([self.data, pd.DataFrame(point.T, columns=["nt_position","is_A","is_C","is_G","is_U","is_other","freq_A","freq_C","freq_G","freq_U","freq_other"])], axis=1)
        # reorder columns:
        cols = [ # 1D structure descriptors
                'index_chain','nt_resnum','position',
                'nt_name','nt_code','nt_align_code',
                'is_A','is_C','is_G','is_U','is_other',
                'freq_A','freq_C','freq_G','freq_U','freq_other',
                
                # 2D structure descriptors
                'dbn','paired','nb_interact',
                'pair_type_LW','pair_type_DSSR',
                
                # 3D strcuture descriptors
                'alpha','beta','gamma','delta','epsilon','zeta','epsilon_zeta','chi',
                'bb_type','glyco_bond','form','ssZp','Dp',
                'eta','theta','eta_prime','theta_prime','eta_base','theta_base',
                'v0', 'v1', 'v2', 'v3', 'v4', 'amplitude', 'phase_angle', 'puckering'
               ]
        self.data = self.data[cols]
        
    def save(self, fformat = "csv"):
        # save to file
        if fformat == "csv":
            self.data.to_csv(path_to_3D_data + "datapoints/" + self.chain_label + str('.'+self.rfam_fam if self.rfam_fam != '' else ''))


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


class AnnotatedStockholmIterator(AlignIO.StockholmIO.StockholmIterator):
    """ A custom Stockholm format MSA parser that returns annotations at the end.

    Inherits from Bio.AlignIO and simply overloads the __next__() method to save the 
    gr, gf and gs dicts at the end.
    """

    def __next__(self):
        """Parse the next alignment from the handle."""
        handle = self.handle

        if self._header is None:
            line = handle.readline()
        else:
            # Header we saved from when we were parsing the previous alignment.
            line = self._header
            self._header = None

        if not line:
            # Empty file - just give up.
            raise StopIteration
        if line.strip() != "# STOCKHOLM 1.0":
            raise ValueError("Did not find STOCKHOLM header")

        # Note: If this file follows the PFAM conventions, there should be
        # a line containing the number of sequences, e.g. "#=GF SQ 67"
        # We do not check for this - perhaps we should, and verify that
        # if present it agrees with our parsing.

        seqs = {}
        ids = OrderedDict()  # Really only need an OrderedSet, but python lacks this
        gs = {}
        gr = {}
        gf = {}
        gc = {}
        passed_end_alignment = False
        while True:
            line = handle.readline()
            if not line:
                break  # end of file
            line = line.strip()  # remove trailing \n
            if line == "# STOCKHOLM 1.0":
                self._header = line
                break
            elif line == "//":
                # The "//" line indicates the end of the alignment.
                # There may still be more meta-data
                passed_end_alignment = True
            elif line == "":
                # blank line, ignore
                pass
            elif line[0] != "#":
                # Sequence
                # Format: "<seqname> <sequence>"
                assert not passed_end_alignment
                parts = [x.strip() for x in line.split(" ", 1)]
                if len(parts) != 2:
                    # This might be someone attempting to store a zero length sequence?
                    raise ValueError(
                        "Could not split line into identifier "
                        "and sequence:\n" + line)
                seq_id, seq = parts
                if seq_id not in ids:
                    ids[seq_id] = True
                seqs.setdefault(seq_id, "")
                seqs[seq_id] += seq.replace(".", "-")
            elif len(line) >= 5:
                # Comment line or meta-data
                if line[:5] == "#=GF ":
                    # Generic per-File annotation, free text
                    # Format: #=GF <feature> <free text>
                    feature, text = line[5:].strip().split(None, 1)
                    # Each feature key could be used more than once,
                    # so store the entries as a list of strings.
                    if feature not in gf:
                        gf[feature] = [text]
                    else:
                        gf[feature].append(text)
                elif line[:5] == "#=GC ":
                    # Generic per-Column annotation, exactly 1 char per column
                    # Format: "#=GC <feature> <exactly 1 char per column>"
                    feature, text = line[5:].strip().split(None, 2)
                    if feature not in gc:
                        gc[feature] = ""
                    gc[feature] += text.strip()  # append to any previous entry
                    # Might be interleaved blocks, so can't check length yet
                elif line[:5] == "#=GS ":
                    # Generic per-Sequence annotation, free text
                    # Format: "#=GS <seqname> <feature> <free text>"
                    seq_id, feature, text = line[5:].strip().split(None, 2)
                    # if seq_id not in ids:
                    #    ids.append(seq_id)
                    if seq_id not in gs:
                        gs[seq_id] = {}
                    if feature not in gs[seq_id]:
                        gs[seq_id][feature] = [text]
                    else:
                        gs[seq_id][feature].append(text)
                elif line[:5] == "#=GR ":
                    # Generic per-Sequence AND per-Column markup
                    # Format: "#=GR <seqname> <feature> <exactly 1 char per column>"
                    seq_id, feature, text = line[5:].strip().split(None, 2)
                    # if seq_id not in ids:
                    #    ids.append(seq_id)
                    if seq_id not in gr:
                        gr[seq_id] = {}
                    if feature not in gr[seq_id]:
                        gr[seq_id][feature] = ""
                    gr[seq_id][feature] += text.strip()  # append to any previous entry
                    # Might be interleaved blocks, so can't check length yet
            # Next line...

        assert len(seqs) <= len(ids)
        # assert len(gs)   <= len(ids)
        # assert len(gr)   <= len(ids)

        self.ids = ids.keys()
        self.sequences = seqs           #
        self.seq_annotation = gs        # This is the new part:
        self.seq_col_annotation = gr    # Saved for later use.
        self.alignment_annotation = gf  #

        if ids and seqs:

            if self.records_per_alignment is not None and self.records_per_alignment != len(ids):
                raise ValueError("Found %i records in this alignment, told to expect %i" % (len(ids), self.records_per_alignment))

            alignment_length = len(list(seqs.values())[0])
            records = []  # Alignment obj will put them all in a list anyway
            for seq_id in ids:
                seq = seqs[seq_id]
                if alignment_length != len(seq):
                    raise ValueError("Sequences have different lengths, or repeated identifier")
                name, start, end = self._identifier_split(seq_id)
                record = SeqRecord(Seq(seq, self.alphabet),
                                   id=seq_id, name=name, description=seq_id,
                                   annotations={"accession": name})
                # Accession will be overridden by _populate_meta_data if an explicit accession is provided:
                record.annotations["accession"] = name

                if start is not None:
                    record.annotations["start"] = start
                if end is not None:
                    record.annotations["end"] = end

                self._populate_meta_data(seq_id, record)
                records.append(record)
            for k, v in gc.items():
                if len(v) != alignment_length:
                    raise ValueError("%s length %i, expected %i" % (k, len(v), alignment_length))
            alignment = MultipleSeqAlignment(records, self.alphabet)

            for k, v in sorted(gc.items()):
                if k in self.pfam_gc_mapping:
                    alignment.column_annotations[self.pfam_gc_mapping[k]] = v
                elif k.endswith("_cons") and k[:-5] in self.pfam_gr_mapping:
                    alignment.column_annotations[self.pfam_gr_mapping[k[:-5]]] = v
                else:
                    # Ignore it?
                    alignment.column_annotations["GC:" + k] = v

            # TODO - Introduce an annotated alignment class?
            # For now, store the annotation a new private property:
            alignment._annotations = gr
            alignment._fileannotations = gf

            return alignment
        else:
            raise StopIteration


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

def read_cpu_number():
    # As one shall not use os.cpu_count() on LXC containers,
    # because it reads info from /sys wich is not the VM resources but the host resources.
    # This function reads it from /proc/cpuinfo instead.
    p = subprocess.run(['grep', '-Ec', '(Intel|AMD)', '/proc/cpuinfo'], stdout=subprocess.PIPE)
    return int(int(p.stdout.decode('utf-8')[:-1])/2)

def warn(message, error=False):
    """Pretty-print warnings and error messages.
    """
    if error:
        print(f"\t> \033[31mERR: {message}\033[0m{errsymb}", flush=True)
    else:
        print(f"\t> \033[33mWARN: {message}\033[0m{warnsymb}", flush=True)

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # put the monitor in a different thread
            assistant_future = executor.submit(monitor.check_mem_usage)
            
            # run the command. subprocess.run will be a child of this process, and stays monitored.
            start_time = time.time()
            r = subprocess.run(j.cmd_, timeout=j.timeout_, stdout=subprocess.DEVNULL)
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

def execute_joblist(fulljoblist, printstats=False):
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

    if printstats:
        # Write statistics in a file (header here)
        f = open("jobstats.csv", "w")
        f.write("label,comp_time,max_mem\n")
        f.close()

    # Process the jobs from priority 1 to nprio
    results = {}
    for i in range(1,nprio+1):
        if i not in jobs.keys(): continue # no job has the priority level i

        print("processing jobs of priority", i)
        different_thread_numbers = sorted(jobs[i].keys())
        
        # jobs should be processed 1 by 1, 2 by 2, or n by n depending on their definition
        res = []
        for n in different_thread_numbers:
            # get the bunch of jobs of same priority and thread number
            bunch = jobs[i][n]
            if not len(bunch): continue # no jobs should be processed n by n

            print("using", n, "processes:")
            # execute jobs of priority i that should be processed n by n:
            p = Pool(processes=n)
            raw_results = p.map(partial(execute_job, jobcount=jobcount), bunch)
            p.close()
            p.join()

            if printstats:
                # Extract computation times
                times = [ r[0] for r in raw_results ]
                mems = [ r[1] for r in raw_results ]

                # Write them to file
                f = open("jobstats.csv", "a")
                for j, t, m in zip(bunch, times, mems):
                    j.comp_time = t
                    j.max_mem = m
                    print(f"\t> {j.label} finished in {t:.2f} sec with {int(m/1000000):d} MB of memory. \t{validsymb}", flush=True)
                    f.write(f"{j.label},{t},{m}\n")
                f.close()
            
            # Separate the job results in a different list 
            res += [ r[2] for r in raw_results ]

        # Add the results of this tree branch to the main list
        results[i] = res
    
    # throw back the money
    return results

def download_Rfam_PDB_mappings():
    """Query the Rfam public MySQL database for mappings between their RNA families and PDB structures.

    """
    # Download PDB mappings to Rfam family
    print("> Fetching latest PDB mappings from Rfam...", end='', flush=True)
    try:
        db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
        mappings = pd.read_sql('SELECT rfam_acc, pdb_id, chain, pdb_start, pdb_end, bit_score, evalue_score, cm_start, cm_end, hex_colour FROM pdb_full_region WHERE is_significant=1;', con=db_connection)
        mappings.to_csv(path_to_3D_data + 'Rfam-PDB-mappings.csv')
        print(f"\t{validsymb}")
    except sqlalchemy.exc.OperationalError:  # Cannot connect :'()
        print(f"\t{errsymb}")
        # Check if a previous run succeeded (if file exists, use it)
        if path.isfile(path_to_3D_data + 'Rfam-PDB-mappings.csv'):
            print("\t> Using previous version.")
            mappings = pd.read_csv(path_to_3D_data + 'Rfam-PDB-mappings.csv')
        else: # otherwise, abort.
            print("Can't do anything without data. Can't reach mysql-rfam-public.ebi.ac.uk on port 4497. Is it open on your system ? Exiting.")
            exit(1)

    return mappings

def download_Rfam_seeds():
    """ Download the seed sequence alignments from Rfam.

    Does not download if already there. It uses their FTP.
    """
    # If the seeds are not available, download them
    if not path.isfile(path_to_seq_data + "seeds/Rfam.seed.gz"):
        _urlcleanup()
        _urlretrieve('ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz', path_to_seq_data + "seeds/Rfam.seed.gz")

    # Prepare containers for the data
    aligned_records = []
    rfam_acc = []
    alignment_len = []
    alignment_nseq = []

    # Tell Biopython to use our overload
    AlignIO._FormatToIterator["stockholm"] = AnnotatedStockholmIterator 

    # Read the seeds
    with gzip.open(path_to_seq_data + "seeds/Rfam.seed.gz", encoding='latin-1') as gz:
        alignments = AlignIO.parse(gz, "stockholm", alphabet=generic_rna)
    
    # Fill the containers
    for align in alignments:
        aligned_records.append('\n'.join([ str(s.seq) for s in align ]))
        rfam_acc.append(align._fileannotations["AC"][0])
        alignment_len.append(align.get_alignment_length())
        alignment_nseq.append(len(align._records))

    # Build a dataframe with the containers
    Rfam_seeds = pd.DataFrame()
    Rfam_seeds["aligned_records"] = aligned_records
    Rfam_seeds["rfam_acc"] = rfam_acc
    Rfam_seeds["alignment_len"] = alignment_len
    Rfam_seeds["alignment_nseq"] = alignment_nseq

    return Rfam_seeds

def download_Rfam_cm():
    """ Download the covariance models from Rfam.
    
    Does not download if already there.
    """

    print(f"\t> Download Rfam.cm.gz from Rfam...\t", end='', flush=True)
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
        print(f"\t{validsymb}\t(no need)", flush=True)

def download_Rfam_family_stats(list_of_families):
    """Query the Rfam public MySQL database for statistics about their RNA families.

    Family ID, number of sequences identified, maximum length of those sequences.
    """
    try: 
        db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')

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

        # Query the database
        d = pd.read_sql(q, con=db_connection)

        # filter the results to families we are interested in
        return d[ d["rfam_acc"].isin(list_of_families) ]
    except sqlalchemy.exc.OperationalError:
        warn("Something's wrong with the SQL database. Check mysql-rfam-public.ebi.ac.uk status and try again later. Not printing statistics.")
        return {}

def download_Rfam_sequences(rfam_acc):
    """ Downloads the unaligned sequences known related to a given RNA family.

    Actually gets a FASTA archive from the public Rfam FTP. Does not download if already there."""

    print(f"\t\t> Download {rfam_acc}.fa.gz from Rfam...", end='', flush=True)
    if not path.isfile(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz"):
        try:
            _urlcleanup()
            _urlretrieve(   f'ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/{rfam_acc}.fa.gz',
                            path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz")
            print(f"\t{validsymb}")
        except:
            warn(f"Error downloading {rfam_acc}.fa.gz. Does it exist ?\t", error=True)
    else:
        print(f"\t{validsymb}\t(already there)", flush=True)

def download_BGSU_NR_list():
    """ Downloads a list of RNA 3D structures proposed by Bowling Green State University RNA research group.

    Does not remove structural redundancy. Resolution threshold used is 4 Angströms.
    """

    print(f"> Fetching latest list of RNA files at {CRYSTAL_RES} A resolution from BGSU website...", end='', flush=True)
    # Download latest BGSU non-redundant list
    try:
        s = requests.get(f"http://rna.bgsu.edu/rna3dhub/nrlist/download/current/{CRYSTAL_RES}A/csv").content
        nr = open(path_to_3D_data + f"latest_nr_list_{CRYSTAL_RES}A.csv", 'w')
        nr.write("class,representative,class_members\n")
        nr.write(io.StringIO(s.decode('utf-8')).getvalue())
        nr.close()
    except:
        warn("Error downloading NR list !\t", error=True)

        # Try to read previous file
        if path.isfile(path_to_3D_data + f"latest_nr_list_{CRYSTAL_RES}A.csv"):
            print("\t> Use of the previous version.\t", end = "", flush=True)
        else:
            return [], []

    nrlist = pd.read_csv(path_to_3D_data + f"latest_nr_list_{CRYSTAL_RES}A.csv")
    full_structures_list = nrlist['class_members'].tolist()
    print(f"\t{validsymb}", flush=True)

    # The beginning of an adventure.
    return full_structures_list

def build_chain(c):
    """ Additionally adds all the desired information to a Chain object.

    """
    # Download the whole mmCIF file containing the chain we are interested in
    c.download_3D()

    # If no problems, extract the portion we want
    if not c.delete_me:
        if HOMOLOGY:
            c.extract_portion()
        else:
            c.extract_all()

    # If no problems, annotate it with DSSR
    if not c.delete_me:
        c.extract_3D_data()

    # If there were newly discovered problems, add this chain to the known issues
    if c.delete_me and c.chain_label not in known_issues:
        warn(f"Adding {c.chain_label} to known issues.\t\t")
        f = open(path_to_3D_data + "known_issues.txt", 'a')
        f.write(c.chain_label + '\n')
        f.close()
        f = open(path_to_3D_data + "known_issues_reasons.txt", 'a')
        f.write(c.chain_label + '\n' + c.error_messages + '\n\n')
        f.close()
    
    # The Chain object is ready
    return c

def cm_realign(rfam_acc, chains, label):
    """ Runs multiple sequence alignements by RNA family.

    It aligns the Rfam hits from a RNA family with the sequences from the list of chains. 
    Rfam covariance models are used with Infernal tools, except for rRNAs. 
    cmalign requires too much RAM for them, so we use SINA, a specifically designed tool for rRNAs.
    """

    # If the computation was already done before, do not recompute.
    if path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.afa"):
        print(f"\t> {label} completed \t{validsymb}\t(already done)", flush=True)
        return

    if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.fa"):
        print("\t> Extracting sequences...", flush=True)

        # Prepare a FASTA file containing Rfamseq hits for that family + our chains sequences
        f = open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "w")

        # Read the FASTA archive of Rfamseq hits, and add sequences to the file
        with gzip.open(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz", 'rt') as gz:
            ids = []
            for record in SeqIO.parse(gz, "fasta"):
                if record.id not in ids:
                    # Note: here we copy the sequences without modification. 
                    # But, sequences with non ACGU letters exit (W, R, M, Y for example)
                    f.write(">"+record.description+'\n'+str(record.seq)+'\n')
                    ids.append(record.id)

        print("Adding PDB chains...")
        
        # Add the chains sequences to the file
        for c in chains:
            f.write(f"> {str(c)}\n"+c.aligned_seq.replace('-', '').replace('U','T')+'\n') 

        f.close()

    if rfam_acc not in ["RF00177", "RF01960", "RF02540", "RF02541", "RF02543"]: # Ribosomal Subunits
        # Align using Infernal for most RNA families

        # Extracting covariance model for this family
        if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}.cm"):
            print("\t> Extracting covariance model (cmfetch)...", flush=True)
            if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}.cm"):
                f = open(path_to_seq_data + f"realigned/{rfam_acc}.cm", "w")
                subprocess.run(["cmfetch", path_to_seq_data + "Rfam.cm", rfam_acc], stdout=f)
                f.close()

        # Running alignment
        print(f"\t> {label} (cmalign)...", flush=True)
        f = open(path_to_seq_data + f"realigned/{rfam_acc}++.stk", "w")
        subprocess.run(["cmalign", "--mxsize", "2048", path_to_seq_data + f"realigned/{rfam_acc}.cm", path_to_seq_data + f"realigned/{rfam_acc}++.fa"], stdout=f)
        f.close()

        # Converting to aligned Fasta
        print("\t> Converting to aligned FASTA (esl-reformat)...")
        f = open(path_to_seq_data + f"realigned/{rfam_acc}++.afa", "w")
        subprocess.run(["esl-reformat", "afa", path_to_seq_data + f"realigned/{rfam_acc}++.stk"], stdout=f)
        f.close()
        # subprocess.run(["rm", path_to_seq_data + f"realigned/{rfam_acc}.cm", path_to_seq_data + f"realigned/{rfam_acc}++.fa", path_to_seq_data + f"realigned/{rfam_acc}++.stk"])
    else:
        # Ribosomal subunits deserve a special treatment.
        # They require too much RAM to be aligned with Infernal.
        # Then we will use SINA instead.

        # Get the seed alignment from Rfam
        print(f"\t> Download latest LSU/SSU-Ref alignment from SILVA...", end="", flush=True)
        if rfam_acc in ["RF02540", "RF02541", "RF02543"] and not path.isfile(path_to_seq_data + "realigned/LSU.arb"):
            try:
                _urlcleanup()
                _urlretrieve('http://www.arb-silva.de/fileadmin/arb_web_db/release_132/ARB_files/SILVA_132_LSURef_07_12_17_opt.arb.gz', path_to_seq_data + "realigned/LSU.arb.gz")
                print(f"\t{validsymb}", flush=True)
            except:
                print('\n')
                warn(f"Error downloading and/or extracting {rfam_acc}'s seed alignment !\t", error=True)
            print(f"\t\t> Uncompressing LSU.arb...", end='', flush=True)
            subprocess.run(["gunzip", path_to_seq_data + "realigned/LSU.arb.gz"], stdout=subprocess.DEVNULL)
            print(f"\t{validsymb}", flush=True)
        else:
            print(f"\t{validsymb}\t(no need)", flush=True)

        if rfam_acc in ["RF00177", "RF01960"] and not path.isfile(path_to_seq_data + "realigned/SSU.arb"):
            try:
                _urlcleanup()
                _urlretrieve('http://www.arb-silva.de/fileadmin/silva_databases/release_138/ARB_files/SILVA_138_SSURef_05_01_20_opt.arb.gz', path_to_seq_data + "realigned/SSU.arb.gz")
                print(f"\t{validsymb}", flush=True)
            except:
                print('\n')
                warn(f"Error downloading and/or extracting {rfam_acc}'s seed alignment !\t", error=True)
            print(f"\t\t> Uncompressing SSU.arb...", end='', flush=True)
            subprocess.run(["gunzip", path_to_seq_data + "realigned/SSU.arb.gz"], stdout=subprocess.DEVNULL)
            print(f"\t{validsymb}", flush=True)
        else:
            print(f"\t{validsymb}\t(no need)", flush=True)

        if rfam_acc in ["RF00177", "RF01960"]:
            arbfile = "realigned/SSU.arb"
        else:
            arbfile = "realigned/LSU.arb"

        # Run alignment
        print(f"\t> {label} (SINA)...", flush=True)
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

def alignment_nt_stats(f):
    """ Computes Position-Specific-Scoring-Matrices given the multiple sequence alignment of the RNA family.
    
    Also saves every chain of the family to file.
    Uses only 1 core, so this function can be called in parallel.
    """

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
        exit(1)

    # Compute statistics per column
    pssm = BufferingSummaryInfo(align).get_pssm(f, thr_idx)
    frequencies = np.array([ summarize_position(pssm[i]) for i in range(align.get_alignment_length()) ]).T
    del pssm

    # For each sequence, find the right chain and save the PSSMs inside.
    pbar = tqdm(total=len(chains_ids), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} chains", leave=False)
    pbar.update(0)
    for s in align:
        if not '[' in s.id: # this is a Rfamseq entry, not a 3D chain
            continue

        # get the right 3D chain:
        idx = chains_ids.index(s.id)

        # call its method to set its frequencies, and save it
        list_of_chains[idx].set_freqs_from_aln(s.seq, frequencies)
        list_of_chains[idx].save(fformat='csv')

        del list_of_chains[idx]  # saves a bit of memory because of the Chain object sizes
        del chains_ids[idx]      # to keep indexes aligned with list_of_chains
        pbar.update(1)

    pbar.close()

    del rfam_acc_to_download[f] # We won't need this family's chain objects anymore, free up
    idxQueue.put(thr_idx) # replace the thread index in the queue
    return 0

def infer_all_mappings(allmappings, codelist):
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
                else: # otherwise, use the inferred mapping
                    pdb_start = int(inferred_mappings.loc[ (inferred_mappings['rfam_acc'] == rfam) ].pdb_start)
                    pdb_end = int(inferred_mappings.loc[ (inferred_mappings['rfam_acc'] == rfam) ].pdb_end)
                chain_label = f"{pdb_id}_{str(pdb_model)}_{pdb_chain_id}_{pdb_start}-{pdb_end}"
                newchains.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label, rfam=rfam, pdb_start=pdb_start, pdb_end=pdb_end))
    
    return newchains

if __name__ == "__main__":

    # Parse options
    try:
        opts, args = getopt.getopt( sys.argv[1:], 
                                    "r:h", 
                                [   "help", "resolution=", "keep-hetatm=", 
                                    "fill-gaps=", "3d-folder=", "seq-folder=", 
                                    "no-homology", "force-retry" ])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h" or opt == "--help":
            print(  "RNANet, a script to build a multiscale RNA dataset from public data\n"
                    "Developped by Louis Becquey (louis.becquey@univ-evry.fr), 2020")
            print()
            print("Options:")
            print("-h [ --help ]\t\t\tPrint this help message")
            print("--version\t\t\tPrint the program version")
            print()
            print("-r 4.0 [ --resolution=4.0 ]\t(1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 20.0)"
                    "\n\t\t\t\tMinimum 3D structure resolution to consider a RNA chain.")
            print("--keep-hetatm=False\t\t\t(True | False) Keep ions, waters and ligands in produced mmCIF files. "
                    "\n\t\t\t\tDoes not affect the descriptors.")
            print("--fill-gaps=True\t\t(True | False) Replace gaps in sequence due to unresolved residues"
                    "\n\t\t\t\tby the most common nucleotide at this position in the alignment.")
            print("--3d-folder=…\t\t\tPath to a folder to store the 3D data files. Subfolders will contain:"
                    "\n\t\t\t\t\tRNAcifs/\t\tFull structures containing RNA, in mmCIF format"
                    "\n\t\t\t\t\trna_mapped_to_Rfam/\tExtracted 'pure' RNA chains"
                    "\n\t\t\t\t\tannotations/\t\tAnnotations by DSSR"
                    "\n\t\t\t\t\tdatapoints/\t\tFinal results in specified file format.")
            print("--seq-folder=…\t\t\tPath to a folder to store the sequence and alignment files."
                    "\n\t\t\t\t\trfam_sequences/fasta/\tCompressed hits to Rfam families"
                    "\n\t\t\t\t\trealigned/\t\tSequences, covariance models, and alignments by family")
            print("--no-homology\t\t\tDo not try to compute PSSMs and do not align sequences."
                    "\n\t\t\t\tAllows to yield more 3D data (consider chains without a Rfam mapping).")
            print("--force-retry\t\t\tIgnore already known issues, and retry to build them from scratch.")
            sys.exit()

        elif opt == '--version':
            print("RNANet 0.4 alpha ")
            sys.exit()
        elif opt == "-r" or opt == "--resolution":
            assert arg in ["1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "20.0"]
            CRYSTAL_RES = arg
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
            print("Storing 3D data into", path_to_3D_data)
        elif opt=='--seq-folder':
            path_to_seq_data = path.abspath(arg)
            if path_to_seq_data[-1] != '/':
                path_to_seq_data += '/'
            print("Storing sequences into", path_to_seq_data)
        elif opt == "--force-retry":
            USE_KNOWN_ISSUES = False
    
    if path_to_3D_data == "tobedefinedbyoptions" or path_to_seq_data == "tobedefinedbyoptions":
        print("usage: RNANet.py --3d-folder path/where/to/store/chains --seq-folder path/where/to/store/alignments")
        print("See RNANet.py --help for more information.")
        
        path_to_3D_data = "/home/lbecquey/Data/RNA/3D/"
        path_to_seq_data = "/home/lbecquey/Data/RNA/sequences/"
        print(f"\n[DEBUG]\tUsing hard-coded paths to data:\n\t\t{path_to_3D_data}\n\t\t{path_to_seq_data}\n")
        # exit(1)

    # ===========================================================================
    # List 3D chains with available Rfam mapping
    # ===========================================================================

    # List all 3D RNA chains below 4Ang resolution
    full_structures_list = download_BGSU_NR_list()

    # Check for a list of known problems:
    known_issues = []
    if path.isfile(path_to_3D_data + "known_issues.txt"):
        f = open(path_to_3D_data + "known_issues.txt", 'r')
        known_issues = [ x[:-1] for x in f.readlines() ]
        f.close()
        if USE_KNOWN_ISSUES:
            print("\t> Ignoring known issues:")
            for x in known_issues:
                print("\t  ", x)

    all_chains = []
    if HOMOLOGY:
        # Ask Rfam if some are mapped to Rfam families
        allmappings = download_Rfam_PDB_mappings()

        print("> Building list of structures...", flush=True)
        ncores = read_cpu_number()
        p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=ncores)

        pbar = tqdm(full_structures_list, maxinterval=1.0, miniters=1, bar_format="{percentage:3.0f}%|{bar}|")
        for i, newchains in enumerate(p.imap_unordered(partial(infer_all_mappings, allmappings), full_structures_list)): 
            all_chains += newchains
            pbar.update(1) # Everytime the iteration finishes, update the global progress bar

        pbar.close()
        p.close()
        p.join()
                        
    else:
        for codelist in tqdm(full_structures_list):
            codes = str(codelist).replace('+',',').split(',')
            for c in codes:
                nr = c.split('|')
                pdb_id = nr[0].lower()
                pdb_model = int(nr[1])
                pdb_chain_id = nr[2].upper()
                chain_label = f"{pdb_id}_{str(pdb_model)}_{pdb_chain_id}"
                all_chains.append(Chain(pdb_id, pdb_model, pdb_chain_id, chain_label))

    del full_structures_list
    n_chains = len(all_chains)
    print(">", validsymb, n_chains, "RNA chains of interest.")

    # ===========================================================================
    # Download 3D structures, extract the desired chain portions, 
    # and extract their informations
    # ===========================================================================

    print("> Building download list...", flush=True)
    mmcif_parser = MMCIFParser()
    joblist = []
    for c in all_chains:
        if (c.chain_label not in known_issues) or not USE_KNOWN_ISSUES:
            joblist.append(Job(function=build_chain,  # Apply function build_chain to every c.chain_label
                               how_many_in_parallel=ncores, args=[c]))

    # Prepare the results folders
    if not path.isdir(path_to_3D_data + "RNAcifs"):
        os.makedirs(path_to_3D_data + "RNAcifs")    # for the whole structures
    if HOMOLOGY and not path.isdir(path_to_3D_data + "rna_mapped_to_Rfam"):
        os.makedirs(path_to_3D_data + "rna_mapped_to_Rfam") # for the portions mapped to Rfam
    if not HOMOLOGY and not path.isdir(path_to_3D_data + "rna_only"):
        os.makedirs(path_to_3D_data + "rna_only") # extract chains of pure RNA
    if not path.isdir(path_to_3D_data+"annotations"):
        os.makedirs(path_to_3D_data+"annotations") # for the annotations by DSSR

    # Run the builds and extractions
    results = execute_joblist(joblist)[1]

    # Remove the chains whose parsing resulted in errors
    loaded_chains = [ c for c in results if not c.delete_me ]

    print(f"> Loaded {len(loaded_chains)} RNA chains ({len(all_chains) - len(loaded_chains)} errors).")
    del all_chains # Here ends its utility, so let's free some memory
    del joblist
    del results

    if not HOMOLOGY:
        # Save chains to file
        for c in loaded_chains:
            c.data.to_csv(path_to_3D_data + "datapoints/" + c.chain_label)
        print("Completed.")
        exit()

    # ===========================================================================
    # Download RNA sequences of the corresponding Rfam families
    # ===========================================================================
    
    # Preparing a results folder
    if not os.access(path_to_seq_data + "realigned/", os.F_OK):
        os.makedirs(path_to_seq_data + "realigned/")

    # Get the list of Rfam families found
    rfam_acc_to_download = {}
    mappings_list = {}
    for c in loaded_chains:
        if c.rfam_fam not in rfam_acc_to_download:
            rfam_acc_to_download[c.rfam_fam] = [ c ]
            mappings_list[c.rfam_fam] = [ c.chain_label ]
        else:
            rfam_acc_to_download[c.rfam_fam].append(c)
            mappings_list[c.rfam_fam].append(c.chain_label)
    pd.DataFrame.from_dict(mappings_list, orient='index').transpose().to_csv(path_to_seq_data + "realigned/mappings_list.csv")
    del mappings_list
    print(f"> Identified {len(rfam_acc_to_download.keys())} families to download and re-align with the crystals' sequences:")

    # Download the covariance models for all families
    download_Rfam_cm()

    # Ask the SQL server how much we have to download for each family
    fam_stats = download_Rfam_family_stats(rfam_acc_to_download.keys())
    fam_list = sorted(rfam_acc_to_download.keys())
    if len(fam_stats.keys()): # 'if' protected, for the case the server is down,  fam_stats is empty
        # save the statistics to CSV file
        n_pdb = [ len(rfam_acc_to_download[f]) for f in fam_stats["rfam_acc"] ]
        fam_stats["n_pdb_seqs"] = n_pdb
        fam_stats["total_seqs"] = fam_stats["n_seq"] + fam_stats["n_pdb_seqs"]
        fam_stats.to_csv(path_to_seq_data + "realigned/statistics.csv")
        # print the stats
        for f in fam_list:
            line = fam_stats[fam_stats["rfam_acc"]==f]
            print(f"\t> {f}: {line.n_seq.values[0]} Rfam hits + {line.n_pdb_seqs.values[0]} PDB sequences to realign")
    del fam_stats

    # Download the sequences
    for f in fam_list:
        download_Rfam_sequences(f)

    # ==========================================================================================
    # Realign sequences from 3D chains to Rfam's identified hits (--> extended full alignement)
    # ==========================================================================================

    # Prepare the job list
    fulljoblist = []
    for f in fam_list:
        label = f"Realign {f} + {len(rfam_acc_to_download[f])} chains"
        fulljoblist.append( Job( function=cm_realign, args=[f, rfam_acc_to_download[f], label],  # Apply cm_realign to each RNA family
                                 how_many_in_parallel=1, label=label))  # the function already uses all CPUs so launch them one by one
    
    # Execute the jobs
    execute_joblist(fulljoblist, printstats=True) # printstats=True will show a summary of time/memory usage of the jobs
    del fulljoblist

    # ==========================================================================================
    # Now compute statistics on base variants at each position of every 3D chain
    # ==========================================================================================

    print("Computing nucleotide frequencies in alignments...")

    # Prepare a results folder
    if not path.isdir(path_to_3D_data + "datapoints/"):
        os.makedirs(path_to_3D_data + "datapoints/")

    # Prepare the architecture of a shiny multi-progress-bars design
    thr_idx_mgr = Manager()                 # Push the number of workers to a queue. 
    idxQueue = thr_idx_mgr.Queue()          # ... Then each Pool worker will
    for i in range(ncores):                 # ... pick a number from the queue when starting the computation for one family, 
        idxQueue.put(i)                     # ... and replace it when the computation has ended so it could be picked up later.

    # Start a process pool to dispatch the RNA families,
    # over multiple CPUs (one family by CPU)
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=int(ncores/2))

    fam_pbar = tqdm(total=len(fam_list), desc="RNA families", position=0, leave=True) 
    for i, _ in enumerate(p.imap_unordered(alignment_nt_stats, fam_list)): # Apply alignment_nt_stats to each RNA family
        fam_pbar.update(1) # Everytime the iteration finishes on a family, update the global progress bar over the RNA families

    fam_pbar.close()
    p.close()
    p.join()

    print("Completed.")  # This part of the code is supposed to release some serotonin in the modeller's brain

    # # so i can sleep for the end of the night
    # subprocess.run(["shutdown","now"]) 
