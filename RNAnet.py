#!/usr/bin/python3.8
import numpy as np
import pandas as pd
import concurrent.futures, Bio.PDB.StructureBuilder, copy, gzip, io, json, os, psutil, re, requests, sqlalchemy, subprocess, sys, time, warnings
from Bio import AlignIO, SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio._py3k import urlretrieve as _urlretrieve
from Bio._py3k import urlcleanup as _urlcleanup
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from collections import OrderedDict
from functools import partial
from os import path, makedirs
from multiprocessing import Pool, Manager
from time import sleep
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

if path.isdir("/home/ubuntu/"): # this is the IFB-core cloud
    path_to_3D_data = "/mnt/Data/RNA/3D/"
    path_to_seq_data = "/mnt/Data/RNA/sequences/"
elif path.isdir("/home/persalteas"): # this is my personal workstation
    path_to_3D_data = "/home/persalteas/Data/RNA/3D/"
    path_to_seq_data = "/home/persalteas/Data/RNA/sequences/"
elif path.isdir("/home/lbecquey"): # this is the IBISC server
    path_to_3D_data = "/home/lbecquey/Data/RNA/3D/"
    path_to_seq_data = "/home/lbecquey/Data/RNA/sequences/"
elif path.isdir("/nhome/siniac/lbecquey"): # this is the office PC
    path_to_3D_data = "/nhome/siniac/lbecquey/Data/RNA/3D/"
    path_to_seq_data = "/nhome/siniac/lbecquey/Data/RNA/sequences/"
else:
    print("I don't know that machine... I'm shy, maybe you should introduce yourself ?")
    exit(1)

m = Manager()
running_stats = m.list()
running_stats.append(0) # n_launched
running_stats.append(0) # n_finished
running_stats.append(0) # n_skipped
runDir = path.dirname(path.realpath(__file__))

validsymb = '\U00002705'
warnsymb = '\U000026A0'
errsymb = '\U0000274C'

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
            return 0         

        # I don't really know what this is but the doc said:          
        if icode != " ":
            warn(f"icode {icode} at position {resseq}\t\t")

        # Accept the residue if it is in the right interval:
        return int(self.start <= resseq <= self.end)

    def accept_atom(self, atom):

        # Refuse hydrogens
        if self.hydrogen_regex.match(atom.get_id()):
            return 0 

        # Accept all atoms otherwise.
        return 1


class Chain:
    """ The object which stores all our data and the methods to process it.

    Chains accumulate information through this scipt, and are saved to files at the end of major steps."""

    def __init__(self, nrlist_code):
        nr = nrlist_code.split('|')
        self.pdb_id = nr[0].lower()             # PDB ID
        self.pdb_model = int(nr[1])             # model ID, starting at 1
        self.pdb_chain_id = nr[2].upper()       # chain ID (mmCIF), multiple letters
        self.reversed = False                   # wether pdb_end > pdb_start in the Rfam mapping
        self.chain_label = ""                   # chain pretty name 
        self.full_mmCIFpath = ""                # path to the source mmCIF structure
        self.file = ""                          # path to the 3D PDB file
        self.rfam_fam = ""                      # mapping to an RNA family
        self.seq = ""                           # sequence with modified nts
        self.aligned_seq = ""                   # sequence with modified nts replaced, but gaps can exist
        self.length = -1                        # length of the sequence (missing residues are not counted)
        self.full_length = -1                   # length of the chain extracted from source structure ([start; stop] interval)
        self.delete_me = False                  # an error occured during production/parsing
        self.error_messages = ""                # Error message(s) if any
        self.frequencies = np.zeros((5,0))      # frequencies of nt at every position: A,C,G,U,Other
        self.data3D = None                      # Pandas DataFrame with all the 3D data extracted by DSSR.

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

    def extract_portion(self, filename, pdb_start, pdb_end):
        """ Extract the part which is mapped to Rfam from the main CIF file and save it to another file.
        """
        
        status = f"\t> Extract {pdb_start}-{pdb_end} atoms from {self.pdb_id}-{self.pdb_chain_id}\t"
        self.file = path_to_3D_data+"rna_mapped_to_Rfam/"+filename+".cif"

        # Check if file exists, if yes, abort (do not recompute)
        if os.path.exists(self.file):
            print(status + f"\t{validsymb}\t(already done)", flush=True)
            return

        model_idx = self.pdb_model - (self.pdb_model > 0) # because arrays start at 0, models start at 1
        pdb_start = int(pdb_start)
        pdb_end = int(pdb_end)

       
        with warnings.catch_warnings():
            # TODO: check if this with and warnings catch is still useful since i moved to CIF files
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

    def set_rfam(self, rfam):
        """ Rember the Rfam mapping for this chain.
        """

        self.rfam_fam = rfam

    def extract_3D_data(self):
        """ Runs DSSR to annotate the 3D chain and get various information about it. """

        # Check if the file exists. If no, compute it.
        if not os.path.exists(path_to_3D_data+f"pseudotorsions/{self.chain_label}.csv"):

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
                df = df.drop(['summary', 'chain_name', 'index',
                               'v0', 'v1', 'v2', 'v3', 'v4', 'splay_angle',
                               'splay_distance', 'splay_ratio', 'sugar_class',
                               'amplitude', 'phase_angle'], axis=1)
                df['P_x'] = [ float(i[0]) if i[0] is not None else np.NaN for i in df['P_xyz'] ]                #
                df['P_y'] = [ float(i[1]) if i[1] is not None else np.NaN for i in df['P_xyz'] ]                #
                df['P_z'] = [ float(i[2]) if i[2] is not None else np.NaN for i in df['P_xyz'] ]                # Flatten the 
                df['C5prime_x'] = [ float(i[0]) if i[0] is not None else np.NaN for i in df['C5prime_xyz'] ]    # Python dictionary
                df['C5prime_y'] = [ float(i[1]) if i[1] is not None else np.NaN for i in df['C5prime_xyz'] ]    #
                df['C5prime_z'] = [ float(i[2]) if i[2] is not None else np.NaN for i in df['C5prime_xyz'] ]    #

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
                res_ids = list(df['nt_id'])
                paired = [ 0 ] * l
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
                            paired[nt1_idx] = nt2_idx + 1
                            paired[nt2_idx] = nt1_idx + 1
                            interacts[nt1_idx] += 1
                            interacts[nt2_idx] += 1
                            pair_type_LW[nt1_idx] = p["LW"]
                            pair_type_LW[nt2_idx] = p["LW"]
                            pair_type_DSSR[nt1_idx] = p["DSSR"]
                            pair_type_DSSR[nt2_idx] = p["DSSR"]
                        elif nt1 in res_ids:
                            nt1_idx = res_ids.index(nt1)
                            interacts[nt1_idx] += 1
                        elif nt2 in res_ids:
                            nt2_idx = res_ids.index(nt2)
                            interacts[nt2_idx] += 1
                df['paired'] = paired
                df['pair_type_LW'] = pair_type_LW
                df['pair_type_DSSR'] = pair_type_DSSR

                # Iterate over multiplets to identify base-base interactions
                if "multiplets" in json_object.keys():
                    multiplets = json_object["multiplets"]
                    for m in multiplets:
                        nts = m["nts_long"].split(',')
                        # iterate  over the nts of a multiplet
                        for j, nt in enumerate(nts):

                            # if the nt is in that chain:
                            if nt in res_ids:
                                i = res_ids.index(nt)
                                # iterate over those other nts
                                for o in nts[:j]+nts[j+1:]:
                                    if o in res_ids and str(res_ids.index(o)+1) not in str(df['paired'][i]): # and it's not already in 'paired'
                                        df.loc[i,'paired'] = str(df['paired'][i]) + ',' + str(res_ids.index(o)+1)
                                interacts[i] = len(str(df['paired'][i]).split(','))
                            
                df['Ninteract'] = interacts

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
            df.to_csv(path_to_3D_data + f"pseudotorsions/{self.chain_label}.csv")
            del df
            print("\t> Saved", self.chain_label, f"pseudotorsions to CSV.\t\t{validsymb}", flush=True)
        else:
            print("\t> Computing", self.chain_label, f"pseudotorsions...\t{validsymb}\t(already done)", flush=True)

        # Now load data from the CSV file
        d = pd.read_csv(path_to_3D_data+f"pseudotorsions/{self.chain_label}.csv", index_col=0)
        self.seq = "".join(d.nt_code.values)
        self.aligned_seq = "".join(d.nt_align_code.values)
        self.length = len([ x for x in self.aligned_seq if x != "-" ])
        self.full_length = len(d.nt_code)
        self.data3D = d
        print(f"\t> Loaded data from CSV\t\t\t\t{validsymb}", flush=True)

        # Remove too short chains
        if self.length < 5:
            warn(f"{self.chain_label} sequence is too short, let's ignore it.\t", error=True)
            self.delete_me = True
            self.error_messages = "Sequence is too short. (< 5 resolved nts)"
        return

    def set_freqs_from_aln(self, s_seq, freqs):
        """Maps the object's sequence to its version in a MSA, to compute nucleotide frequencies at every position.
        
        s_seq: the aligned version of self.aligned_seq
        freqs: the nucleotide frequencies at every position of s_seq
        This also replaces gaps by the most common nucleotide.
        """
        alilen = len(s_seq)

        # Save colums in the appropriate positions
        i = 0
        j = 0
        while i<self.full_length and j<alilen:
            # Here we try to map self.aligned_seq (the sequence of the 3D chain, including gaps when residues are missing), 
            # with s_seq, the sequence aligned in the MSA, containing any of ACGU and two types of gaps, - and .

            if self.aligned_seq[i] == s_seq[j].upper(): # alignment and sequence correspond (incl. gaps)
                self.frequencies = np.concatenate((self.frequencies, freqs[:,j].reshape(-1,1)), axis=1)
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
                    self.frequencies = np.concatenate((self.frequencies, freqs[:,j].reshape(-1,1)), axis=1)
                    i += 1
                    j += 1
                    continue

                # else, just ignore the gap.
                self.frequencies = np.concatenate((self.frequencies, np.array([0.0,0.0,0.0,0.0,1.0]).reshape(-1,1)), axis=1)
                i += 1
            elif s_seq[j] in ['.', '-']: # gap in the alignment, but not in the real chain
                j += 1 # ignore the column
            else: # sequence mismatch which is not a gap...
                print(f"You are never supposed to reach this. Comparing {self.chain_label} in {i} ({self.aligned_seq[i-1:i+2]}) with seq[{j}] ({s_seq[j-3:j+4]}).\n", 
                        self.aligned_seq, 
                        sep='', flush=True)
                exit(1)

        # Replace gapped positions by the consensus sequence:
        c_aligned_seq = list(self.aligned_seq)
        c_seq = list(self.seq)
        letters = ['A', 'C', 'G', 'U', 'N']
        for i in range(self.full_length):
            if c_aligned_seq[i] == '-':      # (then c_seq[i] also is)
                freq = self.frequencies[:,i]
                l = letters[freq.tolist().index(max(freq))]
                c_aligned_seq[i] = l
                c_seq[i] = l
                self.data3D.iloc[i,3] = l # self.data3D['nt_code'][i]
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
            point[6,i] = self.frequencies[0, i]
            point[7,i] = self.frequencies[1, i]
            point[8,i] = self.frequencies[2, i]
            point[9,i] = self.frequencies[3, i]
            point[10,i] = self.frequencies[4, i]
        
        self.data3D = pd.concat([self.data3D, pd.DataFrame(point.T, columns=["position","is_A","is_C","is_G","is_U","is_other","freq_A","freq_C","freq_G","freq_U","freq_other"])], axis=1)

        # save to file
        self.data3D.to_csv(path_to_3D_data + "datapoints/" + self.chain_label)


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

    print("> Fetching latest NR list from BGSU website...", end='', flush=True)
    # Download latest BGSU non-redundant list
    try:
        s = requests.get("http://rna.bgsu.edu/rna3dhub/nrlist/download/current/4.0A/csv").content
        nr = open(path_to_3D_data + "latest_nr_list.csv", 'w')
        nr.write("class,representative,class_members\n")
        nr.write(io.StringIO(s.decode('utf-8')).getvalue())
        nr.close()
    except:
        warn("Error downloading NR list !\t", error=True)

        # Try to read previous file
        if path.isfile(path_to_3D_data + "latest_nr_list.csv"):
            print("\t> Use of the previous version.\t", end = "", flush=True)
        else:
            return [], []

    nrlist = pd.read_csv(path_to_3D_data + "latest_nr_list.csv")
    full_structures_list = nrlist['class_members'].tolist()
    print(f"\t{validsymb}", flush=True)

    # Split the codes
    all_chains = []
    for code in full_structures_list:
        codes = code.replace('+',',').split(',')
        for c in codes:
            # Convert every PDB code into a Chain object
            all_chains.append(Chain(c))

    # The beginning of an adventure.
    return all_chains

def build_chain(c, rfam, pdb_start, pdb_end):
    """ Additionally adds all the desired information to a Chain object.

    """
    # Download the whole mmCIF file containing the chain we are interested in
    c.download_3D()

    # If no problems, extract the portion we want
    if not c.delete_me:
        c.extract_portion(c.chain_label, pdb_start, pdb_end)

    # If no problems, map it to an Rfam family, and annotate it with DSSR
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
                    f.write(">"+record.description+'\n'+str(record.seq)+'\n')
                    ids.append(record.id)

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

def summarize_position(col):
    """ Counts the number of nucleotides at a given position, given a "column" from a MSA.
    """

    # Count the different chars in the column
    counts = { 'A':col.count('A'), 'C':col.count('C'), 
               'G':col.count('G'), 'U':col.count('U'), 
               '-':col.count('-'), '.':col.count('.') }

    # Count modified nucleotides
    known_chars_count = 0
    chars = set(col)
    for char in chars:
        if char in "ACGU":
            known_chars_count += counts[char]
        # elif char not in "-.":
            # counts[char] = col.count(char)
    N = len(col) - counts['-'] - counts['.'] # number of ungapped residues

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
        alilen = align.get_alignment_length()
    except:
        warn(f"{f}'s alignment is wrong. Recompute it and retry.", error=True)
        exit(1)

    # Compute statistics per column
    pbar = tqdm(iterable=range(alilen), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f}", leave=False)
    results = [ summarize_position(align[:,i]) for i in pbar ]
    pbar.close()
    frequencies = np.array(results).T

    # For each sequence, find the right chain and save the PSSMs inside.
    pbar = tqdm(total=len(chains_ids), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} chains", leave=False)
    pbar.update(0)
    for s in align:
        if not '[' in s.id: # this is a Rfamseq entry, not a 3D chain
            continue

        # get the right 3D chain:
        idx = chains_ids.index(s.id)
        list_of_chains[idx].set_freqs_from_aln(s.seq, frequencies)
        pbar.update(1)
    pbar.close()


    idxQueue.put(thr_idx) # replace the thread index in the queue
    return 0

if __name__ == "__main__":
    print("Main process running. (PID", os.getpid(), ")")

    # # temporary, for debugging: start from zero knowledge
    # if os.path.exists(path_to_3D_data + "known_issues.txt"):
    #     subprocess.run(["rm", path_to_3D_data + "known_issues.txt"])

    # ===========================================================================
    # List 3D chains with available Rfam mapping
    # ===========================================================================

    # List all 3D RNA chains below 4Ang resolution
    all_chains = set(download_BGSU_NR_list())

    # Ask Rfam if some are mapped to Rfam families
    mappings = download_Rfam_PDB_mappings()

    # Filter the chains with mapping
    chains_with_mapping = []
    for c in all_chains:
        mapping = mappings.loc[ (mappings.pdb_id == c.pdb_id) & (mappings.chain == c.pdb_chain_id) ]
        n = len(mapping.rfam_acc.values)
        for j in range(n):
            if j == n-1:
                chains_with_mapping.append(c)
            else:
                chains_with_mapping.append(copy.deepcopy(c))
            chains_with_mapping[-1].set_rfam(mapping.rfam_acc.values[j])
    n_chains = len(chains_with_mapping)

    # ===========================================================================
    # Download 3D structures, extract the desired chain portions, 
    # and extract their informations
    # ===========================================================================

    print("> Building download list...", flush=True)

    # Check for a list of known problems:
    known_issues = []
    if path.isfile(path_to_3D_data + "known_issues.txt"):
        f = open(path_to_3D_data + "known_issues.txt", 'r')
        known_issues = [ x[:-1] for x in f.readlines() ]
        f.close()
        print("\t> Ignoring known issues:")
        for x in known_issues:
            print("\t  ", x)

    mmcif_parser = MMCIFParser()
    joblist = []
    for c in chains_with_mapping:

        # read mappings information
        mapping = mappings.loc[ (mappings.pdb_id == c.pdb_id) & (mappings.chain == c.pdb_chain_id) & (mappings.rfam_acc == c.rfam_fam) ]
        pdb_start = str(mapping.pdb_start.values[0])
        pdb_end = str(mapping.pdb_end.values[0])

        # Add a job to build the chain to the list
        c.chain_label = f"{c.pdb_id}_{str(c.pdb_model)}_{c.pdb_chain_id}_{pdb_start}-{pdb_end}"
        ncores = read_cpu_number()
        if c.chain_label not in known_issues:
            joblist.append(Job(function=build_chain,  # Apply function build_chain to every c.chain_label
                               how_many_in_parallel=ncores,
                               args=[c, mapping.rfam_acc.values[0], pdb_start, pdb_end]))

    # Prepare the results folders
    if not path.isdir(path_to_3D_data + "RNAcifs"):
        os.makedirs(path_to_3D_data + "RNAcifs")    # for the whole structures
    if not path.isdir(path_to_3D_data + "rna_mapped_to_Rfam"):
        os.makedirs(path_to_3D_data + "rna_mapped_to_Rfam") # for the portions mapped to Rfam
    if not path.isdir(path_to_3D_data+"pseudotorsions/"):
        os.makedirs(path_to_3D_data+"pseudotorsions/") # for the annotations by DSSR

    # Run the builds and extractions
    results = execute_joblist(joblist)[1]

    # Remove the chains whose parsing resulted in errors
    loaded_chains = [ c for c in results if not c.delete_me ]

    print(f"> Loaded {len(loaded_chains)} RNA chains ({len(chains_with_mapping) - len(loaded_chains)} errors).")

    # ===========================================================================
    # Download RNA sequences of the corresponding Rfam families
    # ===========================================================================

    # Preparing a results folder
    if not os.access(path_to_seq_data + "realigned/", os.F_OK):
        os.makedirs(path_to_seq_data + "realigned/")

    # Get the list of Rfam families found
    rfam_acc_to_download = {}
    for c in loaded_chains:
        if c.rfam_fam not in rfam_acc_to_download:
            rfam_acc_to_download[c.rfam_fam] = [ c ]
        else:
            rfam_acc_to_download[c.rfam_fam].append(c)
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
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=ncores)

    fam_pbar = tqdm(total=len(fam_list), desc="RNA families", position=0, leave=True) 
    for i, _ in enumerate(p.imap_unordered(alignment_nt_stats, fam_list)): # Apply alignment_nt_stats to each RNA family
        fam_pbar.update(1) # Everytime the iteration finishes on a family, update the global progress bar over the RNA families

    fam_pbar.close()
    p.close()
    p.join()

    print("Completed.")  # This part of the code is supposed to release some serotonin in the modeller's brain

    # # so i can sleep for the end of the night
    # subprocess.run(["shutdown","now"]) 
