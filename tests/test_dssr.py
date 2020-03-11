#!/usr/bin/python3.8
import numpy as np
import pandas as pd
import concurrent.futures, Bio.PDB.StructureBuilder, gzip, io, json, os, psutil, re, requests, sqlalchemy, subprocess, sys, time, warnings
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


validsymb = '\U00002705'
warnsymb = '\U000026A0'
errsymb = '\U0000274C'

def warn(message, error=False):
    """Pretty-print warnings and error messages.
    """
    if error:
        print(f"\t> \033[31mERR: {message}\033[0m{errsymb}", flush=True)
    else:
        print(f"\t> \033[33mWARN: {message}\033[0m{warnsymb}", flush=True)

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
        self.aligned_seq = ""                   # sequence with modified nts replaced 
        self.length = -1                        # length of the sequence (missing residues are not counted)
        self.full_length = -1                   # length of the chain extracted from source structure ([start; stop] interval)
        self.delete_me = False                  # an error occured during production/parsing
        self.delete_reason = ""                 # Error message(s) if any
        self.frequencies = np.zeros((5,0))      # frequencies of nt at every position: A,C,G,U,Other

    def __str__(self):
        return self.pdb_id + '[' + str(self.pdb_model) + "]-" + self.pdb_chain_id

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
            self.delete_reason = f"Error downloading {url}"

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
                self.delete_reason = f"Error with CIF file of {self.pdb_id} !"
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
        print("\t> Associating it to", rfam, f"...\t\t\t{validsymb}")

    def extract_3D_data(self):
        """ Runs DSSR to annotate the 3D chain and get various information about it. """

        # Check if the file exists. If no, compute it.
        if not os.path.exists(path_to_3D_data+f"pseudotorsions/{self.chain_label}.csv"):

            # run DSSR (you need to have it in your $PATH, follow x3dna installation instructions)
            output = subprocess.run(
                ["x3dna-dssr", f"-i={self.file}", "--json", "--auxfile=no"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = output.stdout.decode('utf-8') # this contains the results in JSON format, or is empty if there are errors
            stderr = output.stderr.decode('utf-8') # this contains the evenutal errors


            # try:
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
            df = df.drop(['summary', 'frame', 'chain_name', 'index',
                            'v0', 'v1', 'v2', 'v3', 'v4', 'splay_angle',
                            'splay_distance', 'splay_ratio', 'sugar_class',
                            'amplitude', 'phase_angle' ], axis=1)
            df['P_x'] = [ float(i[0]) if i[0] is not None else np.NaN for i in df['P_xyz'] ]                #
            df['P_y'] = [ float(i[1]) if i[1] is not None else np.NaN for i in df['P_xyz'] ]                #
            df['P_z'] = [ float(i[2]) if i[2] is not None else np.NaN for i in df['P_xyz'] ]                # Flatten the 
            df['C5prime_x'] = [ float(i[0]) if i[0] is not None else np.NaN for i in df['C5prime_xyz'] ]    # Python dictionary
            df['C5prime_y'] = [ float(i[1]) if i[1] is not None else np.NaN for i in df['C5prime_xyz'] ]    #
            df['C5prime_z'] = [ float(i[2]) if i[2] is not None else np.NaN for i in df['C5prime_xyz'] ]    #

            # Add a sequence column just for the alignments
            df['nt_align_code'] = [ str(x).upper()
                                            .replace('U','T')         # We align as DNA
                                            .replace('NAN', '-')      # Unresolved nucleotides are gaps
                                            .replace('?','-')         # Unidentified residues, let's delete them
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

                # if 1+df.iloc[-1,0] not in df['nt_resnum'] - resnum_start +1 and  1+df.iloc[-1,0] not in diff:
                #     diff.add(1+df.iloc[-1,0])

                for i in sorted(diff):
                    df = pd.concat([df.iloc[:i-1], pd.DataFrame({"index_chain": i, "nt_resnum": i+resnum_start-1, "nt_code":'-', "nt_name":'-', 'nt_align_code':'-'}, index=[i-1]), df.iloc[i-1:]])
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
            # except KeyError as e:
            #     # Mostly, there are no part about nucleotides in the DSSR output. Abort.
            #     warn(f"Error while parsing DSSR's json output:\n{e}\n\tignoring {self.chain_label}\t\t\t\t", error=True)
            #     self.delete_me = True
            #     self.error_messages = f"Error while parsing DSSR's json output:\n{e}"
            #     return
            
            # Creating a df for easy saving to CSV
            df.to_csv(path_to_3D_data + f"pseudotorsions/{self.chain_label}.csv")
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
        c.set_rfam(rfam)
        c.extract_3D_data()

    # The Chain object is ready
    return c

subprocess.run(["rm", "-f", path_to_3D_data + f"pseudotorsions/4w2e_1_A_1-2912.csv"])
mmcif_parser = MMCIFParser()
thr_idx_mgr = Manager()                  
idxQueue = thr_idx_mgr.Queue()          
idxQueue.put(0)                    

c = Chain("4W2E|1|A")
c.chain_label = "4w2e_1_A_1-2912"
build_chain(c, "RF02541", 1, 2912)
