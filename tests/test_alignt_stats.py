#!/usr/bin/python3.8
import numpy as np
import pandas as pd
import Bio.PDB.StructureBuilder, json, os, psutil, subprocess, sys, time
from Bio import AlignIO, SeqIO
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
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

def warn(message, error=False):
    """Pretty-print warnings and error messages.
    """
    if error:
        print(f"\t> \033[31mERR: {message}\033[0m{errsymb}", flush=True)
    else:
        print(f"\t> \033[33mWARN: {message}\033[0m{warnsymb}", flush=True)

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


def read_cpu_number():
    # As one shall not use os.cpu_count() on LXC containers,
    # because it reads info from /sys wich is not the VM resources but the host resources.
    # This function reads it from /proc/cpuinfo instead.
    p = subprocess.run(['grep', '-Ec', '(Intel|AMD)', '/proc/cpuinfo'], stdout=subprocess.PIPE)
    return int(int(p.stdout.decode('utf-8')[:-1])/2)

def cm_realign(rfam_acc, chains, label):
    """ Runs multiple sequence alignements by RNA family.

    It aligns the Rfam hits from a RNA family with the sequences from the list of chains. 
    Rfam covariance models are used with Infernal tools, except for rRNAs. 
    cmalign requires too much RAM for them, so we use SINA, a specifically designed tool for rRNAs.
    """

    # If the computation was already done before, do not recompute.
    if path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.afa"):
        print(f"\t> {label} completed \t\t{validsymb}\t(already done)", flush=True)
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
    run_dir = path.abspath(os.getcwd())
    if not path.isfile(run_dir + "/tests/RF00177.npz"):
        pbar = tqdm(iterable=range(alilen), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f}", leave=False)
        results = [ summarize_position(align[:,i]) for i in pbar ]
        pbar.close()
        frequencies = np.array(results).T
        np.savez(run_dir + "/tests/RF00177.npz", freqs= frequencies)
    else:
        frequencies = np.load(run_dir + "/tests/RF00177.npz")['freqs']

    # For each sequence, find the right chain and save the PSSMs inside.
    pbar = tqdm(total=len(chains_ids), position=thr_idx+1, desc=f"Worker {thr_idx+1}: {f} chains", leave=False)
    pbar.update(0)
    for s in align:
        if not '[' in s.id: # this is a Rfamseq entry, not a 3D chain
            continue
        
        if not s.id in chains_ids: # skip other RF00177 chains, keep only our test example (5wnt)
            continue

        # get the right 3D chain:
        idx = chains_ids.index(s.id)
        list_of_chains[idx].set_freqs_from_aln(s.seq, frequencies)
        pbar.update(1)
    pbar.close()


    idxQueue.put(thr_idx) # replace the thread index in the queue
    return 0

ncores = read_cpu_number()

c = Chain("5WNT|1|A")
c.chain_label = f"{c.pdb_id}_{str(c.pdb_model)}_{c.pdb_chain_id}_2-1520"
c = build_chain(c, "RF00177", 2, 1520)
rfam_acc_to_download = { c.rfam_fam:[c] }
cm_realign("RF00177", rfam_acc_to_download["RF00177"], "Realign RF00177 + 1 chains")

thr_idx_mgr = Manager()                
idxQueue = thr_idx_mgr.Queue()          
for i in range(ncores):                 
    idxQueue.put(i)                     

alignment_nt_stats("RF00177")