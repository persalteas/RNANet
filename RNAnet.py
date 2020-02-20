#!/usr/bin/python3.8
import numpy as np
import pandas as pd
import concurrent.futures, Bio.PDB.StructureBuilder, gzip, io, itertools, json, multiprocessing, os, psutil, re, requests, sqlalchemy, subprocess, sys, time, warnings
from Bio import AlignIO, SeqIO
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException
from Bio._py3k import urlretrieve as _urlretrieve
from Bio._py3k import urlcleanup as _urlcleanup
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from collections import OrderedDict
from ftplib import FTP
from functools import partial
from os import path, makedirs
from multiprocessing import Pool, cpu_count, Manager
from time import sleep
from tqdm import tqdm

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
hydrogen = re.compile("[123 ]*H.*")
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

    def __init__(self, model_id, chain_id, start, end):
        self.chain_id = chain_id
        self.start = start
        self.end = end
        self.pdb_model_id = model_id

    def accept_model(self, model):
        if model.get_id() == self.pdb_model_id:
            return 1
        return 0

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return 1
        return 0

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()
        if hetatm_flag in ["W", "H_MG"]:
            return 0 # skip waters and magnesium
        if icode != " ":
            warn(f"icode {icode} at position {resseq}\t\t")
        if self.start <= resseq <= self.end:
            return 1
        return 0

    def accept_atom(self, atom):
        name = atom.get_id()
        if hydrogen.match(name):
            return 0 # Get rid of hydrogens
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
        self.seq = ""                           # sequence
        self.length = -1                        # length of the sequence (missing residues are not counted)
        self.full_length = -1                   # length of the chain extracted from source structure ([start; stop] interval)
        self.delete_me = False                  # an error occured during production/parsing
        self.frequencies = np.zeros((5,0))      # frequencies of nt at every position: A,C,G,U,Other
        self.etas = []                          # eta' pseudotorsion at every position, if available
        self.thetas = []                        # theta' pseudotorsion at every position, if available
        self.mask = []                          # nucleotide is available in 3D (1 or 0), for every position

    def __str__(self):
        return self.pdb_id + '[' + str(self.pdb_model) + "]-" + self.pdb_chain_id

    def download_3D(self):
        status = f"\t> Download {self.pdb_id}.cif\t\t\t"
        url = 'http://files.rcsb.org/download/%s.cif' % (self.pdb_id)
        if not os.access(path_to_3D_data+"RNAcifs", os.F_OK):
            os.makedirs(path_to_3D_data+"RNAcifs")
        final_filepath = path_to_3D_data+"RNAcifs/"+self.pdb_id+".cif"

        if os.path.exists(final_filepath):
            print(status + f"\t{validsymb}\t(structure exists)")
            self.full_mmCIFpath = final_filepath
            return
        else:
            try:
                _urlcleanup()
                _urlretrieve(url, final_filepath)
                self.full_mmCIFpath = final_filepath
                print(status + f"\t{validsymb}")
            except IOError:
                print(status + f"\tERR \U0000274E\t\033[31mError downloading {url} !\033[0m")
                self.delete_me = True

    def extract_portion(self, filename, pdb_start, pdb_end):
        status = f"\t> Extract {pdb_start}-{pdb_end} atoms from {self.pdb_id}-{self.pdb_chain_id}\t"
        self.file = path_to_3D_data+"rna_mapped_to_Rfam/"+filename+".cif"

        if os.path.exists(self.file):
            print(status + f"\t{validsymb}\t(already done)", flush=True)
            return

        model_idx = self.pdb_model - (self.pdb_model > 0) # arrays start at 0, models start at 1
        pdb_start = int(pdb_start)
        pdb_end = int(pdb_end)

        # Load the whole mmCIF into a Biopython structure object:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            if self.full_mmCIFpath == "":
                print(status + f"\t\U0000274E\t\033[31mError with CIF file of {self.pdb_id} !\033[0m", flush=True)
                self.delete_me = True
                return
            s = mmcif_parser.get_structure(self.pdb_id, self.full_mmCIFpath)

            c = s[model_idx][self.pdb_chain_id]             # the desired chain
            first_number = c.child_list[0].get_id()[1]      # its first residue is numbered 'first_number'
            if pdb_start < pdb_end:
                start = pdb_start + first_number - 1            # then our start position should be shifted by 'first_number'
                end = pdb_end + first_number - 1                # same for the end position
            else:
                self.reversed = True
                end = pdb_start + first_number - 1
                start = pdb_end + first_number - 1

            # Do the extraction of the 3D data
            sel = NtPortionSelector(model_idx, self.pdb_chain_id, start, end)
            ioobj = MMCIFIO()
            ioobj.set_structure(s)
            ioobj.save(self.file, sel)

        print(status + f"\t{validsymb}")

    def set_rfam(self, rfam):
        self.rfam_fam = rfam
        print("\t> Associating it to", rfam, f"...\t\t\t{validsymb}")

    def extract_3D_data(self):
        if not os.access(path_to_3D_data+"pseudotorsions/", os.F_OK):
            os.makedirs(path_to_3D_data+"pseudotorsions/")

        if not os.path.exists(path_to_3D_data+f"pseudotorsions/{self.chain_label}.csv"):

            # run DSSR (you need to have it in your $PATH, follow x3dna installation instructions)
            output = subprocess.run(
                ["x3dna-dssr", f"-i={self.file}", "--json", "--auxfile=no"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = output.stdout.decode('utf-8')
            stderr = output.stderr.decode('utf-8')
            try:
                if "exception" in stderr:
                    warn(f"Exception while running DSSR: {stderr}\n\tIgnoring {self.chain_label}.\t\t\t", error=True)
                    self.delete_me = True
                    return
                json_object = json.loads(stdout)
                if "warning" in json_object.keys():
                    warn(f"Ignoring {self.chain_label} ({json_object['warning']})\t", error=True)
                    self.delete_me = True
                    return
                nts = json_object["nts"]
            except KeyError as e:
                warn(f"Error while parsing DSSR's json output:\n{json_object.keys()}\n{json_object}\n\tignoring {self.chain_label}\t\t\t\t", error=True)
                self.delete_me = True
                return
            print("\t> Computing", self.chain_label, f"pseudotorsions...\t\t{validsymb}", flush=True)
            
            # extract angles
            l = int(nts[-1]["nt_resnum"]) - int(nts[0]["nt_resnum"]) + 1
            eta = [np.NaN] * l
            theta = [np.NaN] * l
            mask = [ 0 ] * l
            seq = [ "-" ] * l # nts that are not resolved will be marked "-" in the sequence, and their mask at 0.

            resnum_start = int(nts[0]["nt_resnum"])

            for nt in nts:
                if nt["eta_prime"] is None:
                    e = np.NaN
                else:
                    e = float(nt["eta_prime"])
                if nt["theta_prime"] is None:
                    t = np.NaN
                else:
                    t = float(nt["theta_prime"])
                p = int(nt["nt_resnum"]) - resnum_start
                mask[p]  = int(nt["nt_code"] in "ACGU")            # U is a U, u is a modified U and should be replaced by consensus ?
                seq[p]   = nt["nt_code"].upper().replace('?','-').replace('P','U').replace('T','U')  # to align the chain with its family. The final nt should be taken from the PSSM.
                eta[p]   = e
                theta[p] = t

            if self.reversed:
                warn(f"Has {self.chain_label} been numbered from 3' to 5' ? Inverting angles.")
                # the 3D structure is numbered from 3' to 5' instead of standard 5' to 3'
                # or the sequence that matches the Rfam family is 3' to 5' instead of standard 5' to 3'.
                # anyways, you need to invert the angles.
                seq = seq[::-1]
                mask = mask[::-1]
                temp_eta = [ e for e in eta ]
                eta = [ theta[n] for n in range(l) ]        # eta(n)    = theta(l-n+1) forall n in ]1, l] 
                theta = [ temp_eta[n] for n in range(l) ]   # theta(n)  = eta(l-n+1)   forall n in [1, l[ 

            pd.DataFrame({"seq": list(seq), "m": list(mask), "eta": list(eta), "theta": list(theta)}
                        ).to_csv(path_to_3D_data+f"pseudotorsions/{self.chain_label}.csv")
            print("\t> Saved", self.chain_label, f"pseudotorsions to CSV.\t\t{validsymb}", flush=True)
        else:
            print("\t> Computing", self.chain_label, f"pseudotorsions...\t{validsymb}\t(already done)", flush=True)

        # Now load data from the CSV file
        d = pd.read_csv(path_to_3D_data+f"pseudotorsions/{self.chain_label}.csv")
        self.seq = "".join(d.seq.values)
        self.length = len([ x for x in d.seq if x != "-" ])
        self.full_length = len(d.seq)
        self.mask = "".join([ str(int(x)) for x in d.m.values])
        self.etas = d.eta.values
        self.thetas = d.theta.values
        print(f"\t> Loaded data from CSV\t\t\t\t{validsymb}", flush=True)

        if self.length < 5:
            warn(f"{self.chain_label} sequence is too short, let's ignore it.\t", error=True)
            self.delete_me = True
        return


class Job:
    def __init__(self, results="", command=[], function=None, args=[], how_many_in_parallel=0, priority=1, timeout=None, checkFunc=None, checkArgs=[], label=""):
        self.cmd_ = command
        self.func_ = function
        self.args_ = args
        self.checkFunc_ = checkFunc
        self.checkArgs_ = checkArgs
        self.results_file = results
        self.priority_ = priority
        self.timeout_ = timeout
        self.comp_time = -1 # -1 is not executed yet
        self.max_mem = -1 # not executed yet
        self.label = label
        if not how_many_in_parallel:
            self.nthreads = read_cpu_number()
        elif how_many_in_parallel == -1:
            self.nthreads = read_cpu_number() - 1
        else:
            self.nthreads = how_many_in_parallel
        self.useless_bool = False

    def __str__(self):
        if self.func_ is None:
            s = f"{self.priority_}({self.nthreads}) [{self.comp_time}]\t{self.label:25}" + " ".join(self.cmd_)
        else:
            s = f"{self.priority_}({self.nthreads}) [{self.comp_time}]\t{self.label:25}{self.func_.__name__}(" + " ".join([str(a) for a in self.args_]) + ")"
        return s


class AnnotatedStockholmIterator(AlignIO.StockholmIO.StockholmIterator):
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
        self.sequences = seqs
        self.seq_annotation = gs
        self.seq_col_annotation = gr
        self.alignment_annotation = gf

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
    def __init__(self, pid):
        self.keep_watching = True
        self.target_pid = pid

    def check_mem_usage(self):
        #print("\t> Monitoring process", self.target_pid, "from process", os.getpid())
        target_process = psutil.Process(self.target_pid)
        max_mem = -1
        while self.keep_watching:
            try:
                info = target_process.memory_full_info()
                mem = info.rss + info.swap
                for p in target_process.children(recursive=True):
                    info = p.memory_full_info()
                    mem += info.rss + info.swap
            except psutil.NoSuchProcess:
                print("\t> ERR: monitored process does not exist anymore ! Did something go wrong ?")
                self.keep_watching = False
            finally:
                if mem > max_mem:
                    max_mem = mem
            sleep(0.1)
        return max_mem

def read_cpu_number():
    # do not use os.cpu_count() on LXC containers
    # it reads info from /sys wich is not the VM resources but the host resources.
    # Read from /proc/cpuinfo instead.
    p = subprocess.run(['grep', '-Ec', '(Intel|AMD)', '/proc/cpuinfo'], stdout=subprocess.PIPE)
    return int(p.stdout.decode('utf-8')[:-1])/2

def warn(message, error=False):
    if error:
        print(f"\t> \033[31mERR: {message}\033[0m{errsymb}", flush=True)
    else:
        print(f"\t> \033[33mWARN: {message}\033[0m{warnsymb}", flush=True)

def execute_job(j, jobcount):
    running_stats[0] += 1

    if len(j.cmd_):
        # log the command
        logfile = open(runDir + "/log_of_the_run.sh", 'a')
        logfile.write(" ".join(j.cmd_))
        logfile.write("\n")
        logfile.close()
        print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.label}")

        monitor = Monitor(os.getpid())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            assistant_future = executor.submit(monitor.check_mem_usage)

            start_time = time.time()
            r = subprocess.run(j.cmd_, timeout=j.timeout_, stdout=subprocess.DEVNULL)
            end_time = time.time()

            monitor.keep_watching = False
            m = assistant_future.result()

    elif j.func_ is not None:

        #print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.func_.__name__}({', '.join([str(a) for a in j.args_ if not ((type(a) == list) and len(a)>3)])})")

        m = -1
        monitor = Monitor(os.getpid())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            assistant_future = executor.submit(monitor.check_mem_usage)
            start_time = time.time()
            r = j.func_(* j.args_)
            end_time = time.time()
            monitor.keep_watching = False
            m = assistant_future.result()

    # Job is finished
    running_stats[1] += 1
    t = end_time - start_time
    return (t,m,r)

def execute_joblist(fulljoblist, printstats=False):
    # reset counters
    running_stats[0] = 0
    running_stats[1] = 0
    running_stats[2] = 0

    # sort jobs in a tree structure
    jobs = {}
    jobcount = len(fulljoblist)
    for job in fulljoblist:
        if job.priority_ not in jobs.keys():
            jobs[job.priority_] = {}
        if job.nthreads not in jobs[job.priority_].keys():
            jobs[job.priority_][job.nthreads] = []
        jobs[job.priority_][job.nthreads].append(job)
    nprio = max(jobs.keys())

    if printstats:
        # Write stats in a file
        f = open("jobstats.csv", "w")
        f.write("label,comp_time,max_mem\n")
        f.close()

    # for each priority level
    results = {}
    for i in range(1,nprio+1):
        if i not in jobs.keys(): continue # ignore this priority level if no job available
        different_thread_numbers = [n for n in jobs[i].keys()]
        different_thread_numbers.sort()
        print("processing jobs of priority", i)
        res = []
        # jobs should be processed 1 by 1, 2 by 2, or n by n depending on their definition
        for n in different_thread_numbers:
            bunch = jobs[i][n]
            if not len(bunch): continue # ignore if no jobs should be processed n by n
            print("using", n, "processes:")

            # execute jobs of priority i that should be processed n by n:
            p = Pool(processes=n)
            raw_results = p.map(partial(execute_job, jobcount=jobcount), bunch)
            p.close()
            p.join()

            if printstats:
                # extract computation times
                times = [ r[0] for r in raw_results ]
                mems = [ r[1] for r in raw_results ]
                f = open("jobstats.csv", "a")
                for j, t, m in zip(bunch, times, mems):
                    j.comp_time = t
                    j.max_mem = m
                    print(f"\t> {j.label} finished in {t:.2f} sec with {int(m/1000000):d} MB of memory. \t{validsymb}", flush=True)
                    f.write(f"{j.label},{t},{m}\n")
                f.close()
            res += [ r[2] for r in raw_results ]
        results[i] = res
    return results

def download_Rfam_PDB_mappings():
    # download PDB mappings to Rfam family
    print("> Fetching latest PDB mappings from Rfam...", end='', flush=True)
    try:
        db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
        mappings = pd.read_sql('SELECT rfam_acc, pdb_id, chain, pdb_start, pdb_end, bit_score, evalue_score, cm_start, cm_end, hex_colour FROM pdb_full_region WHERE is_significant=1;', con=db_connection)
        mappings.to_csv(path_to_3D_data + 'Rfam-PDB-mappings.csv')
        print(f"\t{validsymb}")
    except sqlalchemy.exc.OperationalError:
        print(f"\t{errsymb}")
        if path.isfile(path_to_3D_data + 'Rfam-PDB-mappings.csv'):
            print("\t> Using previous version.")
            mappings = pd.read_csv(path_to_3D_data + 'Rfam-PDB-mappings.csv')
        else:
            print("Can't do anything without data. Check mysql-rfam-public.ebi.ac.uk status and try again later. Exiting.")
            exit(1)
    return mappings

def download_Rfam_seeds():
    if not path.isfile(path_to_seq_data + "seeds/Rfam.seed.gz"):
        _urlcleanup()
        _urlretrieve('ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz', path_to_seq_data + "seeds/Rfam.seed.gz")

    # Read Rfam seed alignements
    aligned_records = []
    rfam_acc = []
    alignment_len = []
    alignment_nseq = []
    AlignIO._FormatToIterator["stockholm"] = AnnotatedStockholmIterator # Tell biopython to use our overload
    with gzip.open(path_to_seq_data + "seeds/Rfam.seed.gz", encoding='latin-1') as gz:
        alignments = AlignIO.parse(gz, "stockholm", alphabet=generic_rna)
    for align in alignments:
        aligned_records.append('\n'.join([ str(s.seq) for s in align ]))
        rfam_acc.append(align._fileannotations["AC"][0])
        alignment_len.append(align.get_alignment_length())
        alignment_nseq.append(len(align._records))
    Rfam_seeds = pd.DataFrame()
    Rfam_seeds["aligned_records"] = aligned_records
    Rfam_seeds["rfam_acc"] = rfam_acc
    Rfam_seeds["alignment_len"] = alignment_len
    Rfam_seeds["alignment_nseq"] = alignment_nseq
    return Rfam_seeds

def download_Rfam_cm():
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
    try: 
        db_connection = sqlalchemy.create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
        q = """SELECT fr.rfam_acc, COUNT(DISTINCT fr.rfamseq_acc) AS 'n_seq',
                    MAX(
                        (CASE WHEN fr.seq_start > fr.seq_end THEN fr.seq_start
                                ELSE                              fr.seq_end
                        END)
                        -
                        (CASE WHEN fr.seq_start > fr.seq_end THEN fr.seq_end
                                ELSE                              fr.seq_start
                        END)
                    ) AS 'maxlength'
                FROM full_region fr
                GROUP BY fr.rfam_acc"""
        d = pd.read_sql(q, con=db_connection)
        return d[ d["rfam_acc"].isin(list_of_families) ]
    except sqlalchemy.exc.OperationalError:
        warn("Something's wrong with the SQL database. Check mysql-rfam-public.ebi.ac.uk status and try again later. Not printing statistics.")
        return {}

def download_Rfam_sequences(rfam_acc):
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
    # download latest BGSU non-redundant list
    print("> Fetching latest NR list from BGSU website...", end='', flush=True)
    try:
        s = requests.get("http://rna.bgsu.edu/rna3dhub/nrlist/download/current/4.0A/csv").content
        nr = open(path_to_3D_data + "latest_nr_list.csv", 'w')
        nr.write("class,representative,class_members\n")
        nr.write(io.StringIO(s.decode('utf-8')).getvalue())
        nr.close()
    except:
        warn("Error downloading NR list !\t", error=True)

        # try to read previous file
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
            all_chains.append(Chain(c))
    return all_chains

def build_chain(c, rfam, pdb_start, pdb_end):
    c.download_3D()
    if not c.delete_me:
        c.extract_portion(c.chain_label, pdb_start, pdb_end)
    if not c.delete_me:
        c.set_rfam(rfam)
        c.extract_3D_data()

    if c.delete_me and c.chain_label not in known_issues:
        warn(f"Adding {c.chain_label} to known issues.\t\t")
        f = open(path_to_3D_data + "known_issues.txt", 'a')
        f.write(c.chain_label + '\n')
        f.close()
    return c

def cm_realign(rfam_acc, chains, label):
    if path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.afa"):
        print(f"\t> {label} completed \t{validsymb}\t(already done)", flush=True)
        return

    # Preparing job folder
    if not os.access(path_to_seq_data + "realigned/", os.F_OK):
        os.makedirs(path_to_seq_data + "realigned/")

    # Reading Gzipped FASTA file and writing DNA FASTA file
    if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.fa"):
        print("\t> Extracting sequences...", flush=True)
        f = open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "w")
        with gzip.open(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz", 'rt') as gz:
            ids = []
            for record in SeqIO.parse(gz, "fasta"):
                if record.id not in ids:
                    f.write(">"+record.description+'\n'+str(record.seq)+'\n')
                    ids.append(record.id)
        for c in chains:
            seq_str = c.seq.replace('U','T').replace('-','') # We align as DNA
            seq_str = seq_str.replace('P','U') # Replace pseudo-uridines by uridines. We lose information here :'( 
            f.write(f"> {str(c)}\n"+seq_str+'\n') 

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
        print(f"\t> Download latest LSU-Ref alignment from SILVA...", end="", flush=True)
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

def summarize_position(col):
    # this function counts the number of nucleotides at a given position, given a "column" from a MSA.
    chars = ''.join(set(col))
    counts = { 'A': col.count('A') + col.count('a'),  # caps are nt in conformity with the consensus (?)
               'C':col.count('C') + col.count('c'), 
               'G':col.count('G') + col.count('g'), 
               'U':col.count('U') + col.count('u'), 
               '-':col.count('-'), '.':col.count('.')}
    known_chars_count = 0
    for char in chars:
        if char in "ACGU":
            known_chars_count += counts[char]
        elif char not in "-.acgu":
            counts[char] = col.count(char)
    N = len(col) - counts['-'] - counts['.'] # number of ungapped residues
    if N:
        return ( counts['A']/N, counts['C']/N, counts['G']/N, counts['U']/N, (N - known_chars_count)/N) # other residues, or consensus (N, K, Y...)
    else:
        return (0, 0, 0, 0, 0)

def alignment_nt_stats(f, list_of_chains) :
    global idxQueue
    #print("\t>",f,"... ", flush=True)
    chains_ids = [ str(c) for c in list_of_chains ]
    thr_idx = idxQueue.get()

    # Open the alignment
    align = AlignIO.read(path_to_seq_data + f"realigned/{f}++.afa", "fasta")
    alilen = align.get_alignment_length()
    #print("\t>",f,"... loaded", flush=True)

    # Compute statistics per column
    pbar = tqdm(total=alilen, position=thr_idx, desc=f"Worker { thr_idx}: {f}", leave=False, )
    results = [ summarize_position(align[:,i]) for i in pbar ]
    frequencies = np.array(results).T
    #print("\t>",f,"... loaded, computed", flush=True)

    for s in align:
        if not '[' in s.id: # this is a Rfamseq entry, not PDB
            continue

        # get the right 3D chain:
        idx = chains_ids.index(s.id)
        c = list_of_chains[idx]

        # Save colums in the appropriate positions
        i = 0
        j = 0
        warn_gaps = False
        while i<c.full_length and j<alilen:
            # here we try to map c.seq (the sequence of the 3D chain, including gaps when residues are missing), 
            # with s.seq, the sequence aligned in the MSA, containing any of ACGUacguP and two types of gaps, - and .

            if c.seq[i] == s.seq[j].upper(): # alignment and sequence correspond (incl. gaps)
                list_of_chains[idx].frequencies = np.concatenate((list_of_chains[idx].frequencies, frequencies[:,j].reshape(-1,1)), axis=1)
                i += 1
                j += 1
            elif c.seq[i] == '-': # gap in the chain, but not in the aligned sequence

                # search for a gap to the consensus nearby
                k = 0
                while j+k<alilen and s.seq[j+k] in ['.','-']:
                    if s.seq[j+k] == '-':
                        break
                    k += 1

                # if found, set j to that position
                if j+k<alilen and s.seq[j+k] == '-':
                    j = j + k
                    continue

                # if not, search for a insertion gap nearby
                if j<alilen and s.seq[j] == '.':
                    list_of_chains[idx].frequencies = np.concatenate((list_of_chains[idx].frequencies, frequencies[:,j].reshape(-1,1)), axis=1)
                    i += 1
                    j += 1
                    continue

                # else, just ignore the gap.
                warn_gaps = True
                list_of_chains[idx].frequencies = np.concatenate((list_of_chains[idx].frequencies, np.array([0.0,0.0,0.0,0.0,1.0]).reshape(-1,1)), axis=1)
                i += 1
            elif s.seq[j] in ['.', '-']: # gap in the alignment, but not in the real chain
                j += 1 # ignore the column
            else:
                print(f"You are never supposed to reach this. Comparing {c.chain_label} in {i} ({c.seq[i-1:i+2]}) with seq[{j}] ({s.seq[j-3:j+4]}).\n", c.seq, '\n', s.seq, sep='', flush=True)
                exit(1)
        #if warn_gaps:
            #warn(f"Some gap(s) in {c.chain_label} were not re-found in the aligned sequence... Ignoring them.")

        # Replace masked positions by the consensus sequence:
        c_seq = c.seq.split()
        letters = ['A', 'C', 'G', 'U', 'N']
        for i in range(c.full_length):
            if not c.mask[i]:
                freq = list_of_chains[idx].frequencies[:,i]
                c_seq[i] = letters[freq.tolist().index(max(freq))]
        list_of_chains[idx].seq = ''.join(c_seq)

        # Saving 'final' datapoint
        c = list_of_chains[idx] # update the local object
        point = np.zeros((13, c.full_length))
        for i in range(c.full_length):
            point[0,i] = i+1 # position

            # one-hot encoding of the actual sequence
            if c.seq[i] in letters[:4]:
                point[ letters[:4].index(c.seq[i]), i ] = 1
            else:
                point[5,i] = 1

            # save the PSSMs
            point[6,i] = c.frequencies[0, i]
            point[7,i] = c.frequencies[1, i]
            point[8,i] = c.frequencies[2, i]
            point[9,i] = c.frequencies[3, i]
            point[10,i] = c.frequencies[4, i]
            point[11,i] = c.etas[i]
            point[12,i] = c.thetas[i]
        file = open(path_to_3D_data + "datapoints/" + c.chain_label, "w") # no check, always overwrite, this is desired
        for i in range(13):
            line = [str(x) for x in list(point[i,:]) ]
            file.write(','.join(line)+'\n')
        file.close()
        #print("\t\tWritten", c.chain_label, f"to file\t{validsymb}", flush=True)

    #print("\t>", f, f"... loaded, computed, saved\t{validsymb}", flush=True)
    return 0

if __name__ == "__main__":
    print("Main process running. (PID", os.getpid(), ")")

    # temporary, for debugging:
    if os.path.exists(path_to_3D_data + "known_issues.txt"):
        subprocess.run(["rm", path_to_3D_data + "known_issues.txt"])

    # ===========================================================================
    # List 3D chains with available Rfam mapping
    # ===========================================================================
    all_chains = download_BGSU_NR_list()
    mappings = download_Rfam_PDB_mappings()
    chains_with_mapping = []
    for c in all_chains:
        mapping = mappings.loc[ (mappings.pdb_id == c.pdb_id) & (mappings.chain == c.pdb_chain_id) ]
        if len(mapping.rfam_acc.values):
            chains_with_mapping.append(c)
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

    # Download the corresponding CIF files and extract the mapped portions
    mmcif_parser = MMCIFParser()

    joblist = []
    for c in chains_with_mapping:
        # read mappings information
        mapping = mappings.loc[ (mappings.pdb_id == c.pdb_id) & (mappings.chain == c.pdb_chain_id) ]
        pdb_start = str(mapping.pdb_start.values[0])
        pdb_end = str(mapping.pdb_end.values[0])
        # build the chain
        c.chain_label = f"{c.pdb_id}_{str(c.pdb_model)}_{c.pdb_chain_id}_{pdb_start}-{pdb_end}"
        if c.chain_label not in known_issues:
            joblist.append(Job(function=build_chain, args=[c, mapping.rfam_acc.values[0], pdb_start, pdb_end]))

    if not path.isdir(path_to_3D_data + "rna_mapped_to_Rfam"):
        os.makedirs(path_to_3D_data + "rna_mapped_to_Rfam")
    if not path.isdir(path_to_3D_data + "RNAcifs"):
        os.makedirs(path_to_3D_data + "RNAcifs")

    results = execute_joblist(joblist)[1]
    loaded_chains = [ c for c in results if not c.delete_me ]
    print(f"> Loaded {len(loaded_chains)} RNA chains ({len(chains_with_mapping) - len(loaded_chains)} errors).")

    # ===========================================================================
    # Download RNA sequences of the corresponding Rfam families
    # ===========================================================================

    # Get the list of Rfam families found
    rfam_acc_to_download = {}
    for c in loaded_chains:
        if c.rfam_fam not in rfam_acc_to_download:
            rfam_acc_to_download[c.rfam_fam] = [ c ]
        else:
            rfam_acc_to_download[c.rfam_fam].append(c)
    print(f"> Identified {len(rfam_acc_to_download.keys())} families to download and re-align with the crystals' sequences:")

    # Download the sequences from these families
    download_Rfam_cm()
    fam_stats = download_Rfam_family_stats(rfam_acc_to_download.keys())
    if len(fam_stats.keys()):
        n_pdb = [ len(rfam_acc_to_download[f]) for f in fam_stats["rfam_acc"] ]
        fam_stats["n_pdb_seqs"] = n_pdb
        fam_stats["total_seqs"] = fam_stats["n_seq"] + fam_stats["n_pdb_seqs"]
        fam_stats.to_csv(path_to_seq_data + "realigned/statistics.csv")
    for f in sorted(rfam_acc_to_download.keys()):
        if len(fam_stats.keys()):
            line = fam_stats[fam_stats["rfam_acc"]==f]
            print(f"\t> {f}: {line.n_seq.values[0]} Rfam hits + {line.n_pdb_seqs.values[0]} PDB sequences to realign")
        download_Rfam_sequences(f)

    # ==========================================================================================
    # Realign sequences from 3D chains to Rfam's identified hits (--> extended full alignement)
    # ==========================================================================================

    # Build job list
    fulljoblist = []
    for f in sorted(rfam_acc_to_download.keys()):
        label = f"Realign {f} + {len(rfam_acc_to_download[f])} chains"
        fulljoblist.append(Job(function=cm_realign, args=[f, rfam_acc_to_download[f], label], how_many_in_parallel=1, priority=1, label=label))
    execute_joblist(fulljoblist, printstats=True)

    # ==========================================================================================
    # Now compute statistics on base variants at each position of every 3D chain
    # ==========================================================================================

    if not path.isdir(path_to_3D_data + "datapoints/"):
        os.makedirs(path_to_3D_data + "datapoints/")
    print("Computing nucleotide frequencies in alignments...")
    families =  sorted([f for f in rfam_acc_to_download.keys() ])

    # Build job list
    thr_idx_mgr = multiprocessing.Manager()
    idxQueue = thr_idx_mgr.Queue()
    for i in range(10):
        idxQueue.put(i)
    fulljoblist = []
    for f in families:
        label = f"Save {f} PSSMs"
        list_of_chains = rfam_acc_to_download[f]
        fulljoblist.append(Job(function=alignment_nt_stats, args=[f, list_of_chains], how_many_in_parallel=10, priority=1, label=label))
    execute_joblist(fulljoblist, printstats=False)

    print("Completed.")
