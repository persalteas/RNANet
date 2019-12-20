#!/usr/bin/python3
import pandas as pd
import concurrent.futures, Bio.PDB, Bio.PDB.StructureBuilder, gzip, io, multiprocessing, os, psutil, re, requests, subprocess, sys, time, warnings
from Bio import AlignIO, SeqIO
from Bio.PDB import PDBParser, MMCIFParser, MMCIF2Dict, PDBIO, Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException
from Bio._py3k import urlretrieve as _urlretrieve
from Bio._py3k import urlcleanup as _urlcleanup
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from collections import OrderedDict
from ftplib import FTP
from sqlalchemy import create_engine
from os import path, makedirs
from multiprocessing import Pool, cpu_count, Manager
from time import sleep

if path.isdir("/home/persalteas"):
    path_to_3D_data = "/home/persalteas/Data/RNA/3D/"
    path_to_seq_data = "/home/persalteas/Data/RNA/sequences/"
elif path.isdir("/home/lbecquey"):
    path_to_3D_data = "/home/lbecquey/Data/RNA/3D/"
    path_to_seq_data = "/home/lbecquey/Data/RNA/sequences/"
elif path.isdir("/nhome/siniac/lbecquey"):
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
        if hetatm_flag != " ":
            return 0 # skip HETATMs
        if icode != " ":
            print("\t> \U000026A0 \033[33micode %s at position %s\033[0m" % (icode, resseq), flush=True)
        if self.start <= resseq <= self.end:
            return 1
        return 0

    def accept_atom(self, atom):
        name = atom.get_id()
        if hydrogen.match(name):
            return 0 # Get rid of hydrogens
        return 1


class SloppyStructureBuilder(Bio.PDB.StructureBuilder.StructureBuilder):
    """Cope with resSeq < 10,000 limitation by just incrementing internally.

    # Q: What's wrong here??
    #   Some atoms or residues will be missing in the data structure.
    #   WARNING: Residue (' ', 8954, ' ') redefined at line 74803.
    #   PDBConstructionException: Blank altlocs in duplicate residue SOL
    #   (' ', 8954, ' ') at line 74803.
    #
    # A: resSeq only goes to 9999 --> goes back to 0 (PDB format is not really good here)
    """

    # NOTE/TODO:
    # - H and W records are probably not handled yet (don't have examples
    #   to test)

    def __init__(self, verbose=False):
        Bio.PDB.StructureBuilder.StructureBuilder.__init__(self)
        self.max_resseq = -1
        self.verbose = verbose

    def init_residue(self, resname, field, resseq, icode):
        """Initiate a new Residue object.

        Arguments:
        o resname - string, e.g. "ASN"
        o field - hetero flag, "W" for waters, "H" for hetero residues, otherwise blanc.
        o resseq - int, sequence identifier
        o icode - string, insertion code
        """

        if field != " ":
            if field == "H":
                # The hetero field consists of H_ + the residue name (e.g. H_FUC)
                field = "H_" + resname
        res_id = (field, resseq, icode)

        if resseq > self.max_resseq:
            self.max_resseq = resseq

        if field == " ":
            fudged_resseq = False
            while (self.chain.has_id(res_id) or resseq == 0):
                # There already is a residue with the id (field, resseq, icode)
                # resseq == 0 catches already wrapped residue numbers which do not trigger the has_id() test.
                #
                # Be sloppy and just increment... (This code will not leave gaps in resids... I think)
                #
                # XXX: shouldn't we also do this for hetero atoms and water??
                self.max_resseq += 1
                resseq = self.max_resseq
                res_id = (field, resseq, icode)    # use max_resseq!
                fudged_resseq = True

            if fudged_resseq and self.verbose:
                sys.stderr.write("Residues are wrapping (Residue ('%s', %i, '%s') at line %i)." % (field, resseq, icode, self.line_counter)
                                +".... assigning new resid %d.\n" % self.max_resseq)
        residue = Residue(res_id, resname, self.segid)
        self.chain.add(residue)
        self.residue = residue


class SloppyPDBIO(Bio.PDB.PDBIO):
    """PDBIO class that can deal with large pdb files as used in MD simulations

    - resSeq simply wrap and are printed modulo 10,000.
    - atom numbers wrap at 99,999 and are printed modulo 100,000

    """
    # The format string is derived from the PDB format as used in PDBIO.py (has to be copied to the class because of the package layout it is not externally accessible)
    _ATOM_FORMAT_STRING = "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n"

    def _get_atom_line(self, atom, hetfield, segid, atom_number, resname, resseq, icode, chain_id, element="  ", charge="  "):
        """ Returns an ATOM string that is guaranteed to fit the ATOM format.

        - Resid (resseq) is wrapped (modulo 10,000) to fit into %4i (4I) format
        - Atom number (atom_number) is wrapped (modulo 100,000) to fit into
          %5i (5I) format

        """
        if hetfield != " ":
            record_type = "HETATM"
        else:
            record_type = "ATOM  "
        name = atom.get_fullname()
        altloc = atom.get_altloc()
        x, y, z = atom.get_coord()
        bfactor = atom.get_bfactor()
        occupancy = atom.get_occupancy()
        args = (record_type, atom_number % 100000, name, altloc, resname, chain_id, resseq % 10000, icode, x, y, z, occupancy, bfactor, segid, element, charge)
        return self._ATOM_FORMAT_STRING % args


class Chain:
    def __init__(self, nrlist_code):
        nr = nrlist_code.split('|')
        self.pdb_id = nr[0].lower()
        self.pdb_model = int(nr[1])
        self.pdb_chain_id = nr[2].upper()
        self.infile_chain_id = self.pdb_chain_id
        self.full_mmCIFpath = ""
        self.file = ""
        self.rfam_fam = ""
        self.seq = ""
        self.length = -1
        self.delete_me = False

    def __str__(self):
        return self.pdb_id + '[' + str(self.pdb_model) + "]-" + self.pdb_chain_id

    def download_3D(self):
        status = f"\t> Download {self.pdb_id}.cif\t\t\t"
        url = 'http://files.rcsb.org/download/%s.cif' % (self.pdb_id)
        if not os.access(path_to_3D_data+"RNAcifs", os.F_OK):
            os.makedirs(path_to_3D_data+"RNAcifs")
        final_filepath = path_to_3D_data+"RNAcifs/"+self.pdb_id+".cif"

        if os.path.exists(final_filepath):
            print(status + "\t\U00002705\t(structure exists)")
            self.full_mmCIFpath = final_filepath
            return
        else:
            try:
                _urlcleanup()
                _urlretrieve(url, final_filepath)
                self.full_mmCIFpath = final_filepath
                print(status + "\t\U00002705")
            except IOError:
                print(status + f"\t\U0000274E\t\033[31mError downloading {url} !\033[0m")
                self.delete_me = True

    def extract_portion(self, filename, pdb_start, pdb_end):
        status = f"\t> Extract {pdb_start}-{pdb_end} atoms from {self.pdb_id}-{self.pdb_chain_id}\t"
        self.file = path_to_3D_data+"rna_mapped_to_Rfam/"+filename+".pdb"

        if os.path.exists(self.file):
            print(status + "\t\U00002705\t(already done)", flush=True)
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

            # Modify the start/end numbers (given positions are relative to the chain, not the whole PDB)
            c = s[model_idx][self.pdb_chain_id]             # the desired chain
            first_number = c.child_list[0].get_id()[1]      # its first residue is numbered 'first_number'
            start = pdb_start + first_number - 1            # then our start position should be shifted by 'first_number'
            end = pdb_end + first_number - 1                # same for the end position

            chain = self.pdb_chain_id
            # Modify the chain ID: cif files can have 2-char chain names, not PDB files.
            if len(c.id) > 1:
                parent_chains = c.get_parent().child_dict.keys()
                s[model_idx].detach_child(c.id) # detach old chain_id from the model
                if c.id[-1] in parent_chains:
                    s[model_idx].detach_child(c.id[-1]) # detach new chain_id from the model
                    c.detach_parent() # isolate the chain from its model
                chain = c.id[-1]
                c.id = c.id[-1]
                s[model_idx].add(c) # attach updated chain to the model
            self.infile_chain_id = chain

            # Do the extraction of the 3D data
            sel = NtPortionSelector(model_idx, chain, start, end)
            ioobj = SloppyPDBIO()
            ioobj.set_structure(s)
            ioobj.save(self.file, sel)

        print(status + "\t\U00002705")

    def load_sequence(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            s = pdb_parser.get_structure(self.pdb_id, self.file)
        cs = s.get_chains() # the only chain of the file
        seq = ""
        for c in cs: # there should be only one chain, actually, but "yield" forbids the subscript
            for r in c:
                seq += r.get_resname().strip()
        self.seq =  seq
        self.length = len(self.seq)
        print(f"\t> Extract subsequence from {self.pdb_id}-{self.pdb_chain_id}\t\t\U00002705\t(length {self.length})")

    def set_rfam(self, rfam):
        self.rfam_fam = rfam


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    # because the latter is only a wrapper function, not a proper class.
    Process = NoDaemonProcess


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
            self.nthreads = cpu_count()
        elif how_many_in_parallel == -1:
            self.nthreads = cpu_count() - 1
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
        target_process = psutil.Process(self.target_pid)
        max_mem = -1
        while self.keep_watching:
            info = target_process.memory_full_info()
            mem = info.rss + info.swap
            for p in target_process.children(recursive=True):
                info = p.memory_full_info()
                mem += info.rss + info.swap
            if mem > max_mem:
                max_mem = mem
            # sleep(0.1)       
        return max_mem

def execute_job(j):
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
            r = subprocess.call(j.cmd_, timeout=j.timeout_)
            end_time = time.time()

            monitor.keep_watching = False
            m = assistant_future.result()
            

    elif j.func_ is not None:

        print(f"[{running_stats[0]+running_stats[2]}/{jobcount}]\t{j.func_.__name__}({', '.join([str(a) for a in j.args_ if not ((type(a) == list) and len(a)>3)])})")

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
    print(f"\t> finished in {t:.2f} sec with {int(m/1000000):d} MB of memory. \t\U00002705")
    return (t,m,r)

def download_Rfam_PDB_mappings():
    # download PDB mappings to Rfam family
    print("> Fetching latest PDB mappings from Rfam...", end='', flush=True)
    db_connection = create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
    mappings = pd.read_sql('SELECT rfam_acc, pdb_id, chain, pdb_start, pdb_end, bit_score, evalue_score, cm_start, cm_end, hex_colour FROM pdb_full_region WHERE is_significant=1;', con=db_connection)
    mappings.to_csv(path_to_3D_data + 'Rfam-PDB-mappings.csv')
    print("\t\U00002705")
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
            print("\t\U00002705")
            print(f"\t\t> Uncompressing Rfam.cm...", end='', flush=True)
            subprocess.check_call(["gunzip", path_to_seq_data + "Rfam.cm.gz"])
            print("\t\U00002705")
        except:
            print(f"\t\U0000274C\tError downloading and/or extracting Rfam.cm")
    else:
        print("\t\U00002705\t(no need)")

def download_Rfam_family_stats(list_of_families):
    db_connection = create_engine('mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam')
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

def download_Rfam_sequences(rfam_acc):
    print(f"\t\t> Download {rfam_acc}.fa.gz from Rfam...", end='', flush=True)
    if not path.isfile(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz"):
        try:
            _urlcleanup()
            _urlretrieve(   f'ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/{rfam_acc}.fa.gz',
                            path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz")
            print("\t\U00002705")
        except:
            print(f"\t\U0000274C\tError downloading {rfam_acc}.fa.gz. Does it exist ?")
    else:
        print("\t\U00002705\t(already there)")

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
        print("\t\U0000274C\tError downloading NR list !")

        # try to read previous file
        if path.isfile(path_to_3D_data + "latest_nr_list.csv"):
            print("\t> Use of the previous version.\t", end = "", flush=True)
        else:
            return [], []

    nrlist = pd.read_csv(path_to_3D_data + "latest_nr_list.csv")
    nr_structures_list = nrlist['representative'].tolist()
    full_structures_list = nrlist['class_members'].tolist()
    print("\t\U00002705")

    # Split the codes
    nr_chains = []
    all_chains = []
    for code in nr_structures_list:
        nr_chains.append(Chain(code))
    for code in full_structures_list:
        codes = code.replace('+',',').split(',')
        for c in codes:
            all_chains.append(Chain(c))
    return all_chains, nr_chains

def build_chain(c, filename, rfam, pdb_start, pdb_end):
    c.download_3D()
    if not c.delete_me:
        c.extract_portion(filename, pdb_start, pdb_end)
    if not c.delete_me:
        c.load_sequence()
        c.set_rfam(rfam)
    if len(c.seq)<5:
        c.delete_me = True

    if c.delete_me and filename not in known_issues:
        f = open(path_to_3D_data + "known_issues.txt", 'a')
        f.write(filename + '\n')
        f.close()
    return c

def cm_realign(rfam_acc, chains, label):
    if path.isfile(path_to_seq_data + f"realigned/{rfam_acc}++.afa"):
        print(f"\t> {label} completed \t\U00002705\t(already done)")
        return

    # Preparing job folder
    if not os.access(path_to_seq_data + "realigned/", os.F_OK):
        os.makedirs(path_to_seq_data + "realigned/")

    # Reading Gzipped FASTA file and writing DNA FASTA file
    print("\t> Extracting sequences...", flush=True)
    f = open(path_to_seq_data + f"realigned/{rfam_acc}++.fa", "w")
    with gzip.open(path_to_seq_data + f"rfam_sequences/fasta/{rfam_acc}.fa.gz", 'rt') as gz:
        ids = []
        for record in SeqIO.parse(gz, "fasta"):
            if record.id not in ids:
                f.write(">"+record.description+'\n'+str(record.seq)+'\n')
                ids.append(record.id)
    for c in chains:
        f.write(f"> {str(c)}\n"+c.seq.replace('U','T')+'\n') # We align as DNA
    f.close()

    # Extracting covariance model for this family
    print("\t> Extracting covariance model (cmfetch)...")
    if not path.isfile(path_to_seq_data + f"realigned/{rfam_acc}.cm"):
        f = open(path_to_seq_data + f"realigned/{rfam_acc}.cm", "w")
        subprocess.check_call(["cmfetch", path_to_seq_data + "Rfam.cm", rfam_acc], stdout=f)
        f.close()

    # Running alignment
    print(f"\t> {label} (cmalign)...")
    f = open(path_to_seq_data + f"realigned/{rfam_acc}++.stk", "w")
    subprocess.check_call(["cmalign", "--mxsize", "2048", path_to_seq_data + f"realigned/{rfam_acc}.cm", path_to_seq_data + f"realigned/{rfam_acc}++.fa"], stdout=f)
    f.close()

    # Converting to aligned Fasta
    # print("\t> Converting to aligned FASTA (esl-reformat)...")
    f = open(path_to_seq_data + f"realigned/{rfam_acc}++.afa", "w")
    subprocess.check_call(["esl-reformat", "afa", path_to_seq_data + f"realigned/{rfam_acc}++.stk"], stdout=f)
    f.close()
    # subprocess.check_call(["rm", path_to_seq_data + f"realigned/{rfam_acc}.cm", path_to_seq_data + f"realigned/{rfam_acc}++.fa", path_to_seq_data + f"realigned/{rfam_acc}++.stk"])


if __name__ == "__main__":
    print("Main process running. (PID", os.getpid(), ")")
    all_chains, nr_chains = download_BGSU_NR_list()
    mappings = download_Rfam_PDB_mappings()

    mmcif_parser = MMCIFParser(structure_builder=SloppyStructureBuilder())
    pdb_parser = PDBParser()

    # List chains with Rfam mapping
    print("> Searching for mappings in the PDB list...", end="", flush=True)
    chains_with_mapping = []
    for c in all_chains:
        mapping = mappings.loc[ (mappings.pdb_id == c.pdb_id) & (mappings.chain == c.pdb_chain_id) ]
        if len(mapping.rfam_acc.values):
            chains_with_mapping.append(c)
    n_chains = len(chains_with_mapping)
    print("\t\U00002705")


    print("> Building download list...", flush=True)
    # Check for a list of known problems:
    known_issues = []
    if path.isfile(path_to_3D_data + "known_issues.txt"):
        f = open(path_to_3D_data + "known_issues.txt", 'r')
        known_issues = f.readlines()
        f.close()

    # Download the corresponding CIF files and extract the mapped portions
    joblist = []
    for c in chains_with_mapping:
        mapping = mappings.loc[ (mappings.pdb_id == c.pdb_id) & (mappings.chain == c.pdb_chain_id) ]
        pdb_start = str(mapping.pdb_start.values[0])
        pdb_end = str(mapping.pdb_end.values[0])
        filename = f"{c.pdb_id}_{str(c.pdb_model)}_{c.pdb_chain_id}_{pdb_start}-{pdb_end}"
        filepath = path_to_3D_data+"rna_mapped_to_Rfam/"+filename+".pdb"
        if filename not in known_issues:
            joblist.append(Job(function=build_chain, args=[c, filename, mapping.rfam_acc.values[0], pdb_start, pdb_end]))
    if not path.isdir(path_to_3D_data + "rna_mapped_to_Rfam"):
        os.makedirs(path_to_3D_data + "rna_mapped_to_Rfam")
    if not path.isdir(path_to_3D_data + "RNAcifs"):
        os.makedirs(path_to_3D_data + "RNAcifs")
    jobcount = len(joblist)
    p = MyPool(processes=cpu_count(), maxtasksperchild=10)
    results = p.map(execute_job, joblist)
    p.close()
    p.join()
    loaded_chains = [ c[2] for c in results if not c[2].delete_me ]
    print(f"> Loaded {len(loaded_chains)} RNA chains ({len(chains_with_mapping) - len(loaded_chains)} errors).")

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
    n_pdb = [ len(rfam_acc_to_download[f]) for f in fam_stats["rfam_acc"] ]
    fam_stats["n_pdb_seqs"] = n_pdb
    fam_stats["total_seqs"] = fam_stats["n_seq"] + fam_stats["n_pdb_seqs"]
    fam_stats.to_csv(path_to_seq_data + "realigned/statistics.csv")
    for f in sorted(rfam_acc_to_download.keys()):
        line = fam_stats[fam_stats["rfam_acc"]==f]
        print(f"\t> {f}: {line.n_seq.values[0]} Rfam hits + {line.n_pdb_seqs.values[0]} PDB sequences to realign")
        download_Rfam_sequences(f)

    # Build job list
    fulljoblist = []
    running_stats[0] = 0
    running_stats[1] = 0
    running_stats[2] = 0
    for f in sorted(rfam_acc_to_download.keys()):
        if f=="RF02541": continue
        if f=="RF02543": continue
        label = f"Realign {f} + {len(rfam_acc_to_download[f])} chains"
        fulljoblist.append(Job(function=cm_realign, args=[f, rfam_acc_to_download[f], label], how_many_in_parallel=1, priority=1, label=label))

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
    # for each priority level
    for i in range(1,nprio+1):
        if i not in jobs.keys(): continue # ignore this priority level if no job available
        different_thread_numbers = [n for n in jobs[i].keys()]
        different_thread_numbers.sort()
        print("processing jobs of priority", i)
        # jobs should be processed 1 by 1, 2 by 2, or n by n depending on their definition
        for n in different_thread_numbers:
            bunch = jobs[i][n]
            if not len(bunch): continue # ignore if no jobs should be processed n by n
            print("using", n, "processes:")

            # execute jobs of priority i that should be processed n by n:
            p = MyPool(processes=n, maxtasksperchild=10)
            raw_results = p.map(execute_job, bunch)
            p.close()
            p.join()

            # extract computation times
            times = [ r[0] for r in raw_results ]
            mems = [ r[1] for r in raw_results ]
            for j, t, m in zip(bunch, times, mems):
                j.comp_time = t
                j.max_mem = m
                print(j.label, "ran in", t, "seconds with", m, "bytes of memory")

    # Write this in a file
    f = open("jobstats.csv", "w")
    f.write("label,comp_time,max_mem\n")
    f.close()
    for i in range(1,nprio+1):
        if i not in jobs.keys(): continue
        different_thread_numbers = [n for n in jobs[i].keys()]
        different_thread_numbers.sort()
        for n in different_thread_numbers:
            bunch = jobs[i][n]
            if not len(bunch):
                continue
            f = open("jobstats.csv", "a")
            for j in bunch:
                f.write(f"{j.label},{j.comp_time},{j.max_mem}\n")
            f.close()
    print("Completed.")




