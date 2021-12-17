import os
from multiprocessing import Pool
from functools import partial

from microbiomemeta.data.utils import (
    Blosum62Plus,
    read_hmmer_sto,
    preprocess_hmp_metagenome,
)


def _collect_alignments(scandir_it):
    for entry in scandir_it:
        if not entry.name.endswith(".sto"):
            continue
        cluster = entry.name.split(".")[0]
        aln = read_hmmer_sto(entry.path)
        yield cluster, aln


def main(args):
    hmdbbasedir = os.path.join("Data", "HMDB")
    # hmpbasedir = "test"
    aln_dir = os.path.join(hmdbbasedir, "HMDB_hmmer_aln")
    out_dir = os.path.join(hmdbbasedir, "HMDB_corpus")

    aln_files = os.scandir(os.path.join(aln_dir))
    blosum62 = Blosum62Plus()(expand=True)

    worker = partial(
        preprocess_hmp_metagenome, out_dir=out_dir, matrix=blosum62, verbose=True
    )
    jobs = iter([(cl, aln) for cl, aln in _collect_alignments(aln_files)])

    os.makedirs(out_dir, exist_ok=True)
    pool = Pool(args.ncpu)
    pool.imap(worker, jobs, chunksize=10)
    pool.close()
    pool.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ncpu", type=int, default=100)
    args = parser.parse_args()
    main(args)
