import sys
import os
import argparse
from operator import itemgetter

import numpy
import numpy as np

import matplotlib.pyplot as plt

from _load_data import load_a3m
#from _gaussdca_parallel import compute_gdca_scores
from _gaussdca_parallel_opt import compute_gdca_scores

def run_gdca(infile, outfile='', num_threads=1, min_separation=5):
        if not os.path.isfile(infile):
            sys.stderr.write("Alignment file not found.\n")
            sys.exit(1)
        a3m_ali = load_a3m(infile)
        gdca_dict = compute_gdca_scores(a3m_ali, num_threads=num_threads)
        gdca = gdca_dict["gdca"]
        gdca_corr = gdca_dict["gdca_corr"]

        score_lst = compute_ranking(gdca_corr, outfile)

        if outfile:
            with open(outfile, 'w') as outf:
                for (i, j, score) in score_lst:
                    outf.write("%s %s %s\n" %(i, j, score))
        else:
            for (i, j, score) in score_lst:
                print(i, j, score)

        #plt.imshow(gdca_corr, interpolation='none')
        #plt.colorbar()
        #plt.show()


def compute_ranking(scores, outfile, min_separation=5):
    N = scores.shape[0]
    score_lst = []
    for i in range(N-min_separation):
        for j in range(i+min_separation, N):
            score_lst.append((i+1, j+1, scores[i,j]))
    score_lst.sort(key=itemgetter(2), reverse=True)
    return score_lst




if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cython implementation of GaussDCA (doi:10.1371/journal.pone.0092721)")
    p.add_argument('alignment_file')
    p.add_argument('-o', '--output', default='')
    p.add_argument('-s', '--separation', default=5, type=int)
    p.add_argument('-t', '--threads', default=1, type=int)
    args = vars(p.parse_args(sys.argv[1:]))

    run_gdca(args['alignment_file'], outfile=args['output'], num_threads=args['threads'], min_separation=args['separation'])
