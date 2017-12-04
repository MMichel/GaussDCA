import sys
import os
import argparse

import numpy as np

import matplotlib.pyplot as plt

def read_ranking(infile):
        if not os.path.isfile(infile):
            sys.stderr.write("Score file not found.\n")
            sys.exit(1)
        score_lst = []
        N = 0
        with open(infile) as f:
            for l in f:
                l_arr = l.strip().split()
                i = int(l_arr[0])
                j = int(l_arr[1])
                score = float(l_arr[2])
                score_lst.append((i-1,j-1,score))
                N = max(N, i, j)
        print(N)

        cmap = np.zeros((N,N))
        for (i, j, score) in score_lst:
            cmap[i,j] = score
            cmap[j,i] = cmap[i,j]
        
        return cmap

def plot_cmap_difference(cmap1, cmap2, outfile=''):
        plt.imshow(cmap1 - cmap2, interpolation='none')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot of contact score differences")
    p.add_argument('score_file1')
    p.add_argument('score_file2')
    p.add_argument('-o', '--output', default='')
    args = vars(p.parse_args(sys.argv[1:]))

    cmap1 = read_ranking(args['score_file1'])
    cmap2 = read_ranking(args['score_file2'])

    plot_cmap_difference(cmap1, cmap2, outfile=args['output'])
