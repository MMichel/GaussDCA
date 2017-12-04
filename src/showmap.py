import sys
import os
import argparse

import numpy as np

import matplotlib.pyplot as plt

def read_ranking(infile, outfile=''):
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

        plt.imshow(cmap, interpolation='none')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple plot of contact scores")
    p.add_argument('score_file')
    p.add_argument('-o', '--output', default='')
    args = vars(p.parse_args(sys.argv[1:]))

    read_ranking(args['score_file'], outfile=args['output'])
