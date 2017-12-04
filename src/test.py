from __future__ import unicode_literals

import sys
import os

import numpy
import numpy as np

import h5py

import matplotlib.pyplot as plt

from _load_data import load_a3m
#from _gaussdca import compute_gdca_scores
from _gaussdca_parallel import compute_gdca_scores

def generate_data(idfile, inpath, outfile):
    feat_lst = ["gdca", "gdca_corr"]
    with h5py.File(outfile, libver="latest") as h5out:
        for feat in feat_lst:
            if not feat in h5out:
                h5out.create_group(feat)
        with open(idfile) as idlist:
            for l in idlist:
                l = l.strip()
                l_arr = l.split(' ')
                i = l_arr[0]
                ali = "hhE0"
                if len(l_arr) == 2:
                    ali = l_arr[1]

                if "pconsc3" in inpath:
                    fname = u'%s/%s/%s.fa.%s.2016.a3m' % (inpath, i , i, ali)
                else:
                    fname = u'%s/%s.fa.%s.2016.a3m' % (inpath , i, ali)
                if not os.path.isfile(fname):
                    continue
                a3m_ali = load_a3m(fname)
                print(a3m_ali.shape)
                gdca_dict = compute_gdca_scores(a3m_ali)
                gdca = gdca_dict["gdca"]
                gdca_corr = gdca_dict["gdca_corr"]

                h5out.create_dataset("gdca/%s.%s" % (i, ali), data=gdca)
                h5out.create_dataset("gdca_corr/%s.%s" % (i, ali), data=gdca_corr)
                print(i, ali)

                plt.imshow(gdca_corr, interpolation='none')
                plt.colorbar()
                plt.show()






if __name__ == "__main__":

    idfile = sys.argv[1]
    inpath = sys.argv[2]
    outfile = sys.argv[3]

    generate_data(idfile, inpath, outfile)
