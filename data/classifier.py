# Created by Baole Fang at 10/30/23

import argparse
import os

import numpy as np
from Bio import SeqIO


def main():
    Y = [record.description.split()[1] for record in records]
    np.save(path, Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dna dataset to vectors')
    parser.add_argument('-i', '--input', help='path to the input dataset', type=str, required=True)
    parser.add_argument('-t', '--type', help='type of the dataset', type=str, default='fasta')
    parser.add_argument('-o', '--output', help='output path', type=str, required=True)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()

    records = list(SeqIO.parse(args.input, args.type))
    path = os.path.join(args.output, 'class.npy')
    main()
