# Created by Baole Fang at 11/13/23
import argparse
import os

import numpy as np
from Bio import SeqIO

from dna2vec.multi_k_model import MultiKModel
from dna2vec.generators import DisjointKmerFragmenter, SlidingKmerFragmenter, SeqFragmenter
from tqdm import tqdm

import multiprocessing


def add_subtract(nums):
    ans = 0
    flag = True
    for num in nums:
        if flag:
            ans += num
        else:
            ans -= num
        flag = not flag
    return ans


def multiply_divide(nums):
    ans = 1
    flag = True
    for num in nums:
        if flag:
            ans *= num
        else:
            ans /= num
        flag = not flag
    return ans


def four_op(nums):
    ans = 0
    i = 0
    for num in nums:
        if i == 0:
            ans += num
        elif i == 1:
            ans -= num
        elif i == 2:
            ans *= num
        elif i == 3:
            ans /= num
        i += 1
        i %= 4
    return ans


def func1(X):
    ans = []
    for i, fr in enumerate(feature_ranges):
        x = X[fr]
        if i % 3 == 0:
            ans.append(add_subtract(x))
        elif i % 3 == 1:
            ans.append(multiply_divide(x))
        else:
            ans.append(four_op(x))
    return four_op(ans)


class InvalidArgException(Exception):
    pass


def dna2vec(seq):
    seq = ''.join(splitter.get_acgt_seqs(str(seq)))
    vec = np.zeros(model.vec_dim)
    fragments = fragmenter.apply(rng, seq)
    for fragment in fragments:
        vec += model.vector(fragment)
    label = func(vec)
    return vec / len(fragments), label


def main():
    seqs = [record.seq for record in records]
    with multiprocessing.Pool(os.cpu_count()) as p:
        X = list(tqdm(p.imap(dna2vec, seqs), total=len(seqs)))
    np.savez(path, X=[x[0] for x in X], Y=[x[1] for x in X])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dna dataset to vectors')
    parser.add_argument('-m', '--model', help='model path', type=str,
                        default='pretrained/dna2vec-20231031-0333-k3to8-100d-10c-8210Mbp-sliding-pOW.w2v')
    parser.add_argument('-l', '--low', help='lower bound of k', type=int, default=3)
    parser.add_argument('-u', '--up', help='upper bound of k', type=int, default=8)
    parser.add_argument('-i', '--input', help='path to the input dataset', type=str, required=True)
    parser.add_argument('-t', '--type', help='type of the dataset', type=str, default='fasta')
    parser.add_argument('-f', '--fragment', help='style to fragment the sequence: disjoint or sliding',
                        choices=['disjoint', 'sliding'], default='sliding')
    parser.add_argument('-o', '--output', help='output path', type=str, default='inputs')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    parser.add_argument('-g', '--goal', help='goal function', choices=['func1'], default='func1')
    parser.add_argument('-r', '--replace', help='whether to replace', type=bool, default=False)
    parser.add_argument('-x', '--x', help='width', type=int, default=5)
    parser.add_argument('-y', '--y', help='height', type=int, default=5)

    args = parser.parse_args()

    model = MultiKModel(args.model)
    if not model.k_low <= args.low < args.up <= model.k_high:
        raise InvalidArgException(f'Invalid relationship: {model.k_low}<={args.low}<{args.up}<={model.k_high}')

    if args.fragment == 'disjoint':
        fragmenter = DisjointKmerFragmenter(args.low, args.up)
    elif args.fragment == 'sliding':
        fragmenter = SlidingKmerFragmenter(args.low, args.up)
    else:
        raise InvalidArgException('Invalid kmer fragmenter: {}'.format(args.kmer_fragmenter))

    if args.goal == 'func1':
        func = func1
    else:
        raise InvalidArgException('Invalid goal: {}'.format(args.goal))

    splitter = SeqFragmenter()

    records = list(SeqIO.parse(args.input, args.type))
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    feature_ranges = np.random.choice(model.vec_dim, size=(args.y, args.x), replace=args.replace)
    name = os.path.basename(args.input).split('.')[0]
    path = os.path.join(args.output, f'{name}_{args.low}_{args.up}_{args.fragment}_{args.seed}_prediction.npz')
    main()
