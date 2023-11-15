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


def main():
    Y = [func(x) for x in X]
    np.save(path, Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dna dataset to vectors')
    parser.add_argument('-m', '--model', help='model path', type=str,
                        default='pretrained/dna2vec-20231031-0333-k3to8-100d-10c-8210Mbp-sliding-pOW.w2v')
    parser.add_argument('-i', '--input', help='path to the input dataset', type=str, required=True)
    parser.add_argument('-o', '--output', help='output path', type=str, default='inputs')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    parser.add_argument('-g', '--goal', help='goal function', choices=['func1'], default='func1')
    parser.add_argument('-r', '--replace', help='whether to replace', type=bool, default=False)
    parser.add_argument('-x', '--x', help='width', type=int, default=5)
    parser.add_argument('-y', '--y', help='height', type=int, default=5)

    args = parser.parse_args()

    model = MultiKModel(args.model)

    if args.goal == 'func1':
        func = func1
    else:
        raise InvalidArgException('Invalid goal: {}'.format(args.goal))

    splitter = SeqFragmenter()

    np.random.seed(args.seed)
    feature_ranges = np.random.choice(model.vec_dim, size=(args.y, args.x), replace=args.replace)
    path = os.path.join(args.output, 'prediction.npy')
    X = np.load(args.input)
    main()
