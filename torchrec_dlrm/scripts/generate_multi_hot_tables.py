import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cache for multi-hot sampling according to category distribution given"
    )
    parser.add_argument(
        "--category_prob_file",
        type=str,
        required=True,
        help="Pickled file with category distribution per column"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--multi_hot_size",
        type=int,
        default=20,
        help="The target number of multi-hot indices"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main(category_prob_file: str, output_dir: str, multi_hot_size: int, seed: int):
    with open(category_prob_file, "rb") as f:
        category_prob = pickle.load(f)

    print("Generating multi-hot tables according to distribution given")

    np.random.seed(seed)
    for i, p in tqdm(category_prob.items()):
        n = len(p)
        cache = np.random.choice(n, size=(n, multi_hot_size), replace=True, p=p)
        cache[:, 0] = 0  # The first column is reserved for the original category
        with open(os.path.join(output_dir, f"multi_hot_table_{i:0>2}.npy"), "wb") as f:
            np.save(f, cache)

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    assert os.path.isdir(args.results_dir)
    main(args.category_prob_file, args.results_dir, args.multi_hot_size, args.random_seed)
