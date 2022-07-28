import argparse
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

CAT_FEATURE_COUNT = 26
DAYS = 24
NUM_TOTAL_SAMPLES = 4373472329


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute category distribution for each column of Criteo dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with preprocessed data"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=int,
        nargs="+",
        help="The number of embeddings in each table: 26 values are expected for the Criteo dataset."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="category_prob.pkl",
        help="File to save pickled category probabilities"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=CAT_FEATURE_COUNT,
        help="The maximal number of workers for processing data columns in parallel"
    )
    return parser.parse_args()


def load_data(file_path: str) -> np.ndarray:
    data = np.load(file_path)
    data = np.reshape(data, (-1, CAT_FEATURE_COUNT), order='C')
    return data


def value_counts(column: np.ndarray, n: int, i: int) -> Tuple[np.ndarray, int]:
    counts = np.zeros(n, dtype=np.int64)
    counter = Counter(column % n)
    val = list(counter)
    cnt = np.fromiter(counter.values(), dtype=np.int64)
    counts[val] = cnt
    return counts, i


def get_category_count(data_dir: str, num_embeddings_per_feature: List[int], num_workers: int) -> Dict[int, np.ndarray]:
    category_count = {i: np.zeros(num_embeddings_per_feature[i], dtype=np.int64) for i in range(CAT_FEATURE_COUNT)}
    for d in tqdm(range(DAYS)):
        data = load_data(os.path.join(data_dir, f"day_{d}_sparse.npy"))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(value_counts, data[:, i], n, i) for i, n in enumerate(num_embeddings_per_feature)]
            for future in as_completed(futures):
                counts, i = future.result()
                category_count[i] += counts
    return category_count


def get_category_prob(category_counts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    category_prob = {}
    for i in tqdm(category_counts):
        category_prob[i] = np.divide(category_counts[i], NUM_TOTAL_SAMPLES, dtype=np.float32)
    return category_prob


def main():
    args = parse_args()
    assert os.path.isdir(args.results_dir)
    assert len(args.num_embeddings_per_feature) == CAT_FEATURE_COUNT

    print("Computing category distribution for all columns")
    category_count = get_category_count(args.data_dir, args.num_embeddings_per_feature, args.num_workers)
    category_prob = get_category_prob(category_count)

    with open(os.path.join(args.results_dir, args.results_file), "wb") as f:
        pickle.dump(category_prob, f)

    print("Done.")


if __name__ == "__main__":
    main()
