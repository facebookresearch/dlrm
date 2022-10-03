"""
Script to convert current MLPerf dataset to format consumable by Torchrec. See also
https://github.com/pytorch/torchrec/blob/2500a163f3b9a1761e4394a0d1e7212c7d2a0214/torchrec/datasets/criteo.py#L587-L632.

Constant `ROWS_PER_DAY` can be determined using:
```
import numpy as np

for d in range(24):
    print(len(np.load(f"/data/shuffled/day_{d}_labels.npy")))
```
"""

import argparse
import os
import numpy as np
from datetime import datetime
from iopath.common.file_io import PathManagerFactory


ROWS_PER_DAY = [
    195841983,
    199563535,
    196792019,
    181115208,
    152115810,
    172548507,
    204846845,
    200801003,
    193772492,
    198424372,
    185778055,
    153588700,
    169003364,
    194216520,
    194081279,
    187154596,
    177984934,
    163382602,
    142061091,
    156534237,
    193627464,
    192215183,
    189747893,
    178274637,
]
INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
TOT_COLUMN_COUNT = 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT
DAYS = 24
PATH_MANAGER_KEY = "torchrec"


def parse_args():
    assert sum(ROWS_PER_DAY[:-1]) == 4195197692
    assert len(ROWS_PER_DAY) == DAYS
    parser = argparse.ArgumentParser(
        description="Convert MLPerf (v1.0-2.1) DLRM dataset to new format consumable by TorchRec (v3.0)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with the current MLPerf dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to dump data in the new format"
    )
    return parser.parse_args()


def ts(fmt="[%Y-%m-%d %H:%M:%S]:"):
    return datetime.now().strftime(fmt)


def load_dataset(input_dir, key):
    print(f"{ts()} Loading data for key = {key}")
    if key == "train":
        dataset = np.fromfile(
            os.path.join(input_dir, "train_data.bin"), dtype=np.int32
        ).reshape((-1, TOT_COLUMN_COUNT))
    elif key == "test+val":
        val_dataset = np.fromfile(
            os.path.join(input_dir, "val_data.bin"), dtype=np.int32
        ).reshape((-1, TOT_COLUMN_COUNT))
        test_dataset = np.fromfile(
            os.path.join(input_dir, "test_data.bin"), dtype=np.int32
        ).reshape((-1, TOT_COLUMN_COUNT))
        dataset = np.concatenate([test_dataset, val_dataset], axis=0)
    else:
        raise ValueError(f"Unknown key for loading data: {key}")
    print(f"{ts()} Dataset shape: {dataset.shape}.")
    print(f"{ts()} Loading done.")
    return dataset


def stats(dataset):
    print(f"{ts()} Getting dataset stats...")
    print("column        min           max")
    for i, col_min, col_max in zip(
       range(TOT_COLUMN_COUNT),
       dataset.min(axis=0),
       dataset.max(axis=0),
    ):
        print(f"{i:>6} {col_min:>10,}  {col_max:>12,}")
    print(f"{ts()} Stats done.")


def process_dense(data):
    print(f"{ts()} Processing dense features...")
    data = np.log1p(data, dtype=np.float32)
    print(f"{ts()} Processing done.")
    return data


def convert(
    input_dir: str,
    output_dir: str,
    days: int = DAYS,
    int_columns: int = INT_FEATURE_COUNT,
    sparse_columns: int = CAT_FEATURE_COUNT,
    path_manager_key: str = PATH_MANAGER_KEY
):
    print(f"{ts()} Converting data...")
    dataset = load_dataset(input_dir, "train")

    stats(dataset)

    path_manager = PathManagerFactory().get(path_manager_key)

    # Slice and save each portion into dense, sparse and labels
    curr_first_row = 0
    curr_last_row = 0
    for d in range(0, days):
        curr_last_row += ROWS_PER_DAY[d]

        if d == DAYS - 1:
            print(f"{ts()} Handling the last day...")
            dataset = load_dataset(input_dir, "test+val")
            stats(dataset)
            curr_first_row = 0
            curr_last_row = ROWS_PER_DAY[d]

        # write dense columns
        dense_file = os.path.join(
            output_dir, f"day_{d}_dense.npy"
        )
        with path_manager.open(dense_file, "wb") as fout:
            print(
                f"{ts()} Writing rows {curr_first_row}-{curr_last_row-1} dense file: {dense_file}"
            )
            np.save(
                fout,
                process_dense(dataset[curr_first_row:curr_last_row, 1:(int_columns + 1)])
            )

        # write sparse columns
        sparse_file = os.path.join(
            output_dir, f"day_{d}_sparse.npy"
        )
        with path_manager.open(sparse_file, "wb") as fout:
            print(
                f"{ts()} Writing rows {curr_first_row}-{curr_last_row-1} sparse file: {sparse_file}"
            )
            np.save(
                fout,
                dataset[curr_first_row:curr_last_row, (int_columns + 1):(int_columns + sparse_columns + 1)],
            )

        # write labels columns
        labels_file = os.path.join(
            output_dir, f"day_{d}_labels.npy"
        )
        with path_manager.open(labels_file, "wb") as fout:
            print(
                f"{ts()} Writing rows {curr_first_row}-{curr_last_row-1} labels file: {labels_file}"
            )
            np.save(
                fout,
                dataset[curr_first_row:curr_last_row, :1],
            )

        curr_first_row = curr_last_row
    print(f"{ts()} Done.")


if __name__ == "__main__":
    args = parse_args()
    convert(args.input_dir, args.output_dir)
