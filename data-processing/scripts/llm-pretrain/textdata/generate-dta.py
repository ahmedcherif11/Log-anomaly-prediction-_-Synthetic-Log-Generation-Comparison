import os
import argparse
from datasets import Dataset, disable_progress_bars


def main(args):
    datasets_path = args.dataset
    out_path = args.out

    disable_progress_bars()
    logs = Dataset.from_text(datasets_path)
    data = logs.train_test_split(test_size=0.001)

    data.save_to_disk(out_path)


device_map = "auto"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", type=str, help="dataset path")
    parser.add_argument("--out", action="store", type=str, help="folder where to save processed dataset")
    args_pars = parser.parse_args()

    main(args_pars)