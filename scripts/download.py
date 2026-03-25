import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def download_cmnist(data_path):
    """Download EMNIST digits split for the CMNIST benchmark.

    Uses torchvision's built-in EMNIST downloader. If the NIST server is
    unreachable (known issue), the script exits with instructions for
    manual download.
    """
    from torchvision import datasets

    sub_dir = Path(data_path) / "emnist"

    try:
        logging.info("Downloading EMNIST digits to %s ...", sub_dir)
        datasets.EMNIST(sub_dir, split="digits", train=True, download=True)
        datasets.EMNIST(sub_dir, split="digits", train=False, download=True)
        logging.info("EMNIST downloaded successfully.")
    except Exception as exc:
        raw_dir = sub_dir / "EMNIST" / "raw"
        raise RuntimeError(
            f"EMNIST download failed ({exc}).\n"
            f"The NIST server (biometrics.nist.gov) is frequently down.\n\n"
            f"Manual fix:\n"
            f"  1. Download gzip.zip from an EMNIST mirror\n"
            f"  2. Extract the 'gzip/' folder contents into:\n"
            f"       {raw_dir}/\n"
            f"  3. Gunzip the .gz files so you have files like:\n"
            f"       {raw_dir}/emnist-digits-train-images-idx3-ubyte\n"
            f"  4. Re-run this script (it will skip the download)."
        ) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument(
        "datasets",
        nargs="+",
        type=str,
        choices=["cmnist"],
    )
    parser.add_argument("--data_path", type=str, default="data/benchmark")
    parser.add_argument("--download", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    if args.download:
        for dataset in args.datasets:
            if dataset == "cmnist":
                download_cmnist(args.data_path)
