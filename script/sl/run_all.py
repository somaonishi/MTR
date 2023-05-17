import argparse
import subprocess
from pathlib import Path


def main(args):
    script_path = Path(__file__).parent / "all.sh"
    if "," in args.data:
        for data in args.data.split(","):
            cmd = [
                str(script_path),
                data,
                args.train_size,
                args.max_seed,
            ]
            subprocess.run(cmd)
    else:
        cmd = [
            str(script_path),
            args.data,
            args.train_size,
            args.max_seed,
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("train_size", type=str)
    parser.add_argument("-m", "--max-seed", type=str, default="10")

    args = parser.parse_args()
    main(args)
