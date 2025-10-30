from .pipeline import run
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
run(args.config, args.out)
