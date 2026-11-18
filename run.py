"""One command:  python run.py --config configs/A.yaml --name khushi [--suffix 2] [--no-labels] [--eval]"""
import argparse, os, time, yaml, pandas as pd
from src.spend import two_system, single_system
from src.spend.common import finish, load_labels, COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--name", default="yourname")
    ap.add_argument("--suffix", default="", help="e.g. 2 -> yourname__A2.csv"); ap.add_argument("--out", default="output")
    ap.add_argument("--no-labels", action="store_true", help="ignore any labels (zero-label mode)")
    ap.add_argument("--eval", action="store_true", help="score against labels if the config has them")
    ap.add_argument("--set", action="append", default=[], help="override a config value, e.g. --set file.path=data/x.csv")
    a = ap.parse_args(); t0 = time.time()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:                                   # lets you point a frozen config at new files without editing code
        k, v = kv.split("=", 1); node = cfg; parts = k.split(".")
        for p in parts[:-1]: node = node[p]
        node[parts[-1]] = v
    labels = None if a.no_labels else load_labels(cfg)
    print(f"[run] {cfg['name']}  mode={cfg['mode']}  labels={'yes' if labels is not None else 'no'}")
    if cfg["mode"] == "two_system":
        df = two_system.run(cfg)
    else:
        df = single_system.run(cfg, labels=labels)
    df = finish(df, cfg, labels=labels)
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/{a.name}__{cfg['output_prefix']}{a.suffix}.csv"
    df[COLS].to_csv(path, index=False)
    df.to_pickle(f"{a.out}/_{cfg['name']}_debug.pkl")
    print(f"[run] wrote {path}: {len(df)} rows, included {int(df.include.sum())}, spend {df.spend.sum():,.0f} {cfg['reporting_currency']}, {time.time()-t0:.0f}s")
    print(df.exclusion_reason.replace("", "INCLUDED").value_counts().to_dict())
    if a.eval:
        from src.spend.evaluate import evaluate
        evaluate(df, load_labels(cfg), cfg)


if __name__ == "__main__":
    main()
