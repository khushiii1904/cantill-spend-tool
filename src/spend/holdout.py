"""Dev tool: fair test of the label-trained classifier. Hide 30% of labelled rows, train on 70%, score the hidden 30%.
usage: python -m src.spend.holdout configs/A.yaml"""
import sys, yaml, numpy as np, pandas as pd
from .common import load_labels, finish


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path)); labels = load_labels(cfg)
    df = pd.read_pickle(f"output/_{cfg['name']}_debug.pkl").drop(columns=["l1", "l2", "l3", "l4", "confidence"], errors="ignore")
    ids = labels.index.intersection(df.record_id)
    rng = np.random.RandomState(0); test = set(rng.choice(ids, int(0.3 * len(ids)), replace=False))
    train_labels = labels[~labels.index.isin(test)]
    out = finish(df.drop(columns=["item_key"], errors="ignore"), cfg, labels=train_labels)
    t = out[out.record_id.isin(test) & (out.include == 1)].merge(labels, left_on="record_id", right_index=True, suffixes=("", "_g"))
    t = t[t.leaf.map(lambda x: isinstance(x, tuple) and len(x) > 0)]
    for lvl in (1, 2, 3):
        pred = t[[f"l{i}" for i in range(1, lvl + 1)]].apply(tuple, axis=1)
        gold = t.leaf.map(lambda x: tuple(x[:lvl]) if len(x) >= lvl else tuple(x + (x[-1],) * (lvl - len(x))))
        hit = pred == gold; w = t.spend.abs()
        print(f"[holdout] L{lvl}: lines {hit.mean():.3f}  spend-weighted {(hit * w).sum() / max(w.sum(), 1):.3f}  (n={len(t)})")


if __name__ == "__main__":
    main(sys.argv[1])
