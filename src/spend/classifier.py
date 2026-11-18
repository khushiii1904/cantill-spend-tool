"""Classify lines into ANY category tree read from a file (no category names in code).
Zero-label method: TF-IDF (char + word) similarity between line text and each leaf's path text,
then smoothed over the line's item group and supplier group.  If labelled rows are supplied, a supervised
model is trained and blended in.  Optional: sentence-embeddings (if installed) are added to the similarity."""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from .items import clean_text, BLANKISH

NLEV = 4


def load_taxonomy(path, level_cols=None):
    t = pd.read_csv(path, dtype=str).fillna("")
    cols = level_cols or list(t.columns)
    t = t[cols].apply(lambda s: s.str.strip())
    leaves, seen = [], set()
    for row in t.itertuples(index=False):
        p = [x for x in row]
        while p and p[-1] == "":
            p.pop()
        if not p or tuple(p) in seen:
            continue
        seen.add(tuple(p)); leaves.append(tuple(p))
    # a node that is only an inner node of a longer path is not a leaf
    prefixes = {l[:i] for l in leaves for i in range(1, len(l))}
    leaves = [l for l in leaves if l not in prefixes] or leaves
    return leaves


def _leaf_doc(p):
    words = [re.sub(r"([a-z])([A-Z])", r"\1 \2", x) for x in p]
    txt = " ".join(clean_text(w) for w in words)
    return txt + " " + clean_text(words[-1])       # lowest level counts double


def _blank(t):
    return t in BLANKISH or len(t) < 3


def _group_mean(S, groups, skip=()):
    """Replace each row by the mean of its group's rows (rows of 'skip' groups are left unchanged)."""
    g = pd.Series(groups).astype("category")
    codes = g.cat.codes.values
    n, k = len(codes), len(g.cat.categories)
    G = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, k))
    sums = G.T @ S
    cnt = np.asarray(G.sum(axis=0)).ravel()
    M = (sums / cnt[:, None])
    out = np.asarray(M)[codes]
    for sk in skip:
        m = (g == sk).values
        out[m] = S[m]
    return out


def classify(texts, supplier_keys, item_keys, leaves, labels=None, embed_model=None, log=print):
    """texts/supplier_keys/item_keys: aligned Series of included lines.  leaves: list of tuples.
    labels: optional Series (same index) of leaf-tuples or None per row.  Returns DataFrame l1..l4, confidence."""
    ct = texts.map(clean_text)
    docs = [_leaf_doc(p) for p in leaves]
    fit_texts = pd.concat([ct[ct.map(lambda t: not _blank(t))].drop_duplicates(), pd.Series(docs)])
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(fit_texts)
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9]{2,}").fit(fit_texts)
    f = lambda s: sp.hstack([cv.transform(s) * 0.6, wv.transform(s) * 0.8]).tocsr()
    XL, XT = normalize(f(ct)), normalize(f(pd.Series(docs)))
    S = (XL @ XT.T).toarray()
    if embed_model:                                           # optional, only if sentence-transformers is installed
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(embed_model)
            ut = ct.drop_duplicates()
            E = dict(zip(ut, m.encode(list(ut), batch_size=128, normalize_embeddings=True, show_progress_bar=False)))
            EL = np.vstack([E[t] for t in ct]); ET = m.encode([" ".join(p) for p in leaves], normalize_embeddings=True)
            S = 0.5 * S + 0.5 * np.clip(EL @ ET.T, 0, None)
        except Exception as ex:
            log(f"[classifier] embeddings unavailable ({type(ex).__name__}); using TF-IDF only")
    blank = ct.map(_blank).values
    S[blank] = 0.0
    item_part = _group_mean(S, item_keys.values, skip=["I:blank"])
    sup_part = _group_mean(S, supplier_keys.values)
    Z = 0.55 * S + 0.30 * item_part + 0.15 * sup_part         # zero-label score

    final = Z
    if labels is not None and labels.notna().sum() >= 30:
        final = _blend_supervised(Z, ct, supplier_keys, item_keys, leaves, labels, f, log)

    top = np.argsort(-final, axis=1)[:, :2]
    r = np.arange(len(final))
    t1, t2 = final[r, top[:, 0]], final[r, top[:, 1]] if final.shape[1] > 1 else np.zeros(len(final))
    conf = 0.5 * np.clip(t1 / 0.5, 0, 1) + 0.5 * np.clip((t1 - t2) / np.maximum(t1, 1e-9), 0, 1)
    conf = np.clip(conf, 0.02, 0.99)
    out = pd.DataFrame("", index=texts.index, columns=[f"l{i+1}" for i in range(NLEV)])
    for i, k in enumerate(top[:, 0]):
        for j, name in enumerate(leaves[k][:NLEV]):
            out.iat[i, j] = name
    out["confidence"] = np.round(conf, 3)
    return out


def _blend_supervised(Z, ct, sup, item, leaves, labels, f, log):
    from sklearn.linear_model import SGDClassifier
    idx = {p: i for i, p in enumerate(leaves)}
    ok = labels.map(lambda p: p in idx if isinstance(p, tuple) else False).values
    if ok.sum() < 30:
        return Z
    toks = CountVectorizer(token_pattern=r"\S+", binary=True).fit(("SUP_" + sup + " ITM_" + item).values)
    X = sp.hstack([f(ct), toks.transform(("SUP_" + sup + " ITM_" + item).values) * 0.8]).tocsr()
    y = np.array([idx[p] for p in labels[ok]])
    clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=25, tol=None, random_state=0, n_jobs=-1).fit(X[ok], y)
    P = np.zeros_like(Z)
    P[:, clf.classes_] = clf.predict_proba(X)
    log(f"[classifier] supervised model trained on {int(ok.sum())} labelled rows, {len(clf.classes_)} classes")
    Zn = Z / np.maximum(Z.max(axis=1, keepdims=True), 1e-9)
    return 0.8 * P + 0.2 * Zn * 0.6
