# APPROACH

**AI used:** Claude (Anthropic chat assistant) helped me design, write and debug the code and the docs; I reviewed and ran it.
No LLM or paid API is called at run time: the tool is local (pandas, scikit-learn) and costs **$0 per 1,000 lines**.

**Pipeline** (config-driven): load -> clean amounts/dates -> exclusions -> count once -> supplier keys -> item keys -> classify -> write.
- *Cleaning:* number parser handles `1,234.56` / `1.234,56` / `(12)`; A's gross amounts -> net = gross/(1+VAT) (blank VAT = 15%); B -> USD = amount / line rate,
  else monthly FX table (invoice month, then posting month, nearest month, median line rate). Dates: many formats tried, failures flagged.
- *Exclusions* (fixed priority CANCELLED > NOT_APPROVED > DUPLICATE/NON_PROCUREMENT): status values from config; keyword rules (default list + config);
  if labels exist, value rules are learned (a column value where >=90% of >=5 labelled rows were excluded).
- *Count once (A):* tool line is DUPLICATE if (1) same PO + amount within 0.5%, (2) invoice = sum of all receipts on the PO, (3) no/odd PO but same supplier + amount.
  Exact repeats of a tool invoice are DUPLICATE (exported twice). Tool-only lines stay as spend.
- *Suppliers:* normalise (case, legal suffixes, OCR space splits), merge near-identical spellings (difflib >= 0.90); B uses vendor id.
- *Items:* a code defines an item; codes merge only through a specific shared text; codeless lines join their text's code, otherwise a cluster of
  near-identical cleaned texts (char TF-IDF cosine >= 0.90); blank/generic text -> `I:blank`.
- *Classifier (unseen tree, zero labels):* tree read from file; each leaf = its path text. TF-IDF (char 3-5 + word 1-2) cosine between line text and leaf text,
  smoothed with the mean score of the line's item group (30%) and supplier group (15%). Confidence = absolute similarity + margin to 2nd best.
  If labels exist a linear model (text + supplier + item tokens) is trained and blended (80/20). Optional sentence embeddings if installed.

**Measured (A, labels hidden, `--no-labels --eval`, ERP lines):** L1 57% / L2 37% / L3 29% of spend correct (44% / 21% / 15% of lines).
Supplier dedup precision 1.00 / recall 1.00 (on the clean part of the labels, see DECISIONS). Item dedup precision 0.53 / recall 0.89 (labels contain big catch-all items).
**With labels (random 30% of labelled rows hidden):** A L3 98% of lines / 95% of spend; B L3 94% / 99% (optimistic: items recur between rows).
**B zero-label:** exclusions precision 1.00 / recall 0.89 vs the noisy seed; L3 spend-weighted 45% vs seed labels.
**Runtime:** A 65k rows 23 s (zero-label) / 38 s (with labels); B 41k rows 5-7 s on a laptop-class CPU. Limit is 60 min per 25k lines.

**Known weaknesses:** zero-label accuracy is limited by short/abbreviated texts and blank descriptions (these rely on item/supplier smoothing);
no multilingual model by default; item dedup is conservative on generic texts.
