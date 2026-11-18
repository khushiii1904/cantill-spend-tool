# DECISIONS (what I found, what I assumed)

## Company A
- ERP values are net; tool amounts are gross with VAT typed 3 ways (`15`, `0.15`, `15%`) or blank -> parsed all, blank = 15% (rule from brief).
- ~7,900 tool amounts use thousands commas (`3,552.97`) -> custom number parser.
- Tool dates: several formats and 65% blank -> matching never relies on dates. Impossible dates are flagged, not trusted.
- PO references are reformatted (`PO#4510108243`, `4510019411/20.0`) -> extract the 10-digit PO number.
- ERP file has two layouts (about 24k rows have no date/vendor id/site but do have supplier name) -> only supplier name, PO, amount, text are used.
- Matching order: PO+amount (0.5% or <1 ZAR), PO-level sum, supplier+amount for missing/unknown PO. A tool line that matches is DUPLICATE (ERP record counted).
  ~3,600 tool lines match nothing and are counted as spend that exists only in the tool. Risk: two different purchases with the same supplier and amount.
- Status: approved/paid count; cancelled -> CANCELLED; rejected/on hold/other -> NOT_APPROVED. Exact repeated tool lines -> DUPLICATE (exported twice).
- ERP negative values (reversals) are kept as negative spend so they net out. No tax/intercompany-looking suppliers or texts were found in A, so no NON_PROCUREMENT rows.
- Supplier labels: one golden supplier group holds ~21k lines and 122 different names (almost all rows without vendor id). This looks like a catch-all in the labels,
  so I report supplier accuracy on the other 69 groups (my 69 groups match exactly there). Item labels also contain big catch-all items (e.g. 794 texts in one), so item precision is understated.
- OCR text fixes: `0` between letters -> `o`; supplier spaces removed (`AL TON` = `ALTON`).

## Company B
- No status column: all lines treated as approved/paid; seed labels contain only NON_PROCUREMENT exclusions (no DUPLICATE), so I do not de-duplicate B.
- NON_PROCUREMENT = intercompany GL descriptions (`I/C Payable`, `IC Payables`, `IC05`...) + tax text (`Sales Tax Expense`) via default keyword rules;
  when seed labels exist, GL descriptions / supplier types / vendors that were >=90% excluded in the seed are excluded too.
- USD = amount / currency_rate; USD lines as is; missing rate -> FX table by invoice month, then posting month, then nearest month, then median line rate. 3 unparseable amounts -> spend 0.
- Reversals (negative lines, e.g. B000001/B000002) kept: they net to zero.
- Supplier key = vendor id (no names). Item key = material number, else cleaned description.
- Blank descriptions: classified from item/supplier group scores, with low confidence.

## Company C / general
- The tree is read from file (any depth up to 4, leaf = deepest non-blank path); no category names in code.
- Exclusion defaults are a keyword list in `configs/defaults.yaml` matched on columns named in the config.
- Seed/golden labels are used only when the config points to them; the tool runs with `--no-labels` too.
