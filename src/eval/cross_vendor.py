"""Leave-one-vendor-out validation harness. Operationalizes the Siemens/GE/Philips domain-shift caveat."""

# Natural vendor partition across the public coronary datasets. Each dataset maps to
# a SET of ATOMIC vendors (siemens/ge/philips/imss/...) because several datasets are
# multi-vendor; treating a composite string as one atomic unit understates the domain
# gap (holding out one vendor would leave a different dataset that ALSO has that vendor
# in train).
VENDOR_SPLITS = {
    "arcade":  {"philips", "siemens"},  # Philips Azurion / Siemens Artis Zee
    "dca1":    {"imss"},                # institutional (Mexico IMSS)
    "xcad":    {"ge"},                  # GE Innova IGS 520
    "danilov": {"siemens", "ge"},       # Coroscop (Siemens) + Innova (GE)
}

def leave_one_vendor_out(datasets):
    """Pure function -> list of (train_datasets, held_vendor, eval_datasets) folds.

    Iterate over the ATOMIC vendors present. For each held-out vendor V:
      * eval  = every dataset whose vendor-set contains V, and
      * train = every dataset whose vendor-set does NOT contain V
    so a held-out vendor can never leak into training through a multi-vendor dataset.
    (Model/metric wiring is still a stub -- see TODO below.)"""
    vendor_sets = {d: set(VENDOR_SPLITS[d]) for d in datasets}
    vendors = sorted(set().union(*vendor_sets.values())) if vendor_sets else []
    assert len(vendors) >= 2, f"leave-one-vendor-out needs >=2 distinct vendors, got {vendors}"
    folds = []
    for held in vendors:
        eval_sets = [d for d in datasets if held in vendor_sets[d]]
        train = [d for d in datasets if held not in vendor_sets[d]]
        folds.append((train, held, eval_sets))
    return folds

def report_gap(in_domain_score, held_out_score):
    """Domain-shift gap = in-domain minus held-out-vendor metric. Gate: gap <= agreed bound."""
    return round(in_domain_score - held_out_score, 4)

if __name__ == "__main__":
    for tr, held, ev in leave_one_vendor_out(["arcade", "xcad", "danilov"]):
        print(f"held-out vendor={held}  train={tr}  ->  eval(held-out)={ev}")
    print("example gap:", report_gap(0.78, 0.71))
# TODO: wire to src.train + src.eval.metrics; emit a per-vendor Dice/F1 table + the worst-case gap.
