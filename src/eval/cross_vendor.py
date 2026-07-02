"""Leave-one-vendor-out validation harness. Operationalizes the Siemens/GE/Philips domain-shift caveat."""

# Natural vendor partition across the public coronary datasets:
VENDOR_SPLITS = {
    "arcade":  "philips_siemens",   # Philips Azurion / Siemens Artis Zee
    "dca1":    "mexico_imss",        # institutional
    "xcad":    "ge_innova",          # GE Innova IGS 520
    "danilov": "siemens_ge",         # Coroscop (Siemens) + Innova (GE)
}

def leave_one_vendor_out(datasets):
    """Yield (train_vendors, test_vendor) folds so the held-out vendor never appears in training."""
    vendors = sorted(set(VENDOR_SPLITS[d] for d in datasets))
    for held in vendors:
        train = [v for v in vendors if v != held]
        yield train, held

def report_gap(in_domain_score, held_out_score):
    """Domain-shift gap = in-domain minus held-out-vendor metric. Gate: gap <= agreed bound."""
    return round(in_domain_score - held_out_score, 4)

if __name__ == "__main__":
    for tr, te in leave_one_vendor_out(["arcade", "xcad", "danilov"]):
        print(f"train={tr}  ->  test(held-out)={te}")
    print("example gap:", report_gap(0.78, 0.71))
# TODO: wire to src.train + src.eval.metrics; emit a per-vendor Dice/F1 table + the worst-case gap.
