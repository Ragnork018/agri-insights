# Deployment and acceptance checklist

- Install with `pip install -e ".[dev,dashboard]"`.
- Run the CLI on a documented dataset and inspect all written artefacts.
- Build the dashboard image with Docker only after installing the dashboard extra.
- Verify target units, label mappings, spatial boundaries and temporal cutoff before accepting any metric.
- Run a later-year holdout and a separate held-out-district evaluation before operational use.
- Record dataset version, checksum, feature cutoffs, random seed, package versions and model checksum.
- Do not convert district-level accuracy into plot-level guarantees or agronomic advice.
