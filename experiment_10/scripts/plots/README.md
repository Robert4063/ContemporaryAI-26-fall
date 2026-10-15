# Plotting scripts

All visualization code is kept in this directory.

The scripts resolve the repository root from their own location, read numeric
records from `results/loss` and `results/baseline`, and write generated images
to `results/plots` or `results/evaluation/figures`.

```powershell
python scripts/plots/plot_4charts.py
python scripts/plots/plot_loss.py --versions raw body_only
python scripts/plots/plot_mlp_versions.py --hidden "(100, 50)" --alpha 0.001
python scripts/plots/plot_evaluation.py
```

`plot_4charts.py` produces the four requested comparison figures and removes
only its own previous four outputs before drawing them again.
