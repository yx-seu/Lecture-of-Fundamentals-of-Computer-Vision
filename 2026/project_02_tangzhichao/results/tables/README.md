# Results Tables

This directory is used for exported metrics and report-ready tables.

After training or evaluation, run:

```bash
python -m src.export_submission --output_dir outputs/segformer_b0_pet --results_dir results
```

Expected exported files include `summary.json`, `test_metrics.json`, `test_metrics.csv`, and `report_assets.md` when they are available in the selected output directory.
