# Result CSVs from the same user-provided zip

55 CSV files (36 per-seed summaries and 19 per-epoch logs), copied byte-for-byte from the zip whose SHA-256 is `f1af1a3d268527110d6bcca9120a7760931bfffde02e7736749bb1160dc87752`.

Zip entry timestamp on these members: 2026-07-05 05:55. That is archive metadata, not a verified time for each training run.

Some summary rows point `epoch_csv` at `/content/drive/MyDrive/mgosc_results/<filename>`. Those strings were left unchanged. Every non-empty path's filename is one of the 19 epoch logs in this directory.

Neither surviving trainer writes this schema exactly. Read `artifacts/AUDIT.md` before treating a script as the producer.

Per-file SHA-256 sums are in `artifacts/SHA256SUMS`.
