
Prototype Anomaly Detection PAD
================================

Artifacts:
- pad_dummy_transactions.csv : synthetic dataset (2,500 rows)
- anomaly_scores.csv         : detected anomalies (sorted by severity)
- prototype_anomaly_pad.py   : reusable script to re-run anomaly detection
- This notebook also renders charts below for quick inspection.

How to reuse locally:
1) pip install pandas scikit-learn matplotlib
2) python prototype_anomaly_pad.py pad_dummy_transactions.csv anomaly_scores.csv

Model notes:
- Isolation Forest (unsupervised)
- Features: ratio_paid_expected, month, txn_wp_month, sector_code, kec_code, log_paid, log_expected
- Contamination set to 6% (adjust based on tolerance)
