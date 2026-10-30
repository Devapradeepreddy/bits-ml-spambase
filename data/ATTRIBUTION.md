# Dataset attribution

Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999).
Spambase [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C53G6X

Dataset page: https://archive.ics.uci.edu/dataset/94/spambase

Download: https://archive.ics.uci.edu/static/public/94/spambase.zip

License: Creative Commons Attribution 4.0 International (CC BY 4.0).
License text: https://creativecommons.org/licenses/by/4.0/

`spambase.data`, `spambase.names`, and `spambase.DOCUMENTATION` are unchanged
copies from the official archive. `feature_schema.csv`, `test_data.csv`, model
artifacts and experimental results are derived from these data. Feature names
come from `spambase.names`; class 1 denotes spam and class 0 denotes non-spam.

Changes: a deterministic, duplicate-group-aware train/test partition; a header
row added to the derived held-out CSV; preprocessing and classifier training.
No original source rows were removed from the modeling experiment.

The 1999 data contain historical and person-specific indicators. They should
not be treated as a representative sample of present-day email traffic.
