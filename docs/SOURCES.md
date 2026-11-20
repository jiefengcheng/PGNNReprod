# Source ledger (no data bytes)

| Source | Reference | Role |
| --- | --- | --- |
| TxGNN public code | https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023 | Frozen executable comparison oracle |
| TxGNN preprint, 28 Sep 2023 | https://doi.org/10.1101/2023.03.19.23287458 | Research method; graph construction pp. 22–23 |
| Harvard Dataverse | https://doi.org/10.7910/DVN/IXA7BM | Curation snapshot and supporting data publication |
| kg.csv | https://dataverse.harvard.edu/api/access/datafile/7144484 | Upstream graph rows; fetch locally |
| node.csv (tab-separated) | https://dataverse.harvard.edu/api/access/datafile/7144482 | Source node registry; fetch locally |
| edges.csv | https://dataverse.harvard.edu/api/access/datafile/7144483 | Global-index edges; fetch locally |

The upstream disease grouping included manual curation. Rebuilding from newer individual databases is a different exercise; preserve the released identifiers for code-reproduction parity. This repository does not redistribute source data or third-party PDF files.
