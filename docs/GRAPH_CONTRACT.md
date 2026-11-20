# Frozen graph contract

**Status:** signed for model-facing work.  
**Bundle:** `autoimmune_owned_v1` only.  
**Measured:** 2026-11-20, local directory `artifacts/local/owned_autoimmune_v1`.  
**Ordered graph hash:** `4af13ede84b3bfd2829fd10ecfea2ea974797e8d2b9d52c81e3525d939b147d2`

`complex_disease_owned_v1` remains a separate preparation fixture. It is not this contract.

## Identity

| Field | Value |
| --- | --- |
| Protocol | `autoimmune_owned_v1` |
| DOID root | 417 |
| Fold seed | 42 |
| Preprocess seed | 42, consumed by the area mask |
| Input seed | 0 |
| Input width | 100 |
| Input dtype | float32 |
| Input generator | `numpy.random.Generator(PCG64)` |
| Hops | 2 |
| Area test size | 0.05 |
| Node-type order | alphabetical |
| Relation order | first appearance in the reverse-augmented training table |

Node-type order and counts:

| Order | Type | Nodes |
| ---: | --- | ---: |
| 0 | anatomy | 14033 |
| 1 | biological_process | 28642 |
| 2 | cellular_component | 4176 |
| 3 | disease | 17080 |
| 4 | drug | 7957 |
| 5 | effect/phenotype | 15311 |
| 6 | exposure | 818 |
| 7 | gene/protein | 27610 |
| 8 | molecular_function | 11169 |
| 9 | pathway | 2516 |
|  | total | 129312 |

Training graph: 50 canonical relations and 6,406,350 directed endpoints. The ordered list is:

1. `gene/protein, protein_protein, gene/protein`
2. `drug, drug_protein, gene/protein`
3. `drug, contraindication, disease`
4. `drug, indication, disease`
5. `drug, off-label use, disease`
6. `drug, drug_drug, drug`
7. `gene/protein, phenotype_protein, effect/phenotype`
8. `effect/phenotype, phenotype_phenotype, effect/phenotype`
9. `disease, disease_phenotype_negative, effect/phenotype`
10. `disease, disease_phenotype_positive, effect/phenotype`
11. `gene/protein, disease_protein, disease`
12. `disease, disease_disease, disease`
13. `drug, drug_effect, effect/phenotype`
14. `biological_process, bioprocess_bioprocess, biological_process`
15. `molecular_function, molfunc_molfunc, molecular_function`
16. `cellular_component, cellcomp_cellcomp, cellular_component`
17. `gene/protein, molfunc_protein, molecular_function`
18. `gene/protein, cellcomp_protein, cellular_component`
19. `gene/protein, bioprocess_protein, biological_process`
20. `exposure, exposure_protein, gene/protein`
21. `exposure, exposure_disease, disease`
22. `exposure, exposure_exposure, exposure`
23. `exposure, exposure_bioprocess, biological_process`
24. `exposure, exposure_molfunc, molecular_function`
25. `exposure, exposure_cellcomp, cellular_component`
26. `pathway, pathway_pathway, pathway`
27. `gene/protein, pathway_protein, pathway`
28. `anatomy, anatomy_anatomy, anatomy`
29. `gene/protein, anatomy_protein_present, anatomy`
30. `gene/protein, anatomy_protein_absent, anatomy`
31. `gene/protein, rev_drug_protein, drug`
32. `disease, rev_contraindication, drug`
33. `disease, rev_indication, drug`
34. `disease, rev_off-label use, drug`
35. `effect/phenotype, rev_phenotype_protein, gene/protein`
36. `effect/phenotype, rev_disease_phenotype_negative, disease`
37. `effect/phenotype, rev_disease_phenotype_positive, disease`
38. `disease, rev_disease_protein, gene/protein`
39. `effect/phenotype, rev_drug_effect, drug`
40. `molecular_function, rev_molfunc_protein, gene/protein`
41. `cellular_component, rev_cellcomp_protein, gene/protein`
42. `biological_process, rev_bioprocess_protein, gene/protein`
43. `gene/protein, rev_exposure_protein, exposure`
44. `disease, rev_exposure_disease, exposure`
45. `biological_process, rev_exposure_bioprocess, exposure`
46. `molecular_function, rev_exposure_molfunc, exposure`
47. `cellular_component, rev_exposure_cellcomp, exposure`
48. `pathway, rev_pathway_protein, gene/protein`
49. `anatomy, rev_anatomy_protein_present, gene/protein`
50. `anatomy, rev_anatomy_protein_absent, gene/protein`

A later model reads this order as relation-parameter rows 0 through 49. Inputs are one float32 matrix per type, shape `(count, 100)`, in the type order above. They are loaded, not redrawn.

## What the preprint totals are

The 2023 preprint reports 123,527 nodes and 8,063,026 undirected edges for the compiled biomedical graph. This contract's 129,312 nodes are the typed endpoints of the processed release after representative orientation. Its 6,406,350 integers are directed training endpoints, including reverse rows and excluding held-out edges. Those two descriptions are not interchangeable, and the preprint totals are not acceptance targets for this bundle.

## Not included

Historical DGL object identity, Torch input bit patterns, and the unseeded upstream area mask are outside this contract. The local bundle files stay out of Git.
