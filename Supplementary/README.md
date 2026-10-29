# Supplementary result scripts

Run from the OriGENE repository root. Use the original project environment (numpy, pandas, scipy, scikit-learn, matplotlib). NT2-D1 plotting retains LaTeX rendering and needs TeX. Set ORIGENE_PIPELINE_ROOT if the pipeline directory is not epigenetic_snakemake_pipline. Outputs go to results/supplementary/.

| Script under analysis_scripts/ | Required inputs |
| --- | --- |
| permutation_comparison.py | final_prediction_merged_all.csv in ORIGENE_BENCHMARK_ROOT, default Data/Main_figures/benchmark; gene/model/dataset/label/score columns |
| paired_gene_bootstrap.py | Same CSV; override via ORIGENE_S6_CSV. Produces paired gene-bootstrap 95% intervals |
| padding_prevalence_and_no_padding.py | ORIGENE_PADDING_ROOT/validation/{set}/{source}_exact_padding_validation.tsv and no_padding_heldout/exact_no_padding_heldout_dataset_metrics.tsv |
| matched_promoter_padding_perturbation.py | Pipeline results/07_model_evaluation/reviewer_singletrack/interpretability/original_ng/{all_sources,v3_three_methods}/ |
| regional_perturbation.py | Pipeline results/07_model_evaluation/reviewer_singletrack/biological_interpretability_v2/region_equal_length/ |
| seed_stability.py | Original and repeated-seed results below pipeline results/07_model_evaluation/reviewer_singletrack and reviewer_original_ng/original_ng |
| nt2d1_40kb_markers.py | Pipeline results/08_reviewer4_fig6_40kbp_singlemarkers/{marker}/OOF_predictions.csv |

Example:

```bash
python Supplementary/analysis_scripts/permutation_comparison.py
python Supplementary/analysis_scripts/paired_gene_bootstrap.py
```

Each script specifies its exact expected filenames and column checks. Provide the original accepted inputs; these large result trees are not included. These plotting scripts do not create missing model predictions or perturbation tables. Names describe analyses to avoid relying on historical supplementary numbering. Permutation and bootstrap are research analyses, not development tests, and remain included.
