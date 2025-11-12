# IDEA Datasets

Place all experimental data artifacts under this directory so the new pipeline
remains self-contained.

Recommended layout:

```
idea/datasets/
  sts_train.csv              # training split (question, tag)
  sts_eval.csv               # optional eval split
  tag_to_answer.json         # answer texts per tag
  features/
    manual_ngrams.json       # generated via idea/featurizer/generate_features.py
    auto_ngrams.json         # alternate feature set
```
