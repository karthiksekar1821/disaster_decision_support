# Results

All model weights and output files are stored on Google Drive.
Local copies are not kept due to file size (models ~500MB each).

## Drive Location
Google Drive → MyDrive → disaster_project/

## Folder Structure on Drive
```
disaster_project/
├── disaster_models/
│   ├── roberta/best_model/
│   ├── deberta/best_model/
│   └── electra/best_model/
├── disagreement_analysis/
├── ensemble_results/
├── confidence_results/
├── explainability_results/
└── evaluation_results/
```

## Access
- Drive must be mounted in Colab before running any script
- All scripts mount Drive automatically in Cell 1
- Paths are hardcoded as /content/drive/MyDrive/disaster_project/ 