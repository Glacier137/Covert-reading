# Covert Reading: figure reproduction and neural decoding models

This repository contains the public code accompanying the Covert Reading study. It provides:

1. twelve Jupyter notebooks that redraw the reported figures from `Source_Data.xlsx`; and
2. source code for the syllable-classification, ECoG-to-log-mel, and articulatory-movement models.

The Source Data workbook and static brain images required by the figure notebooks are included in the repository. Figure reproduction can therefore be run directly after cloning the repository and installing the environment.

## Repository structure

```text
Covert_reading/
  notebooks/                         # Source Data figure notebooks
  src/covert_reading_figures/        # Reproduction command-line runner
  provenance/notebook_mapping.csv    # Mapping from source code to normalized notebooks
  workspace/
    README.md                         # Workspace description
    Source_Data.xlsx                  # Included Source Data workbook
    elecs/
      MNI.png                         # Included group-brain background
      Brain2D/                        # Included participant brain images
  model_code/
    electrode_lists.example.json
    classifier_and_speech_synthesizer/
    articulatory_movement_synthesizer/
  environment.yml                    # Figure-reproduction environment
  requirements.txt                   # Figure-reproduction Python requirements
  pyproject.toml                     # Installable figure runner
  LICENSE                            # Apache License 2.0 for project code
```

The notebooks are the distributed implementations of the figure code. Separate Python exports of the same notebook cells are intentionally not included.

## Reproducing figures from Source Data

### 1. Create the environment

Python 3.9 is recommended. From the repository root:

```bash
conda env create -f environment.yml
conda activate covert-reading-figures
```

Alternatively, with Python 3.9 or 3.10:

```bash
python -m pip install -e .
```

### 2. Check the included workspace

The Source Data workbook and required brain images are already included:

```text
workspace/
  Source_Data.xlsx                    # included in the repository
  elecs/                              # included in the GitHub repository
    MNI.png
    Brain2D/
      HS44_brain2D.png
      HS45_brain2D.png
      HS47_brain2D.png
      HS48_brain2D.png
      HS50_brain2D.png
      HS54_brain2D.png
      HS71_brain2D.png
      HS73_brain2D.png
      HS76_brain2D.png
      HS78_brain2D.png
```

No additional figure input files need to be downloaded or moved. The runner checks the workbook and all included brain images before executing any notebook. `Source_Data.xlsx` is treated as read-only by the reproduction workflow; notebook cells that would write derived values back into the workbook are skipped.

The workbook tabs are organized with all `Fig.` sheets first and all `Ext. Fig.` sheets second; sheet names are sorted alphabetically within each group.

The repository `.gitignore` keeps `workspace/Source_Data.xlsx`, `workspace/elecs/`, and `workspace/README.md` under version control while excluding private model data, electrode selections, checkpoints, and generated results.

### 3. Run the notebooks

Run all figure notebooks:

```bash
covert-reading-figures --output-dir results/reproduction_run
```

The output directory must be new or empty. The equivalent module command is:

```bash
python -m covert_reading_figures --output-dir results/reproduction_run
```

To run selected notebooks only, use their normalized names:

```bash
covert-reading-figures \
  --output-dir results/fig5_run \
  --only 08_fig5_akt_decoding_source_data 09_fig5_accuracy_mcd_source_data
```

Each run creates:

```text
results/reproduction_run/
  input_inventory.json    # input paths and SHA256 values
  run_summary.json        # notebook-level execution summary
  logs/                   # cell-level execution records
  reproduced/             # captured and copied figure outputs
```

The panel-to-notebook mapping is provided in [`provenance/notebook_mapping.csv`](provenance/notebook_mapping.csv).

### Figure-specific notes

- Figure 1f and Figure 2a were prepared in GraphPad Prism 9.5.1; no Python plotting notebook is expected for these two panels.
- The figure notebooks redraw reported values from Source Data.

## Neural decoding and synthesis code

The model code is documented separately:

| Component | Framework | Documentation |
| --- | --- | --- |
| Syllable classifier | TensorFlow/Keras | [`model_code/classifier_and_speech_synthesizer/README.md`](model_code/classifier_and_speech_synthesizer/README.md) |
| ECoG-to-log-mel/MCD notebook | PyTorch | [`model_code/classifier_and_speech_synthesizer/README.md`](model_code/classifier_and_speech_synthesizer/README.md) |
| Articulatory-movement model | PyTorch/VQGAN | [`model_code/articulatory_movement_synthesizer/README.md`](model_code/articulatory_movement_synthesizer/README.md) |

All model paths default to the same repository-level `workspace/` directory. See [`model_code/README.md`](model_code/README.md) for the shared layout and environment-variable overrides.

### Electrode selections

To use your electrode indices. Copy:

```text
model_code/electrode_lists.example.json
```

to:

```text
workspace/private/electrode_lists.json
```

Then replace the empty arrays with the electrode indices used for each participant and experiment. The classifier, ECoG-to-log-mel notebook, and articulatory-movement notebook read this JSON file directly.

## Citation

If this code contributes to a publication, please cite:

```bibtex
@article{Zhao2025.05.27.656311,
  author = {Zhao, Zehao and Wang, Zhenjie and Liu, Yan and Qian, Youkun and Yin, Yuan and Gao, Xiaowei and Yuan, Binke and Tong, Shelley Xiuli and Tian, Xing and Chen, Gao and Li, Yuanning and Lu, Junfeng and Wu, Jinsong},
  title = {Neural hierarchy for coding articulatory dynamics in speech imagery and production},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.05.27.656311},
  url = {https://www.biorxiv.org/content/early/2025/05/31/2025.05.27.656311}
}
```

## License and third-party code

Original code in this repository is distributed under the Apache License 2.0; see [`LICENSE`](LICENSE).

The VQGAN implementation in `model_code/articulatory_movement_synthesizer/models/` contains code adapted from [dome272/VQGAN-pytorch](https://github.com/dome272/VQGAN-pytorch). That code retains its upstream MIT License in [`LICENSE-VQGAN-MIT.txt`](model_code/articulatory_movement_synthesizer/models/LICENSE-VQGAN-MIT.txt). See the model-directory README for details of the adaptation.
