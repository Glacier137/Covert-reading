# Workspace

This directory is the single input and output root used by the figure-reproduction and model code. The public Source Data workbook and brain-image assets are included in the repository.

## Figure reproduction

The complete public input set for figure reproduction is included:

```text
workspace/
  Source_Data.xlsx
  elecs/
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

Run the figure command directly from the repository root as described in the main `README.md`. The reproduction runner reads `Source_Data.xlsx` without modifying it and skips notebook cells that would write derived values back into the workbook.

Within `Source_Data.xlsx`, all `Fig.` sheets appear first and all `Ext. Fig.` sheets appear second. Sheet names are sorted alphabetically within each group.

## Model code

Model retraining uses additional non-public inputs below the same workspace:

```text
workspace/
  model_data/
  private/
    electrode_lists.json
  model_results/
  speech_checkpoints/
```

Participant data, private electrode indices, checkpoints, and generated results are excluded by `.gitignore` and must not be committed to the public repository. `Source_Data.xlsx` and `elecs/` are intentionally tracked because they are the public inputs required to redraw the figures.
