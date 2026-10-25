# Uncertainty Quantification for Compressed-Sensing MRI Reconstruction

This repository contains research code for comparing three
uncertainty-quantification (UQ) methods for deep-learning-based
compressed-sensing MRI reconstruction:

-   **MC Dropout**: estimates uncertainty from multiple stochastic
    forward passes.
-   **Evidential Learning (EDL)**: predicts reconstruction and
    uncertainty in a single forward pass using an evidential regression
    head.
-   **Deep Ensemble (DE)**: estimates uncertainty from disagreement
    among independently trained models.

The primary reconstruction backbone is **COAST**. A **U-Net** backbone
is also used for additional backbone-level verification. The evaluation
focuses on reconstruction quality and whether uncertainty can indicate
potentially unreliable reconstructions when reference images are
unavailable.

## Repository structure

The repository will use descriptive, consistent filenames rather than
experiment-specific suffixes:

``` text
.
├── models/
│   ├── coast_model_ensemble.py
│   ├── coast_model_MC.py
│   ├── coast_model_evidential.py
│   └── unet_model.py
├── train_COAST_ensemble.py
├── train_COAST_MC.py
├── train_COAST_evidential.py
├── test_COAST_ensemble.py
├── test_COAST_MC.py
├── test_COAST_evidential.py
├── sampling_matrix/          # local sampling masks; not included
├── data/                     # local datasets; not included
├── README.md
└── .gitignore
```

The filenames above describe the intended cleaned repository. The final
folder and U-Net module names should match the source files included in
the release.

## Environment

Use Python with a PyTorch build compatible with the installed CUDA
driver. Install PyTorch using the instructions for your operating system
and CUDA version at <https://pytorch.org/get-started/locally/>.

The code uses common scientific Python packages, including:

``` bash
pip install numpy scipy scikit-image matplotlib tqdm thop
```

`thop` is used for model complexity reporting in the EDL evaluation
code. Matplotlib is required only where plotting or visualization
remains enabled. Exact package versions should be pinned after the code
has been tested in the release environment.

## Data

**Datasets, sampling masks, trained checkpoints, and experiment outputs
are not distributed in this repository.** Users must obtain data from
the relevant dataset providers and comply with each dataset's terms and
access requirements. Do not commit private medical data, subject
identifiers, model checkpoints, or large generated outputs to GitHub.

Prepare the local dataset and sampling-mask files in the locations
expected by the scripts, or update the path arguments/configuration to
point to your own locations. The sampling-mask MATLAB file should
contain the `mask_matrix` variable expected by the implementation. Use
the same preprocessing, image dimensions, intensity normalization,
subject-level split, and sampling ratio as the experiment being
reproduced.

The scripts are designed for 2D MRI inputs. Ensure that the training
data, test images, reference images, and sampling masks are compatible
with the selected experiment. The repository provides code only; it does
not provide data access credentials or redistribute the datasets.

## Main experimental settings

The following settings describe the intended benchmark configuration.
Use the actual arguments and checkpoint configuration associated with a
particular experiment when reproducing results.

  Setting                            Value
  ---------------------------------- ----------------------------------
  Reconstruction backbone            COAST
  Additional backbone verification   U-Net
  Number of COAST layers             9
  Compressed-sensing ratio           30% by default
  Optimizer                          Adam
  Initial learning rate              `1e-4`
  Training duration                  400 epochs
  MC Dropout probability             `0.05`
  MC stochastic forward passes       `T = 8`
  Ensemble size                      `8` independently trained models
  EDL regularization coefficient     `lambda_reg = 0.1`
  `group_num`                        `1`

`group_num=1` is used consistently in the documented commands. It is an
experiment/configuration identifier in the current scripts, not a
separate UQ method.

## Training

Run commands from the repository root. Update the GPU index and local
data/mask paths as needed.

### MC Dropout

``` bash
python train_COAST_MC.py \
  --start_epoch 0 \
  --end_epoch 400 \
  --layer_num 9 \
  --learning_rate 1e-4 \
  --group_num 1 \
  --cs_ratio 30 \
  --gpu_list 0
```

### Evidential Learning

``` bash
python train_COAST_evidential.py \
  --start_epoch 0 \
  --end_epoch 400 \
  --layer_num 9 \
  --learning_rate 1e-4 \
  --group_num 1 \
  --cs_ratio 30 \
  --gpu_list 0 \
  --lambda_reg 0.1
```

### Deep Ensemble (8 models)

Train eight independently initialized models using eight distinct random
seeds:

``` bash
python train_COAST_ensemble.py \
  --ensemble_size 8 \
  --random_seeds 131,132,133,134,135,136,137,138 \
  --end_epoch 400 \
  --layer_num 9 \
  --learning_rate 1e-4 \
  --group_num 1 \
  --cs_ratio 30 \
  --gpu_list 0
```

Each ensemble member must have its own checkpoint. The cleaned release
should save or collect the eight final checkpoints in the format
expected by the ensemble test script, without requiring manual renaming.
Do not overwrite one member's weights with another member's checkpoint.

## Testing

The test scripts evaluate reconstruction quality and uncertainty-based
reliability indicators. The cleaned test versions omit nonessential
figure-generation code while retaining numerical evaluation and result
export.

### Deep Ensemble

``` bash
python test_COAST_ensemble.py \
  --epoch_num 400 \
  --layer_num 9 \
  --group_num 1 \
  --cs_ratio 30 \
  --gpu_list 0 \
  --ensemble_num 8
```

### MC Dropout

``` bash
python test_COAST_MC.py \
  --epoch_num 400 \
  --layer_num 9 \
  --group_num 1 \
  --cs_ratio 30 \
  --gpu_list 0 \
  --dropout_prob 0.05 \
  --T 8
```

### Evidential Learning

``` bash
python test_COAST_evidential.py \
  --epoch_num 400 \
  --layer_num 9 \
  --group_num 1 \
  --cs_ratio 30 \
  --gpu_list 0 \
  --lamR1 0.1 \
  --lamR2 0
```

The commands show the intended public interface for the cleaned scripts.
Before release, confirm that argument names, dataset paths, checkpoint
paths, and output directories in the final source match these examples.
Evaluation should use the same test split, preprocessing, sampling mask,
and checkpoint as the corresponding experiment.

## Evaluation outputs

Depending on the selected method and enabled options, the test scripts
can save numerical reconstruction metrics, uncertainty-related
statistics, per-slice results, foreground-region measurements,
confidence intervals, calibration/failure-detection results, and
data-consistency measurements. Exact filenames are defined by the final
scripts.

Generated outputs are experiment artifacts and should normally remain
outside the source repository. The `.gitignore` should exclude local
datasets, checkpoints, logs, caches, and result folders.

## Citation

If you use this code in academic work, please cite the associated
manuscript and the COAST backbone paper. Add the final journal, year,
and DOI details for the manuscript once they are confirmed.

**Manuscript**

Zhongfu Huang, Haolin Zhan, Rencheng Song, and Yu Liu, *Uncertainty
Quantification in Neural Networks for Compressed Sensing MRI: A
Benchmark Study*.

``` bibtex
@article{huang_uncertainty_cs_mri,
  title   = {Uncertainty Quantification in Neural Networks for Compressed Sensing MRI: A Benchmark Study},
  author  = {Huang, Zhongfu and Zhan, Haolin and Song, Rencheng and Liu, Yu},
  journal = {To be completed},
  year    = {To be completed},
  doi     = {To be completed}
}
```

**COAST backbone**

D. You, J. Zhang, J. Xie, B. Chen, and S. Ma, "COAST: COntrollable
Arbitrary-Sampling NeTwork for Compressive Sensing," *IEEE Transactions
on Image Processing*, vol. 30, pp. 6066--6080, 2021.

## License

A license has not yet been specified. Add a license after confirming
that you have the right to distribute the code and that third-party
components and dataset terms are respected.
