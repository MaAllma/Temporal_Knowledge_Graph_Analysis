# Visual Analytics Project SS24

## Vast Challenge 2024
Our answers for challenge 3 can be found [here](Vast_Challenge_Answers_RTPU-Sohns_MC3.md)

## Getting started

First clone the repository
```
cd your_parent_folder
git clone https://gitlab.rhrk.uni-kl.de/iselborn/visual-analytics-project-ss24.git
```

Now download the [VAST MC3-data](https://vast-challenge.github.io/2024/MC3.html) and copy `mc3.json` and the 
`Oceanus Information` folder to `/data` (note that the data folder is not tracked by git, so it may need to be 
created). (You should now have copied `/data/mc3.json` and `/data/Oceanus Information/*`)

Then setup a conda environment for the project:
```
conda create -n vis_proj python=3.11 tqdm numpy scipy pandas xarray matplotlib seaborn bokeh numba scikit-learn=1.4.2 pytorch=2.2.1 torchvision torchaudio pytorch-cuda=12.1 torchmetrics conda-forge::networkx python-graphblas  cuda-version=12.2 -c conda -c conda-forge -c pytorch -c nvidia
conda activate vis_proj
//pip install -U networkx[default] python-graphblas[default] graphblas-algorithms
// cudf=24.04 cuml=24.04 cugraph=24.04 nx-cugraph -c rapidsai
```
and use it as appropiate for your IDE.
## Organizational

- Information files for the [VAST-Challenge](VAST%202024.md) are added to the Repository
