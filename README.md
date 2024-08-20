# Visual Anomaly Detection in Temporal Knowledge Graphs

This is the repository for our submission to the visualization challenges posed by Mini Challenge 3 of the VAST Challenge 2024, which involves detecting illegal fishing activities within a dynamic network of companies and individuals. The task requires effective anomaly detection in a time-dependent knowledge graph, a scenario where conventional graph visualization tools often fall short due to their limited ability to integrate temporal data and the undefined nature of the anomalies. We demonstrate how to overcome these challenges through well-crafted views in standard software libraries. Our approach involves decomposing the time-dependent knowledge graph into separate time and structure components, as well as providing data-driven guidance for identifying anomalies. These components are then interconnected through extensive interactivity, enabling exploration of anomalies in a complex, temporally evolving network. The source code and a demonstration video are publicly available here.

## Vast Challenge 2024
[![YouTube](http://i.ytimg.com/vi/pOLVmvl17jM/hqdefault.jpg)](https://www.youtube.com/watch?v=pOLVmvl17jM)

Our answers for challenge 3 can be found [here](Vast_Challenge_Answers_MC3.md) and our submission video [here](https://youtu.be/pOLVmvl17jM).
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
conda create -n vis_proj python=3.11 tqdm numpy scipy pandas xarray matplotlib seaborn bokeh numba scikit-learn=1.4.2  conda-forge::networkx python-graphblas  cuda-version=12.2 -c conda -c conda-forge -c pytorch -c nvidia
conda activate vis_proj
```
and use it as appropiate for your IDE.
## Organizational

- Information files for the [VAST-Challenge](VAST%202024.md) are added to the Repository
