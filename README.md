# Uniform spanning tree sampling on grid graphs.

This repository contains the source code of my student research project. We adapt D3PMs for the task of uniform spanning tree sampling on a grid graph, proposing an autoregressive
approach that iteratively refines the sampled edge set along a fixed node schedule.

## Environment setup

```bash
conda create --name D3PM --file requirements.txt
```

## Data Preprocessing
You can create the dataset by running
```bash
python dataset.py --size SIZE --width WIDTH --height HEIGHT --dir DIR
```

### Arguments

- `WIDTH` and `HEIGHT` define the underlying grid graph.
- `SIZE` is the number of spanning trees of the underlying grid graph in the dataset.
- `DIR` is the path of the dataset's root folder (e.g dataset/).

## Training
You can custom the model in model.py and train it using
```bash
python train.py --model_path MODEL_PATH --dataset_path DATASET_PATH --curves_path CURVES_PATH
```

### Arguments

- `MODEL_PATH` is the path of the file in which the model should be stored (e.g models/model.pt).
- `DATASET_PATH` is the path of the dataset's root folder.
- `CURVES_PATH` is the name of the image file featuring the loss and accuracy curves (e.g curves/curves.png).

## Sampling
Run 
```bash
python sampling.py --width WIDTH --height HEIGHT --model_path MODEL_PATH --num_samples NUM_SAMPLES --samples_path SAMPLE_PATH --stats_path STATS_PATH
```
To generate samples and statistics.

### Arguments

- `WIDTH` and `HEIGHT` define the underlying grid graph.
- `MODEL_PATH` is the path in which the model should be stored.
- `NUM_SAMPLES` is the number of samples to generate
- `SAMPLE_PATH` is the path of the samples images folder
- `STATS_PATH` is the path of the txt file storing the statistics over the generated samples (e.g stats/stats.txt).

## Reproducibility

You can reproduce the results of the report's experiments (section 4) by running 
```bash
run.sh
```
