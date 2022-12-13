# Second Split Forgetting

My (possibly incorrect) implementation of [Characterizing Datapoints via Second-Split Forgetting](https://arxiv.org/abs/2210.15031)

Sadly their implementation came out literally an hour after I coded this, but I find this easier to iterate off of :)

Here is the paper citation to give credit where credit is due:
```
@inproceedings{
maini2022characterizing,
title={Characterizing Datapoints via Second-Split Forgetting},
author={Pratyush Maini and Saurabh Garg and Zachary Chase Lipton and J Zico Kolter},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
}
```

*Warning* I copied code from another one of my repos, so there could be some unnecessary stuff in here...

## Getting Started

To test out this second split forgetting metric, simply run the `base.yaml` file of any dataset folder in configs

ex.
```
python main.py --config configs/Imagenette/base.yaml
```

If you want to just do a normal training run on all the data, set `data.train_first_split = 'all'`

It should save the model predictions and labels for each epoch on the first split for stage 1 and 2 of training in `results/{DATASET}/{RUN}/{SPLIT}/predictions-epoch_{EPOCH}.csv`

Right now computing FSLT and SSFT is done in a notebook, will change to be logged in the future (hopefully)

### Mislabeling Data

If you want to run on mislabled data, you can change `noisy.method = random`, or add in your own custom mislabeling method to methods.py and use that method name. `noise.p` is the percentage of the data to mislabel. 

## Remove Samples (beta)

To test out the different metrics for determaning mislabled examples, set `data.remove = true` and define `data.num_samples_to_remove`, `data.removal_method`, and `data.results_dir`. The available removal methods are in [removal_methods.py](removal_methods.py).

Note that you need to train a model on this dataset beforehand (duh).

ex.
```
python main.py --config configs/waterbirds/base.yaml data.remove = true data.results_dir = predictions/Waterbirds95/waterbirds95/first-split data.samples_to_remove = 0.1 data.removal_method = high_loss
```

TODO:
- add in wandb vis of mislabeled, rare, and hard examples [ ]
- add in functionality for removing specified datapoints given the csv's of the predictions [X]
- add in more robustness datasets (Imagenet-A/R/O) [ ]
- add in method to generate more likeley mislables (proportional to commonly confused classes) [ ]
- add in google scraped ImageNet [ ]
- add in other tasks (captioning?) [ ]
