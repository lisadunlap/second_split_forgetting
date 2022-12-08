# Second Split Forgetting

My (possibly incorrect) implementation of [Characterizing Datapoints via Second-Split Forgetting](https://arxiv.org/abs/2210.15031)

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