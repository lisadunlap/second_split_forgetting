import metrics

class Selector:
    """
    Wrapper class for selecting "intersting" samples
    """
    def __init__(self, cfg, trainloader, valloader):
        self.cfg = cfg
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_interesting_samples(self, results):
        """
        Returns the interesting samples based on the results.
        """
        return []

class JustTrainTwice(Selector):
    """
    Butchering of Just Train Twice Paper: after warmup period, picks the samples
    that are still incorrect and upweights them.
    """
    def get_interesting_samples(self, results):
        last_epoch = results['epoch'].max()
        results['correct'] = results['prediction'] == results['label']
        return results[(results['epoch'] == last_epoch) & (results['correct'] == False)]['image_id'].tolist()

class MetricSelect(Selector):
    """
    Get outliers (+- 2 std) belove the mean for a given metric.
    """
    def get_interesting_samples(self, results):
        assert self.cfg.hps.profile_freq == 0, 'Cannot use MetricSelect with profile_freq > 0'
        metric = self.cfg.select.metric
        df = results.groupby('image_id', as_index=False).apply(getattr(metrics, metric))
        df.columns = ['image_id', metric]
        mean = df[metric].mean()
        std = df[metric].std()
        if self.cfg.select.mode == 'min':
            return df[(df[metric] < mean - 2*std)]['image_id'].tolist()
        return df[(df[metric] > mean + 2*std)]['image_id'].tolist()

class RandomSelect(Selector):
    """
    Randomly select a given number of samples.
    """
    def get_interesting_samples(self, results):
        if self.cfg.select.num_samples < 1:
            num_samples = int(self.cfg.select.num_samples * len(results['image_id'].unique()))
        else:
            num_samples = self.cfg.select.num_samples 
        print(f"NUM SAMPLES {num_samples}")
        return results[results['epoch'] == results['epoch'].max()]['image_id'].sample(num_samples).tolist()

class Profiler:
    """
    Wrapper class for profiling different weights.
    """
    def __init__(self, cfg, trainloader, valloader):
        self.cfg = cfg
        self.trainloader = trainloader
        self.valloader = valloader

    def get_new_weights(self, weights, samples):
        """
        Returns new weights based on the current weights.
        """
        return weights
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, iteration=0):
        """
        Runs a single profiling step. Given val metrics of old and new weights, determine what the 
        new weights should be based on provided metric.
        """
        return weights, new_val_metrics[self.cfg.profile.metric] - val_metrics[self.cfg.profile.metric], False

class Upweight(Profiler):
    """
    Upweight the samples.
    """
    def get_new_weights(self, weights, samples):
        """
        Returns new weights based on the current weights.
        """
        new_weights = weights.copy()
        new_weights[samples] *= self.cfg.data.upweight_factor
        return new_weights
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, iteration=0):
        return new_weights, new_val_metrics[self.cfg.profile.metric] - val_metrics[self.cfg.profile.metric], True

class Downweight(Profiler):
    """
    Downweight the samples.
    """
    def get_new_weights(self, weights, samples):
        """
        Returns new weights based on the current weights.
        """
        new_weights = weights.copy()
        new_weights[samples] /= self.cfg.data.upweight_factor
        return new_weights
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, iteration=0):
        return new_weights, new_val_metrics[self.cfg.profile.metric] - val_metrics[self.cfg.profile.metric], True

class LeaveOneOut(Profiler):
    """
    Essentially leave one out; only upweight one sample at a time.
    If val acc goes up by some delta, keep the new weights. If val acc goes
    down by some delta, divide weight by upweight factor. Otherwise, keep old weights.
    """
    def get_new_weights(self, weights, samples, upweight=True):
        """
        Returns new weights based on the current weights.
        """
        new_weights = weights.copy()
        if upweight:
            new_weights[samples] *= self.cfg.data.upweight_factor
        else:
            new_weights[samples] /= self.cfg.data.upweight_factor
        return new_weights
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, iteration=0):
        # delta = val_metrics[self.cfg.profile.metric] / 100 # kind of arbitrary metric
        delta = 50 / val_metrics[self.cfg.profile.metric] # kind of arbitrary metric
        diff = new_val_metrics[self.cfg.profile.metric] - val_metrics[self.cfg.profile.metric]
        print(f"Delta {round(delta, 3)} Val {self.cfg.profile.metric} went from {round(val_metrics[self.cfg.profile.metric], 3)} to {round(new_val_metrics[self.cfg.profile.metric], 3)} ({diff})")
        if diff > delta:
            return new_weights, diff, True # upweighting helped
        # elif diff < - delta:
        #     return self.get_new_weights(weights, iteration, upweight=False), diff, False # upweighting hurt
        else:
            return weights, diff, False
