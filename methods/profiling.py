class Profiler:
    """
    Wrapper class for profiling different weights.
    """
    def __init__(self, cfg, trainloader, valloader):
        self.cfg = cfg
        self.trainloader = trainloader
        self.valloader = valloader

    def get_new_weights(self, weights, iteration=0):
        """
        Returns new weights based on the current weights.
        """
        return weights
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, iteration=0):
        """
        Runs a single profiling step. Given val metrics of old and new weights, determine what the 
        new weights should be based on provided metric.
        """
        raise NotImplementedError

class LeaveOneOut(Profiler):
    """
    Essentially leave one out; only upweight one sample at a time.
    If val acc goes up by some delta, keep the new weights. If val acc goes
    down by some delta, divide weight by upweight factor. Otherwise, keep old weights.
    """
    def get_new_weights(self, weights, iteration=0, upweight=True):
        """
        Returns new weights based on the current weights.
        """
        new_weights = weights.copy()
        if upweight:
            new_weights[weights == iteration] *= self.cfg.data.upweight_factor
        else:
            new_weights[weights == iteration] /= self.cfg.data.upweight_factor
        return new_weights
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, iteration=0):
        # delta = val_metrics[self.cfg.profile.metric] / 100 # kind of arbitrary metric
        delta = 50 / val_metrics[self.cfg.profile.metric] # kind of arbitrary metric
        diff = new_val_metrics[self.cfg.profile.metric] - val_metrics[self.cfg.profile.metric]
        print(f"Delta {round(delta, 3)} Val {self.cfg.profile.metric} went from {round(val_metrics[self.cfg.profile.metric], 3)} to {round(new_val_metrics[self.cfg.profile.metric], 3)} ({diff})")
        if diff > delta:
            return new_weights, diff, True # upweighting helped
        elif diff < - delta:
            return self.get_new_weights(weights, iteration, upweight=False), diff, False # upweighting hurt
        else:
            return weights, diff, False

class JustTrainTwice(Profiler):
    """
    Butchering of Just Train Twice Paper: after warmup period, picks the samples
    that are still incorrect and upweights them.
    """
    def get_new_weights(self, weights, iteration=0):
        return super().get_new_weights(weights, iteration)


class HalfandHalf(Profiler):
    """
    train with some weights for the first half of training, then train
    with other weights for the second half of training.
    """
    def get_new_weights(self, weights, iteration=0):
        return super().get_new_weights(weights, iteration)