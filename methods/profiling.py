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
        raise NotImplementedError
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, metric='val balanced class acc'):
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
    
    def profile_step(self, val_metrics, new_val_metrics, weights, new_weights, metric='val balanced class acc', iteration=0):
        delta = val_metrics[metric] / 100 # kind of arbitrary metric
        if new_val_metrics[metric] > val_metrics[metric] + delta:
            return new_weights # upweighting helped
        elif new_val_metrics[metric] < val_metrics[metric] - delta:
            return self.get_new_weights(weights, iteration, upweight=False) # upweighting hurt
        else:
            return weights