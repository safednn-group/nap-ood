# Configuration of NeuronActivationPatterns method

There is a couple of configurable parameters of the method. They are read from this [file](methods/nap/cfg/strategies.json).

### store_monitor
Boolean flag; set to true if you want to pickle and store a monitor containing binary activation patterns after the train_H phase.
Storing a monitor requires a lot of disk space, but it can save some time during further experiments. Default: false.

### use_tree
Boolean flag; set to true if you want to use our Numba implementation of a ball tree. The implementation is ~2x faster than sklearn BallTree, but still inferior to PyTorch Cuda computations. Default: false.

### accuracy_criterion
Boolean flag; set to true if you want to take validation accuracy as the criterion of NAP extraction parameters optimization.
If set to false distance threshold magnitude will be used as the criterion. Selected results of both scenarios are included in this [table](../README.md#NAP results table).
Default: true.

### steps
Integer; the number of considered activation pattern extraction parameter values during grid search optimization. The higher the more time is needed to train the method. We suggest to use a value higher than 2. Default: 5.

### binary_voting
Boolean flag; parameter defining how to the OOD detector decides if considered sample is an outlier. If set to true layers make binary votes, else every layer returns a numeric uncertainty estimation. Selected results of both scenarios are included in this [table](../README.md#NAP results table).
Default: true. 

### n_votes
Integer; number of monitored (voting) layers. See: [table](../README.md#NAP results table).


