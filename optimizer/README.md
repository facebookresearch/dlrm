# From PipeDream (https://github.com/msr-fiddle/pipedream)
# PipeDream Optimizer

This directory contains an implementations of the PipeDream optimizer used to partition
deep learning models amongst different workers.

`optimizer_graph_hierarchical.py` takes profiles returned by the PipeDream profiler, and determines how to
partition models across the different available workers, while taking into account the fact
that the input machine topologies might be hierarchical.

`python optimizer_graph_hierarchical.py -h` will show you command line options, and should be fairly self-explanatory.

Example of a location of a profile is at ../profiler/image_classification/profiles/resnet101.

This won't work for Inception (or networks with a lot of branching like NASNets).
For Inception, run the `compress_graph_branches.py` script.
It will create a `optimizer/compressed_graphs` directory which will have `graph.txt` files for `inception_v3` and `nasneta*`.
You can then use these files as inputs for the optimizers.

`convert_graph_to_model.py` converts the output of the profiler to a partitioned PyTorch model
that can be used by the PipeDream runtime to perform a combination of model, data, and
input pipelining. `python convert_graph_to_model.py -h` to show all command line options.
