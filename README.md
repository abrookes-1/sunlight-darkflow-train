## Training Darkflow ##
The way darkflow works, each training run corresponds to one cfg/[cfg-name].cfg file. The base cfg (tiny-yolo-voc.cfg) must not be changed, see darkflow github for explanation of this and instructions on how to create a config file for custom data sets.

Ignored files you must have:
- bin/tiny-yolo-voc.weights (base weights file, is loaded on train and test)
- data/ogars_in_flight/images/* for example
- data/ogars_in_flight/annotations/* for example

Training scripts ([cfg-name] = "darkflow-ogar-load")
- train-ogar-load-init.sh to start training (use for first time training on this config file only. ie. when no checkpoints for that cfg-name exist yet.)
- train-ogar-load.sh to continue that training session
- keyboard interrupt (ctrl-c) to stop trainin at any point
- save-pb-sh to generate built_graph/[cfg-name].pb and built_graph/[cfg-name].meta
