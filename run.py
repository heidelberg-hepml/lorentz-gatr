from absl import app
from absl import flags
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from gatr.utils.misc import load_config
from experiments.mass_regression.mass_regression import MassRegressionExperiment

def define_flags():
    flags.DEFINE_string('warm_start_path', None, "Path to the pre-trained model folder")
    flags.DEFINE_string('overwrite_config_path', None, "Path to another config file to overwrite some config")
    
    flags.DEFINE_boolean('train', None, "Overwrite train parameter in config")
    flags.DEFINE_boolean('compute_logL', None, "Overwrite compute_logL parameter in config")
    flags.DEFINE_boolean('sample', None, "Overwrite sample parameter in config")
    flags.DEFINE_boolean("reweight", None, "Overwrite reweight in config")
    flags.DEFINE_boolean('load_samples', None, "Overwrite load_samples parameter in config")
    flags.DEFINE_boolean('plot', None, "Overwrite plot parameter in config")
    
    flags.DEFINE_integer('n_samples', None, "Overwrite n_samples in config")
    flags.DEFINE_integer('n_epochs', None, "Overwrite n_epochs in config")
    flags.DEFINE_integer('iterations', None, "Overwrite iterations in config")
    
    flags.DEFINE_boolean('save_samples', None, "Overwrite save_samples parameter in config")
    flags.DEFINE_boolean("redirect_console", None, "Overwrite redirect_console in config")
    
    flags.DEFINE_list("channels", None, "Overwrite channels in config")
    flags.DEFINE_boolean("use_df", None, "Overwrite use_df in config")

def main(argv):
    if FLAGS.warm_start_path is None:
        config = load_config(sys.argv[1])
        if config.get("warm_start", False):
            print("Warning: Set warm_start=True in the config, but this does not make sense (see docu in run.py). "
                  "Manually setting warm_start=False.")
            config["warm_start"] = False
    else:
        config = load_config(os.path.join(FLAGS.warm_start_path, "config.yaml"))
        config["warm_start"] = True
        config["warm_start_path"] = FLAGS.warm_start_path

    if FLAGS.overwrite_config_path is not None:
        overwrite_config = load_config(FLAGS.overwrite_config_path)
        config = config | overwrite_config

    if FLAGS.train is not None:
        config["train"] = FLAGS.train
    if FLAGS.compute_logL is not None:
        config["compute_logL"] = FLAGS.compute_logL
    if FLAGS.sample is not None:
        config["sample"] = FLAGS.sample
    if FLAGS.reweight is not None:
        config["reweight"] = FLAGS.reweight
    if FLAGS.load_samples is not None:
        config["load_samples"] = FLAGS.load_samples
    if FLAGS.plot is not None:
        config["plot"] = FLAGS.plot
        
    if FLAGS.n_samples is not None:
        config["n_samples"] = FLAGS.n_samples
    if FLAGS.n_epochs is not None:
        config["n_epochs"] = FLAGS.n_epochs
    if FLAGS.iterations is not None:
        config["iterations"] = FLAGS.iterations

    if FLAGS.redirect_console is not None:
        config["redirect_console"] = FLAGS.redirect_console
    if FLAGS.save_samples is not None:
        config['save_samples'] = FLAGS.save_samples

    if FLAGS.channels is not None:
        config["channels"] = [int(i) for i in FLAGS.channels]
    if FLAGS.use_df is not None:
        config["use_df"] = FLAGS.use_df

    # Instantiate the experiment class
    experiment_type = config.get("experiment_type", None)
    if experiment_type == "massregression":
        experiment = MassRegressionExperiment(config)
    else:
        raise ValueError(f"main: experiment_type {experiment_type} not implemented.")

    # Run the experiment
    experiment.full_run()

if __name__ == '__main__':
    # Read in the flags
    FLAGS = flags.FLAGS
    define_flags()
    # Run the main program
    app.run(main)
