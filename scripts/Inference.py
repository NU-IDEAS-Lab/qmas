from onpolicy.scripts.render.render_pettingzoo import main
import os
import yaml
os.environ["WANDB__SERVICE_WAIT"] = "300"


model_dir = "/home/ppe7517/qmas_task_allocation/patrolling_zoo/results/hanabi_v5_minimal_obs/mappo/hanabi-comms-e200-new/wandb/run-20260106_171827-oxvc9huq/files"

# Load arguments from the config file.
config_file = os.path.join(model_dir, "config.yaml")
args = yaml.load(open(config_file), Loader=yaml.FullLoader)

def wrap_arg_val(val):
    return {"value": val}

# Set required render-specific arguments. Do not change these!
args["use_wandb"] = wrap_arg_val(False)
args["use_render"] = wrap_arg_val(True)
args["model_dir"] = wrap_arg_val(model_dir)
args["cuda"] = wrap_arg_val(True)
args["cuda_idx"] = wrap_arg_val(0)
args["cuda_idx_predictor"] = wrap_arg_val(0)

# Feel free to change these arguments or add additional arguments here.
args["render_episodes"] = wrap_arg_val(100)
args["episode_length"] = wrap_arg_val(100)
args["max_cycles"] = args["episode_length"]
# args["seed"] = wrap_arg_val(1492)

# args["algorithm_class"] = wrap_arg_val("qmas.variant_three_modules.algorithm.QmasAlgorithm")
# args["policy_class"] = wrap_arg_val("qmas.variant_three_modules.policy.QmasPolicy")

# args["observation_mask"] = wrap_arg_val(False)
# args["observation_radius"] = wrap_arg_val(2)
# args["observation_probability"] = wrap_arg_val(0.1)
args["noisy_memory"] = wrap_arg_val(False)

args["diffusion_autoregression_steps"] = wrap_arg_val(0)
# args["prediction_history_window"] 

# args["prediction_disable"] = wrap_arg_val(True)

# Convert arguments to the appropriate format.
def add_arg(arglist, key, value):
    if value == None:
        return
    if type(value) == bool:
        if value:
            arglist.append(f"--{key}")
        else:
            arglist.append(f"--no-{key}")
    elif type(value) == list:
        arglist.append(f"--{key}")
        arglist.extend([str(v) for v in value])
    else:
        arglist.append(f"--{key}")
        arglist.append(str(value))
arglist = []
for key, value in args.items():
    if type(value) == dict and "value" in value and not key.startswith("_"):
        add_arg(arglist, key, value["value"])
    else:
        add_arg(arglist, key, value)

main(arglist)