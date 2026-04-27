import yaml
import os

def convert_args(argsdict):
    # Convert arguments to the appropriate format.
    arglist = []
    def add_arg(arglist, key, value):
        if value == None:
            return
        if type(value) == bool:
            if value:
                arglist.append(f"--{key}")
            else:
                arglist.append(f"--no-{key}")
        elif type(value) == list and len(value) > 0:
            arglist.append(f"--{key}")
            arglist.extend([str(v) for v in value])
        else:
            arglist.append(f"--{key}")
            arglist.append(str(value))
    for key, value in argsdict.items():
        if type(value) == dict and "value" in value and not key.startswith("_"):
            add_arg(arglist, key, value["value"])
        else:
            add_arg(arglist, key, value)
    print("Converted arguments:", arglist)
    return arglist


def wrap_arg_val(val):
    return {"value": val}


def load_args(model_dir, default_args):
    # Load arguments from the config file.
    config_file = os.path.join(model_dir, "config.yaml")
    args = yaml.load(open(config_file), Loader=yaml.FullLoader)

    # Set required eval-specific arguments. Do not change these!
    args["use_wandb"] = wrap_arg_val(False)
    args["use_render"] = wrap_arg_val(False)
    args["use_eval"] = wrap_arg_val(True)
    args["model_dir"] = wrap_arg_val(model_dir)

    # Update with any default arguments provided.
    for key, value in default_args.items():
        args[key] = value

    return args