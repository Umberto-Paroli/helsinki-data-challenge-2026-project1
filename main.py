"""
python main.py [config.yaml]

Everything (data paths, physical constants, model choice/hyperparameters,
training schedule) comes from the YAML file -- nothing is passed on the
command line except the config path itself, so a run is fully reproducible
from the single file saved into its own runs/ folder. Defaults to
./config_adam.yaml if no path is given.

Only the ADAM optimizer path is wired in (LBFGS is not currently used --
see legacy/train_LBFGS_old.py if you need it back).

Two ways to use this file now that there's a convex warm-start (see
main_convex.py / train_convex.py):
  - config_adam.yaml    : fits directly, ellipsoid warm start (unchanged).
  - config_refine.yaml  : model.hypernet: warmstart_points -- continues
    from a finished main_convex.py run's checkpoint.pt (see that file's
    comments for how to point it at one).
"""

import sys

import torch
import yaml

from train import train


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    context = {
        "torch": torch,
        "LMAX": cfg["model"]["LMAX"],
        "log_every": cfg["training"]["log_every"],
    }

    cfg["model"]["dtype"] = eval(cfg["model"]["dtype"], {"__builtins__": {}}, context)
    hypernet_kwargs = cfg["model"].get("hypernet_kwargs")
    if hypernet_kwargs and "latent_dim" in hypernet_kwargs:
        hypernet_kwargs["latent_dim"] = eval(hypernet_kwargs["latent_dim"], {"__builtins__": {}}, context)
    cfg["validation"]["every"] = eval(cfg["validation"]["every"], {"__builtins__": {}}, context)
    return cfg


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config_refine.yaml"
    cfg = load_config(config_path)
    assert cfg.get("optimizer", "ADAM").upper() == "ADAM", (
        f"optimizer={cfg.get('optimizer')!r} -- only ADAM is wired in right now "
        f"(see legacy/train_LBFGS_old.py for the old LBFGS path)"
    )
    train(cfg, config_path)
