#!/usr/bin/env python3
"""
Evaluation entry point.

Usage:
    python scripts/eval.py --config-path ../gaide/config/eval/bins --config-name run \
        model_path=/path/to/checkpoint.pt \
        normalization_info_path=/path/to/norm.pth \
        problem_dir=/path/to/problems \
        ws_dir=/path/to/workspace \
        env_dir=/path/to/env

Scene configs: gaide/config/eval/{tabletop,box,bins,shelf_task_i,shelf_task_ii,shelf_task_iii}/run.yaml
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../gaide/config/eval/tabletop", config_name="run")
def main(cfg: DictConfig):
    import hydra as _hydra
    agent = _hydra.utils.instantiate(cfg, _recursive_=False)
    agent.run()


if __name__ == "__main__":
    main()
