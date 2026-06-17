#!/usr/bin/env python3
"""
Training entry point.

Usage:
    python scripts/train.py --config-path ../gaide/config/train --config-name gaide \
        dataset_path=/path/to/dataset

Configs: gaide/config/train/{gaide,gaide_h,gaide_v}.yaml
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../gaide/config/train", config_name="gaide")
def main(cfg: DictConfig):
    import hydra as _hydra
    agent = _hydra.utils.instantiate(cfg, _recursive_=False)
    agent.run()


if __name__ == "__main__":
    main()
