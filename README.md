# GAIDE

**Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning**

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2603.04463-b31b1b.svg)](https://arxiv.org/abs/2603.04463)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official code release for **GAIDE**, a learning-based motion planner for robotic
manipulation. GAIDE uses a kinematic-chain graph to build an adjacency mask over
a transformer encoder, which helps the model respect the robot embodiment while
attending to surrounding scene geometry. During inference, GAIDE evaluates
parallel trajectory rollouts and selects the best collision-free solution.

**Authors:** Davood Soleymanzadeh, Xiao Liang, Minghui Zheng  
**Venue:** IROS 2026  
**Paper:** <https://arxiv.org/abs/2603.04463>

## Highlights

- Embodiment-aware transformer masking from the robot kinematic graph.
- Scene-aware motion planning from point-cloud observations.
- Test-time trajectory optimization with parallel rollout candidates.
- Training and evaluation configs for tabletop, box, bins, and shelf scenes.
- Includes ablations for always-masked and vanilla full-attention transformers.

## Model Variants

| Variant | Config | Description |
| --- | --- | --- |
| GAIDE | `gaide/config/train/gaide.yaml` | Proposed model with alternating graph-masked and unmasked attention layers. |
| GAIDE-H | `gaide/config/train/gaide_h.yaml` | Ablation that applies graph masking at every layer. |
| GAIDE-V | `gaide/config/train/gaide_v.yaml` | Ablation with standard full attention and no graph mask. |

## Repository Layout

```text
GAIDE/
+-- gaide/
|   +-- agent/          # Training and evaluation agents
|   +-- common/         # Checkpointing, timing, and training helpers
|   +-- config/         # Hydra configs for training and evaluation
|   +-- dataset/        # Planning dataset loader
|   +-- eval_task/      # Scene-specific planning/evaluation tasks
|   +-- model/          # GAIDE model variants and shared transformer modules
|   +-- policy/         # Policy wrapper around learned networks
|   +-- utils/          # Point-cloud, PyBullet, URDF, and robot utilities
+-- scripts/
|   +-- train.py        # Training entry point
|   +-- eval.py         # Evaluation entry point
+-- train.sh            # Thin wrapper around scripts/train.py
+-- eval.sh             # Thin wrapper around scripts/eval.py
+-- setup.py
+-- LICENSE
```

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/DavoodSZ/GAIDE.git
cd GAIDE
pip install -e .
```

This repository currently keeps dependency installation lightweight. Install the
runtime dependencies used by the code before training or evaluation:

```bash
pip install hydra-core omegaconf numpy scipy torch tqdm wandb trimesh numba lxml \
    networkx pillow six pybullet open3d
```

Additional external dependencies:

- [cuRobo](https://curobo.org): required for forward kinematics and collision
  checking during evaluation.
- `pointnet2_ops`: required by the point-cloud encoder.
- `ikfast_ur5e`: required by UR5e utility code.

> TODO: replace the manual dependency list with pinned `requirements.txt` or
> `pyproject.toml` dependencies before tagging a public release.

## Data

Training expects a serialized planning dataset passed through the Hydra
`dataset_path` override. The training configs document this as a path to data
containing `planning_data.pth`.

```bash
python scripts/train.py dataset_path=/path/to/planning_data.pth
```

Evaluation expects scene assets and planning problems:

- `problem_dir`: directory containing planning problem definitions
- `ws_dir`: workspace geometry directory
- `env_dir`: environment geometry/config directory
- `normalization_info_path`: normalization metadata saved with the dataset/model
- `model_path`: trained checkpoint

> TODO: publish or document the exact dataset schema, preprocessing command, and
> expected directory tree for training and evaluation assets.

## Training

Train the proposed GAIDE model:

```bash
python scripts/train.py \
    --config-name gaide \
    dataset_path=/path/to/planning_data.pth
```

Train ablation models:

```bash
python scripts/train.py --config-name gaide_h dataset_path=/path/to/planning_data.pth
python scripts/train.py --config-name gaide_v dataset_path=/path/to/planning_data.pth
```

Useful overrides:

```bash
python scripts/train.py \
    --config-name gaide \
    dataset_path=/path/to/planning_data.pth \
    device=cuda:0 \
    seed=42 \
    dataloader.batch_size=256
```

Training outputs are written to:

```text
outputs/train/<model-name>/<date>_<time>_<seed>/
```

## Evaluation

Available scene configs:

- `tabletop`
- `box`
- `bins`
- `shelf_task_i`
- `shelf_task_ii`
- `shelf_task_iii`

Run evaluation for one scene:

```bash
python scripts/eval.py \
    --config-path ../gaide/config/eval/tabletop \
    --config-name run \
    model_path=/path/to/checkpoint.pt \
    normalization_info_path=/path/to/norm.pth \
    problem_dir=/path/to/problems \
    ws_dir=/path/to/workspace \
    env_dir=/path/to/env
```

Change `tabletop` in `--config-path` to evaluate another scene. Results are
written to the configured `save_dir`, for example:

```text
outputs/eval/tabletop/
```

## Configuration

GAIDE uses [Hydra](https://hydra.cc) configs. Primary configs live under:

- `gaide/config/train/` for model training
- `gaide/config/eval/<scene>/` for scene evaluation
- `gaide/config/experiment/` for robot/environment metadata

Most parameters can be overridden from the command line:

```bash
python scripts/train.py train.gradient_steps=100000 optimizer.lr=1e-4
```

## Release Status

This repository is prepared as a research-code release. The core source,
training entry point, evaluation entry point, configs, and license are present.
Some public-release conveniences are still pending; see the checklist below.

## TODO

- [ ] Add pinned environment files (`requirements.txt`, `environment.yml`, or
      `pyproject.toml`) with supported Python, CUDA, PyTorch, cuRobo, and
      `pointnet2_ops` versions.
- [ ] Document dataset generation/preprocessing and the exact
      `planning_data.pth` schema.
- [ ] Add a small sample dataset or smoke-test fixture that can run without
      private assets.
- [ ] Publish pretrained checkpoints and normalization files, or document how to
      reproduce them.
- [ ] Add expected evaluation asset directory trees for `problem_dir`, `ws_dir`,
      and `env_dir`.
- [ ] Add quick-start smoke tests for import, one training step, and one
      evaluation dry run.
- [ ] Add CI for formatting/import checks and lightweight tests.
- [ ] Add `CITATION.cff` for GitHub citation metadata.
- [ ] Add release notes or a `CHANGELOG.md` before the first tagged release.
- [ ] Verify all script examples on a clean machine/container.

## Citation

If you find this work useful, please cite:

```bibtex
@article{soleymanzadeh2026gaide,
  title   = {GAIDE: Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning},
  author  = {Soleymanzadeh, Davood and Liang, Xiao and Zheng, Minghui},
  journal = {arXiv preprint arXiv:2603.04463},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
