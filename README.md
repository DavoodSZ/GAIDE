# GAIDE

**[Website](TODO) | [arXiv](https://arxiv.org/abs/2603.04463) | [PDF](https://arxiv.org/pdf/2603.04463)**

**IROS 2026** — Official code release for *GAIDE: Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning*.

GAIDE is a learning-based motion planner for robot arms that encodes both the robot's kinematic structure and the scene geometry using graph-based attention masking. A kinematic-chain adjacency matrix determines which joints can attend to each other, enabling the transformer encoder to respect the robot's embodiment while reasoning about surrounding obstacles.

Three variants are provided:

| Variant | Description |
|---|---|
| **GAIDE** | Proposed method — alternates between masked and unmasked attention layers |
| **GAIDE-H** | Ablation — graph adjacency mask applied at every layer |
| **GAIDE-V** | Ablation — standard full-attention transformer (no masking) |

---

## Installation

```bash
git clone https://github.com/DavoodSZ/GAIDE.git
cd GAIDE
pip install -e .
```

Dependencies: `torch`, `hydra-core`, `wandb`, `curobo`, `open3d`, `trimesh`, `numba`, `scipy`.

---

## Training

```bash
# GAIDE (proposed)
python scripts/train.py --config-name gaide dataset_path=/path/to/planning_data.pth

# GAIDE-H (ablation)
python scripts/train.py --config-name gaide_h dataset_path=/path/to/planning_data.pth

# GAIDE-V (ablation)
python scripts/train.py --config-name gaide_v dataset_path=/path/to/planning_data.pth
```

Checkpoints and logs are saved under `outputs/train/`.

---

## Evaluation

```bash
python scripts/eval.py \
    --config-path ../gaide/config/eval/<scene> \
    --config-name run \
    model_path=/path/to/checkpoint.pt \
    normalization_info_path=/path/to/norm.pth \
    problem_dir=/path/to/problems \
    ws_dir=/path/to/workspace \
    env_dir=/path/to/env
```

Replace `<scene>` with one of:

| Scene | Config path |
|---|---|
| TabletTop | `gaide/config/eval/tabletop` |
| Box | `gaide/config/eval/box` |
| Bins | `gaide/config/eval/bins` |
| Shelf Task I | `gaide/config/eval/shelf_task_i` |
| Shelf Task II | `gaide/config/eval/shelf_task_ii` |
| Shelf Task III | `gaide/config/eval/shelf_task_iii` |

Results are written to `outputs/eval/<scene>/`.

---

## Citation

```bibtex
@inproceedings{soleymanzadeh2026gaide,
  title     = {GAIDE: Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning},
  author    = {Soleymanzadeh, Davood and Liang, Xiao and Zheng, Minghui},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
