# GAIDE: Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning

**Authors**: Davood Soleymanzadeh, Xiao Liang, Minghui Zheng  
**Paper**: [arXiv:2603.04463](https://arxiv.org/abs/2603.04463)  
**Venue**: IROS 2026

This is the official code release for GAIDE, a learning-based motion planner for robotic manipulation. GAIDE uses a kinematic-chain graph to build an adjacency mask over a transformer encoder, allowing the model to respect the robot's embodiment structure while attending to surrounding scene geometry. At inference, test-time optimization runs multiple parallel trajectory rollouts and selects the best collision-free solution.

We provide three variants:
- **GAIDE** (proposed): alternates between graph-masked and unmasked attention layers
- **GAIDE-H**: applies the graph mask at every layer (ablation)
- **GAIDE-V**: standard full-attention transformer with no masking (ablation)

## Installation

```bash
git clone https://github.com/DavoodSZ/GAIDE.git
cd GAIDE
pip install -e .
```

Install [cuRobo](https://curobo.org) separately following the official instructions — it is used for forward kinematics and collision checking during evaluation.

## Training

```bash
# Train GAIDE (proposed method)
python scripts/train.py --config-name gaide dataset_path=/path/to/planning_data.pth

# Train GAIDE-H
python scripts/train.py --config-name gaide_h dataset_path=/path/to/planning_data.pth

# Train GAIDE-V
python scripts/train.py --config-name gaide_v dataset_path=/path/to/planning_data.pth
```

Checkpoints are saved under `outputs/train/`. Training config files are at `gaide/config/train/`.

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

Available scenes: `tabletop`, `box`, `bins`, `shelf_task_i`, `shelf_task_ii`, `shelf_task_iii`.  
Results are saved under `outputs/eval/<scene>/`.

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
