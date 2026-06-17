#!/usr/bin/env python3

from gaide.agent.eval.eval_agent import EvalAgent
from gaide.common.timer import Timer


class EvalGAIDEAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        timer = Timer()

        for planning_difficulty in self.cfg.planning_difficulties:
            for optim_sample_trajectories in self.cfg.optim_sample_trajectories:
                self.planner.plan(
                    planning_difficulty=planning_difficulty,
                    model=self.policy,
                    optim_sample_trajectories=optim_sample_trajectories,
                )

        print(f"Evaluation complete in {timer(reset=False):.1f}s")
