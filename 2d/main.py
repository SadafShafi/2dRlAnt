from __future__ import division

from datetime import datetime
import os
from pathlib import Path

from test import TestConfig, evaluate
from train import TrainConfig, train


def make_run_output_dir(base_dir="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "run_{0}_pid{1}".format(timestamp, os.getpid())
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    run_dir = make_run_output_dir()
    train_config = TrainConfig(output_dir=str(run_dir))
    result = train(train_config)
    evaluate(Path(result["q_table_path"]), TestConfig(output_dir=str(run_dir)))


if __name__ == "__main__":
    main()
