from __future__ import annotations

import argparse
import itertools
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", default="python -m src.train.train --config configs/train/aat_pgd.yaml")
    args = parser.parse_args()

    ablations = [
        ("full", {"AAT_ENABLED": "1", "USE_VIT": "1", "BLINK_REG": "1"}),
        ("no_aat", {"AAT_ENABLED": "0", "USE_VIT": "1", "BLINK_REG": "1"}),
        ("no_vit", {"AAT_ENABLED": "1", "USE_VIT": "0", "BLINK_REG": "1"}),
        ("no_blink_reg", {"AAT_ENABLED": "1", "USE_VIT": "1", "BLINK_REG": "0"}),
    ]

    for name, env_overrides in ablations:
        cmd = args.train_script
        env = {**os.environ, **env_overrides}
        print(f"[ablation] {name}: {cmd} with {env}")
        subprocess.run(cmd, shell=True, check=False, env=env)

    # Attack sweep skeleton
    for attack, eps in itertools.product(["fgsm", "pgd"], [0.01, 0.02, 0.03]):
        print(f"[sweep] attack={attack}, eps={eps}")


if __name__ == "__main__":
    main()
