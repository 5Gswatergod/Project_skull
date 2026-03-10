import argparse
import yaml
from skull.train.trainer_cpt import TrainerCPT


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    trainer = TrainerCPT(cfg)

    trainer.train()


if __name__ == "__main__":
    main()
