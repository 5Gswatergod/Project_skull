import argparse
import yaml
from skull.train.trainer_pretrain import Trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    trainer = Trainer(cfg)

    trainer.train()


if __name__ == "__main__":
    main()
