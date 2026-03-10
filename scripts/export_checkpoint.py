import argparse
import torch
from transformers import GPT2Config, GPT2LMHeadModel


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt")
    parser.add_argument("--out")

    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")

    config = GPT2Config()

    model = GPT2LMHeadModel(config)

    model.load_state_dict(state["model"])

    model.save_pretrained(args.out)

    print("exported to", args.out)


if __name__ == "__main__":
    main()
