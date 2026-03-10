import argparse
import re


def clean_text(text):

    text = text.strip()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    text = text.replace("\t", " ")
    text = re.sub(" +", " ", text)

    return text


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()

    kept = 0

    with open(args.output, "w", encoding="utf-8") as out:

        for line in open(args.input, encoding="utf-8"):

            line = clean_text(line)

            if len(line) < 10:
                continue

            out.write(line + "\n")
            kept += 1

    print("kept", kept)


if __name__ == "__main__":
    main()
