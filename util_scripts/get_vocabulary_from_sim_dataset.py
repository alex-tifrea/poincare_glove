import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        help='File with the similarity pairs from which to extract the vocabulary')
    parser.add_argument('--vocab_file', type=str,
                        help='Filename in which to save the vocabulary.')
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        vocab = {}
        for line in f:
            if "#" in line:
                continue

            words = line.strip().split('\t')[:2]
            print(words)

