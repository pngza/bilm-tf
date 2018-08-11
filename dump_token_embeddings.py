import argparse
import re
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings


def load_json_file(filename):
    return json.load(open(filename, 'r'))

def filename_variation(filename, replacement):
    return filename.replace('weights', replacement)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', help='', required=True)
    parser.add_argument('--weight', help='', required=True)
    args = parser.parse_args()

    # Dump the token embeddings to a file. Run this once for your dataset.
    options_file = filename_variation(args.weight, 'options').replace('.hdf5', '.json')
    token_embedding_file = filename_variation(args.weight, 'token_embedding')

    print(f'output file: {token_embedding_file}')

    dump_token_embeddings(
        args.vocab, options_file, args.weight, token_embedding_file
    )

if __name__ == "__main__":
    main()
