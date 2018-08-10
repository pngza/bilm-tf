import argparse
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings


def load_json_file(filename):
    return json.load(open(filename, 'r'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', help='', required=True)
    parser.add_argument('--options', help='', required=True)
    parser.add_argument('--weight', help='', required=True)
    args = parser.parse_args()

    # Dump the token embeddings to a file. Run this once for your dataset.
    token_embedding_file = 'elmo_token_embeddings.hdf5'
    dump_token_embeddings(
        args.vocab, args.options, args.weight, token_embedding_file
    )

if __name__ == "__main__":
    main()
