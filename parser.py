import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', metavar='PATH', default='./experiments/',
                        help='directory to save trained models')
    parser.add_argument('--epochs', metavar='N', type=int, default=100,
                        help='training epochs')
    parser.add_argument('--cath-data', metavar='PATH', default='./dataset/chain_set.jsonl',
                        help='location of CATH dataset')
    parser.add_argument('--cath-splits', metavar='PATH', default='./dataset/chain_set_splits.json',
                        help='location of CATH split file')
    parser.add_argument('--train', default=True, help="train a model")
    parser.add_argument('--gpu', type=int, default=7, 
                        help='use which gpu, default is cuda:0')
    parser.add_argument('--model-type', default='gca', choices=['structTrans', 'structGNN', 'gca'])
    parser.add_argument('--node-dim', type=tuple, default=(100, 16))
    parser.add_argument('--edge-dim', type=tuple, default=(32, 1))
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature-type', type=str, default='full')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-tokens', type=int, default=2500)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--is-attention', type=int, default=1)
    parser.add_argument('--augment', type=int, default=0)
    return parser