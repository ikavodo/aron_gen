import argparse
from aron_gen.cli import run_generation


def main():
    parser = argparse.ArgumentParser(description='Generate Aronson sequence data files.')
    parser.add_argument('n', type=int, nargs='?', default=3, help='Number of iterations to generate (default: 3)')
    args = parser.parse_args()
    run_generation(args.n)
    print('Done')


if __name__ == '__main__':
    main()
