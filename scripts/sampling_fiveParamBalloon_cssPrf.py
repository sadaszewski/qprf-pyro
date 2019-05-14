from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--input-filename', '-i', type=str, required=True)
    parser.add_argument('--lookup-table-filename', '-t', type=str, required=True)
    parser.add_argument('--output-filename', '-o', type=str, required=True)
    parser.add_argument('--variable-name', '-v', type=str,
        default='stimulus')


def main():
    parser = create_parser()
    args = parser.parse_args()


if __name__ == '__main__':
    main()
