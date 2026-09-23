import sys

from .window import Window


def main():
    Window(sys.argv[1] if len(sys.argv) > 1 else None).run()


if __name__ == '__main__':
    main()
