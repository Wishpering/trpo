from argparse import ArgumentParser, Namespace
from sys import stderr


class ModifiedArgumentParser(ArgumentParser):
    def exit_with_error(self, error_text):
        print(error_text, file=stderr)
        exit(1)

    def print_help_and_exit(self):
        self.print_help()
        exit(0)
