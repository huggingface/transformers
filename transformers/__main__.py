# coding: utf8

def main():
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ["convert", "train", "predict", "serve"]:
        print(
        "First argument to `transformers` command line interface should be one of: \n"
        ">> convert serve train predict")
    if sys.argv[1] == "convert":
        from transformers.commands import convert
        convert(sys.argv)
    elif sys.argv[1] == "train":
        from transformers.commands import train
        train(sys.argv)
    elif sys.argv[1] == "serve":
        pass
        # from argparse import ArgumentParser
        # from transformers.commands.serving import ServeCommand
        # parser = ArgumentParser('Transformers CLI tool', usage='transformers serve <command> [<args>]')
        # commands_parser = parser.add_subparsers(help='transformers-cli command helpers')


        # # Register commands
        # ServeCommand.register_subcommand(commands_parser)

        # # Let's go
        # args = parser.parse_args()

        # if not hasattr(args, 'func'):
        #     parser.print_help()
        #     exit(1)
        # # Run
        # service = args.func(args)
        # service.run()

if __name__ == '__main__':
    main()
