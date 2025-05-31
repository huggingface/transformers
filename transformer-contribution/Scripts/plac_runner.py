#!D:\contributions\transformers\transformer-contribution\Scripts\python.exe
from __future__ import with_statement
import os
import sys
import shlex
import plac


def run(fnames, cmd, verbose):
    "Run batch scripts and tests"
    for fname in fnames:
        with open(fname) as f:
            lines = list(f)
        if not lines[0].startswith('#!'):
            sys.exit('Missing or incorrect shebang line!')
        firstline = lines[0][2:]  # strip the shebang
        init_args = shlex.split(firstline)
        tool = plac.import_main(*init_args)
        command = getattr(plac.Interpreter(tool), cmd)  # doctest or execute
        if verbose:
            sys.stdout.write('Running %s with %s' % (fname, firstline))
        command(lines[1:], verbose=verbose)


@plac.annotations(
    verbose=('verbose mode', 'flag', 'v'),
    interactive=('run plac tool in interactive mode', 'flag', 'i'),
    multiline=('run plac tool in multiline mode', 'flag', 'm'),
    serve=('run plac server', 'option', 's', int),
    batch=('run plac batch files', 'flag', 'b'),
    test=('run plac test files', 'flag', 't'),
    fname='script to run (.py or .plac or .placet)',
    extra='additional arguments',
    )
def main(verbose, interactive, multiline, serve, batch, test, fname='',
         *extra):
    "Runner for plac tools, plac batch files and plac tests"
    baseparser = plac.parser_from(main)
    if not fname:
        baseparser.print_help()
    elif sys.argv[1] == fname:  # script mode
        plactool = plac.import_main(fname)
        plactool.prog = os.path.basename(sys.argv[0]) + ' ' + fname
        out = plac.call(plactool, sys.argv[2:], eager=False)
        if plac.iterable(out):
            for output in out:
                print(output)
        else:
            print(out)
    elif interactive or multiline or serve:
        plactool = plac.import_main(fname, *extra)
        plactool.prog = ''
        i = plac.Interpreter(plactool)
        if interactive:
            i.interact(verbose=verbose)
        elif multiline:
            i.multiline(verbose=verbose)
        elif serve:
            i.start_server(serve)
    elif batch:
        run((fname,) + extra, 'execute', verbose)
    elif test:
        run((fname,) + extra, 'doctest', verbose)
        print('run %s plac test(s)' % (len(extra) + 1))
    else:
        baseparser.print_usage()


main.add_help = False

if __name__ == '__main__':
    plac.call(main)
