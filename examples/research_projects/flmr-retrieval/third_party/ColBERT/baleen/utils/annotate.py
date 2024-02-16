import os
import ujson

from colbert.utils.utils import print_message, file_tqdm


def annotate_to_file(qas_path, ranking_path):
    output_path = f'{ranking_path}.annotated'
    assert not os.path.exists(output_path), output_path

    QID2pids = {}

    with open(qas_path) as f:
        print_message(f"#> Reading QAs from {f.name} ..")

        for line in file_tqdm(f):
            example = ujson.loads(line)
            QID2pids[example['qid']] = example['support_pids']

    with open(ranking_path) as f:
        print_message(f"#> Reading ranked lists from {f.name} ..")

        with open(output_path, 'w') as g:
            for line in file_tqdm(f):
                qid, pid, *other = line.strip().split('\t')
                qid, pid = map(int, [qid, pid])

                label = int(pid in QID2pids[qid])

                line_ = [qid, pid, *other, label]
                line_ = '\t'.join(map(str, line_)) + '\n'
                g.write(line_)

    print_message(g.name)
    print_message("#> Done!")

    return g.name
