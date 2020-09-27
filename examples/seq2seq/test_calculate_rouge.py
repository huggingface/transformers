from collections import defaultdict
from pathlib import Path

import pandas as pd

from rouge_cli import calculate_rouge_path
from utils import calculate_rouge


PRED = [
    'Prosecutor: "No videos were used in the crash investigation" German papers say they saw a cell phone video of the final seconds on board Flight 9525. The Germanwings co-pilot says he had a "previous episode of severe depression" German airline confirms it knew of Andreas Lubitz\'s depression years before he took control.',
    "The Palestinian Authority officially becomes the 123rd member of the International Criminal Court. The formal accession was marked with a ceremony at The Hague, in the Netherlands. The Palestinians signed the ICC's founding Rome Statute in January. Israel and the United States opposed the Palestinians' efforts to join the body.",
    "Amnesty International releases its annual report on the death penalty. The report catalogs the use of state-sanctioned killing as a punitive measure across the globe. At least 607 people were executed around the world in 2014, compared to 778 in 2013. The U.S. remains one of the worst offenders for imposing capital punishment.",
]

TGT = [
    'Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports . Journalists at Bild and Paris Match are "very confident" the video clip is real, an editor says . Andreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says .',
    "Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June . Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis .",
    "Amnesty's annual death penalty report catalogs encouraging signs, but setbacks in numbers of those sentenced to death . Organization claims that governments around the world are using the threat of terrorism to advance executions . The number of executions worldwide has gone down by almost 22% compared with 2013, but death sentences up by 28% .",
]


def test_disaggregated_scores_are_determinstic():
    no_aggregation = calculate_rouge(PRED, TGT, bootstrap_aggregation=False, rouge_keys=["rouge2", "rougeL"])
    assert isinstance(no_aggregation, defaultdict)
    no_aggregation_just_r2 = calculate_rouge(PRED, TGT, bootstrap_aggregation=False, rouge_keys=["rouge2"])
    assert (
        pd.DataFrame(no_aggregation["rouge2"]).fmeasure.mean()
        == pd.DataFrame(no_aggregation_just_r2["rouge2"]).fmeasure.mean()
    )


def test_newline_cnn_improvement():
    k = "rougeLsum"
    score = calculate_rouge(PRED, TGT, newline_sep=True, rouge_keys=[k])[k]
    score_no_sep = calculate_rouge(PRED, TGT, newline_sep=False, rouge_keys=[k])[k]
    assert score > score_no_sep


def test_newline_irrelevant_for_other_metrics():
    k = ["rouge1", "rouge2", "rougeL"]
    score_sep = calculate_rouge(PRED, TGT, newline_sep=True, rouge_keys=k)
    score_no_sep = calculate_rouge(PRED, TGT, newline_sep=False, rouge_keys=k)
    assert score_sep == score_no_sep


def test_single_sent_scores_dont_depend_on_newline_sep():
    pred = [
        "Her older sister, Margot Frank, died in 1945, a month earlier than previously thought.",
        'Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports .',
    ]
    tgt = [
        "Margot Frank, died in 1945, a month earlier than previously thought.",
        'Prosecutor: "No videos were used in the crash investigation" German papers say they saw a cell phone video of the final seconds on board Flight 9525.',
    ]
    assert calculate_rouge(pred, tgt, newline_sep=True) == calculate_rouge(pred, tgt, newline_sep=False)


def test_pegasus_newline():

    pred = [
        """" "a person who has such a video needs to immediately give it to the investigators," prosecutor says .<n> "it is a very disturbing scene," editor-in-chief of bild online tells "erin burnett: outfront" """
    ]
    tgt = [
        """ Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports . Journalists at Bild and Paris Match are "very confident" the video clip is real, an editor says . Andreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says ."""
    ]

    prev_score = calculate_rouge(pred, tgt, rouge_keys=["rougeLsum"], newline_sep=False)["rougeLsum"]
    new_score = calculate_rouge(pred, tgt, rouge_keys=["rougeLsum"])["rougeLsum"]
    assert new_score > prev_score


def test_rouge_cli():
    data_dir = Path("examples/seq2seq/test_data/wmt_en_ro")
    metrics = calculate_rouge_path(data_dir.joinpath("test.source"), data_dir.joinpath("test.target"))
    assert isinstance(metrics, dict)
    metrics_default_dict = calculate_rouge_path(
        data_dir.joinpath("test.source"), data_dir.joinpath("test.target"), bootstrap_aggregation=False
    )
    assert isinstance(metrics_default_dict, defaultdict)
