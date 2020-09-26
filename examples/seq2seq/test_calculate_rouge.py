

PRED = ['Prosecutor: "No videos were used in the crash investigation" German papers say they saw a cell phone video of the final seconds on board Flight 9525. The Germanwings co-pilot says he had a "previous episode of severe depression" German airline confirms it knew of Andreas Lubitz\'s depression years before he took control.',
        "The Palestinian Authority officially becomes the 123rd member of the International Criminal Court. The formal accession was marked with a ceremony at The Hague, in the Netherlands. The Palestinians signed the ICC's founding Rome Statute in January. Israel and the United States opposed the Palestinians' efforts to join the body.",
        'Amnesty International releases its annual report on the death penalty. The report catalogs the use of state-sanctioned killing as a punitive measure across the globe. At least 607 people were executed around the world in 2014, compared to 778 in 2013. The U.S. remains one of the worst offenders for imposing capital punishment.',
        'Amnesty International releases its annual review of the death penalty worldwide. The number of executions around the world dropped significantly in 2014 compared with the previous year. The rise in death sentences recorded in 2014 -- up more than 500 -- can also be attributed to governments using death penalty as political tool.',
        'Researchers re-examined archives of the Red Cross, the International Training Service and the Bergen-Belsen concentration camp. Anne Frank died of typhus at the age of 15. Her older sister, Margot Frank, died in 1945, a month earlier than previously thought.']

TGT = ['Prosecutor: "No videos were used in the crash investigation" German papers say they saw a cell phone video of the final seconds on board Flight 9525. The Germanwings co-pilot says he had a "previous episode of severe depression" German airline confirms it knew of Andreas Lubitz\'s depression years before he took control.',
       "The Palestinian Authority officially becomes the 123rd member of the International Criminal Court. The formal accession was marked with a ceremony at The Hague, in the Netherlands. The Palestinians signed the ICC's founding Rome Statute in January. Israel and the United States opposed the Palestinians' efforts to join the body.",
       'Amnesty International releases its annual report on the death penalty. The report catalogs the use of state-sanctioned killing as a punitive measure across the globe. At least 607 people were executed around the world in 2014, compared to 778 in 2013. The U.S. remains one of the worst offenders for imposing capital punishment.',
       'Amnesty International releases its annual review of the death penalty worldwide. The number of executions around the world dropped significantly in 2014 compared with the previous year. The rise in death sentences recorded in 2014 -- up more than 500 -- can also be attributed to governments using death penalty as political tool.',
       'Researchers re-examined archives of the Red Cross, the International Training Service and the Bergen-Belsen concentration camp. Anne Frank died of typhus at the age of 15. Her older sister, Margot Frank, died in 1945, a month earlier than previously thought.']
#metrics = calculate_rouge_score(PRED, TGT)
import numpy as np
import pandas as pd
from utils import calculate_rouge_score

def test_basic_kwargs():
    metrics = calculate_rouge_score(PRED, TGT)
    assert isinstance(metrics, dict)
    assert np.round(metrics['rouge2'], 1) == 13.1

