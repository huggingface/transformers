# coding=utf-8
# Copyright 2020 Huggingface
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from sacrebleu import corpus_bleu

from parameterized import parameterized
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers.testing_utils import require_torch, slow, torch_device


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


# BLEU eval data was generated using the following code:
#
# #!/bin/bash
#
# export OBJS=8
#
# pairs=(ru-en en-ru en-de de-en)
# printf "data = {\n"
# for pair in "${pairs[@]}"
# do
#     export PAIR=$pair
#     printf "    \"$PAIR\": {\n"
#     printf "        \"src\": [\n"
#     sacrebleu -t wmt19 -l $PAIR --echo src | head -$OBJS | perl -ne 'chomp; s#(^"|"$)#\\"#; print qq[    """$_""",\n]'
#     printf "        ],\n"
#     printf "        \"tgt\": [\n"
#     sacrebleu -t wmt19 -l $PAIR --echo ref | head -$OBJS | perl -ne 'chomp; s#(^"|"$)#\\"#; print qq[    """$_""",\n]'
#     printf "        ],\n"
#     printf "    },\n"
# done
# printf "}\n"

bleu_data = {
    "ru-en": {
        "src": [
            """Названо число готовящихся к отправке в Донбасс новобранцев из Украины""",
            """Официальный представитель Народной милиции самопровозглашенной Луганской Народной Республики (ЛНР) Андрей Марочко заявил, что зимой 2018-2019 года Украина направит в Донбасс не менее 3 тыс. новобранцев.""",
            """По его словам, таким образом Киев планирует "хоть как-то доукомплектовать подразделения".""",
            """\"Нежелание граждан Украины проходить службу в рядах ВС Украины, массовые увольнения привели к низкой укомплектованности подразделений", - рассказал Марочко, которого цитирует "РИА Новости".""",
            """Он также не исключил, что реальные цифры призванных в армию украинцев могут быть увеличены в случае необходимости.""",
            """В 2014-2017 годах Киев начал так называемую антитеррористическую операцию (АТО), которую позже сменили на операцию объединенных сил (ООС).""",
            """Предполагалось, что эта мера приведет к усилению роли украинских силовиков в урегулировании ситуации.""",
            """В конце августа 2018 года ситуация в Донбассе обострилась из-за убийства главы ДНР Александра Захарченко.""",
        ],
        "tgt": [
            """The number of new Ukrainian recruits ready to go to Donbass has become public""",
            """Official representative of the peoples’ militia of the self-proclaimed Lugansk People’s Republic Andrey Marochko claimed that Ukrainian will send at least 3 thousand new recruits to Donbass in winter 2018-2019.""",
            """This is how Kyiv tries “at least somehow to staff the units,” he said.""",
            """“The unwillingness of Ukrainian citizens to serve in the Ukraine’s military forces, mass resignments lead to low understaffing,” said Marochko cited by RIA Novosti.""",
            """Also, he doesn’t exclude that the real numbers of conscripts in the Ukrainian army can be raised is necessary.""",
            """In 2014-2017, Kyiv started so-called antiterrorist operation, that ws later changed to the united forces operation.""",
            """This measure was supposed to strengthen the role of the Ukrainian military in settling the situation.""",
            """In the late August 2018, the situation in Donbass escalated as the DNR head Aleksandr Zakharchenko was killed.""",
        ],
    },
    "en-ru": {
        "src": [
            """Welsh AMs worried about 'looking like muppets'""",
            """There is consternation among some AMs at a suggestion their title should change to MWPs (Member of the Welsh Parliament).""",
            """It has arisen because of plans to change the name of the assembly to the Welsh Parliament.""",
            """AMs across the political spectrum are worried it could invite ridicule.""",
            """One Labour AM said his group was concerned "it rhymes with Twp and Pwp.\"""",
            """For readers outside of Wales: In Welsh twp means daft and pwp means poo.""",
            """A Plaid AM said the group as a whole was "not happy" and has suggested alternatives.""",
            """A Welsh Conservative said his group was "open minded" about the name change, but noted it was a short verbal hop from MWP to Muppet.""",
        ],
        "tgt": [
            """Члены Национальной ассамблеи Уэльса обеспокоены, что "выглядят как куклы\"""",
            """Некоторые члены Национальной ассамблеи Уэльса в ужасе от предложения о том, что их наименование должно измениться на MPW (члены Парламента Уэльса).""",
            """Этот вопрос был поднят в связи с планами по переименованию ассамблеи в Парламент Уэльса.""",
            """Члены Национальной ассамблеи Уэльса всего политического спектра обеспокоены, что это может породить насмешки.""",
            """Один из лейбористских членов Национальной ассамблеи Уэльса сказал, что его партия обеспокоена тем, что "это рифмуется с Twp и Pwp".""",
            """Для читателей за предлами Уэльса: по-валлийски twp означает "глупый", а pwp означает "какашка".""",
            """Член Национальной ассамблеи от Плайд сказал, что эта партия в целом "не счастлива" и предложил альтернативы.""",
            """Представитель Консервативной партии Уэльса сказал, что его партия "открыта" к переименованию, но отметил, что между WMP и Muppet небольшая разница в произношении.""",
        ],
    },
    "en-de": {
        "src": [
            """Welsh AMs worried about 'looking like muppets'""",
            """There is consternation among some AMs at a suggestion their title should change to MWPs (Member of the Welsh Parliament).""",
            """It has arisen because of plans to change the name of the assembly to the Welsh Parliament.""",
            """AMs across the political spectrum are worried it could invite ridicule.""",
            """One Labour AM said his group was concerned "it rhymes with Twp and Pwp.\"""",
            """For readers outside of Wales: In Welsh twp means daft and pwp means poo.""",
            """A Plaid AM said the group as a whole was "not happy" and has suggested alternatives.""",
            """A Welsh Conservative said his group was "open minded" about the name change, but noted it was a short verbal hop from MWP to Muppet.""",
        ],
        "tgt": [
            """Walisische Ageordnete sorgen sich "wie Dödel auszusehen\"""",
            """Es herrscht Bestürzung unter einigen Mitgliedern der Versammlung über einen Vorschlag, der ihren Titel zu MWPs (Mitglied der walisischen Parlament) ändern soll.""",
            """Der Grund dafür waren Pläne, den Namen der Nationalversammlung in Walisisches Parlament zu ändern.""",
            """Mitglieder aller Parteien der Nationalversammlung haben Bedenken, dass sie sich dadurch Spott aussetzen könnten.""",
            """Ein Labour-Abgeordneter sagte, dass seine Gruppe "sich mit Twp und Pwp reimt".""",
            """Hinweis für den Leser: „twp“ im Walisischen bedeutet „bescheuert“ und „pwp“ bedeutet „Kacke“.""",
            """Ein Versammlungsmitglied von Plaid Cymru sagte, die Gruppe als Ganzes sei "nicht glücklich" und hat Alternativen vorgeschlagen.""",
            """Ein walisischer Konservativer sagte, seine Gruppe wäre „offen“ für eine Namensänderung, wies aber darauf hin, dass es von „MWP“ (Mitglied des Walisischen Parlaments) nur ein kurzer verbaler Sprung zu „Muppet“ ist.""",
        ],
    },
    "de-en": {
        "src": [
            """Schöne Münchnerin 2018: Schöne Münchnerin 2018 in Hvar: Neun Dates""",
            """Von az, aktualisiert am 04.05.2018 um 11:11""",
            """Ja, sie will...""",
            """\"Schöne Münchnerin" 2018 werden!""",
            """Am Nachmittag wartet erneut eine Überraschung auf unsere Kandidatinnen: sie werden das romantische Candlelight-Shooting vor der MY SOLARIS nicht alleine bestreiten, sondern an der Seite von Male-Model Fabian!""",
            """Hvar - Flirten, kokettieren, verführen - keine einfachen Aufgaben für unsere Mädchen.""",
            """Insbesondere dann, wenn in Deutschland ein Freund wartet.""",
            """Dennoch liefern die neun "Schöne Münchnerin"-Kandidatinnen beim Shooting mit People-Fotograf Tuan ab und trotzen Wind, Gischt und Regen wie echte Profis.""",
        ],
        "tgt": [
            """The Beauty of Munich 2018: the Beauty of Munich 2018 in Hvar: Nine dates""",
            """From A-Z, updated on 04/05/2018 at 11:11""",
            """Yes, she wants to...""",
            """to become "The Beauty of Munich" in 2018!""",
            """In the afternoon there is another surprise waiting for our contestants: they will be competing for the romantic candlelight photo shoot at MY SOLARIS not alone, but together with a male-model Fabian!""",
            """Hvar with its flirting, coquetting, and seduction is not an easy task for our girls.""",
            """Especially when there is a boyfriend waiting in Germany.""",
            """Despite dealing with wind, sprays and rain, the nine contestants of "The Beauty of Munich" behaved like real professionals at the photo shoot with People-photographer Tuan.""",
        ],
    },
}


@require_torch
class ModelTester(unittest.TestCase):
    def get_tokenizer(self, mname):
        return FSMTTokenizer.from_pretrained(mname)

    def get_model(self, mname):
        model = FSMTForConditionalGeneration.from_pretrained(mname).to(torch_device)
        if torch_device == "cuda":
            model.half()
        return model

    @parameterized.expand(
        [
            ["en-ru", 28.21],
            ["ru-en", 23.49],
            ["en-de", 22.11],
            ["de-en", 29.31],
        ]
    )
    @slow
    def test_bleu_scores(self, pair, min_bleu_score):
        # note: this test is not testing the best performance since it only evals a small batch
        # but it should be enough to detect a regression in the output quality
        mname = f"stas/fsmt-wmt19-{pair}"
        tokenizer = self.get_tokenizer(mname)
        model = self.get_model(mname)

        src_sentences = bleu_data[pair]["src"]
        tgt_sentences = bleu_data[pair]["tgt"]

        batch = tokenizer(src_sentences, return_tensors="pt", truncation=True, padding="longest").to(torch_device)
        outputs = model.generate(
            input_ids=batch.input_ids,
            num_beams=8,
        )
        decoded_sentences = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        scores = calculate_bleu(decoded_sentences, tgt_sentences)
        print(scores)
        self.assertGreaterEqual(scores["bleu"], min_bleu_score)
