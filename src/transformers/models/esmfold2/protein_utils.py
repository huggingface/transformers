# coding=utf-8
# Copyright 2026 Biohub. All rights reserved.
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
"""Self-contained protein featurization for ESMFold2 inference."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

MOL_TYPE_PROTEIN = 0
PROTEIN_UNK_RES_TYPE = 22
MSA_GAP_TOKEN_ID = 1

PROTEIN_RESIDUE_TO_RES_TYPE: dict[str, int] = {
    "ALA": 2,
    "ARG": 3,
    "ASN": 4,
    "ASP": 5,
    "CYS": 6,
    "GLN": 7,
    "GLU": 8,
    "GLY": 9,
    "HIS": 10,
    "ILE": 11,
    "LEU": 12,
    "LYS": 13,
    "MET": 14,
    "PHE": 15,
    "PRO": 16,
    "SER": 17,
    "THR": 18,
    "TRP": 19,
    "TYR": 20,
    "VAL": 21,
}

PROTEIN_1TO3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}

ESM_PROTEIN_VOCAB: dict[str, int] = {
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 3,
}

# Heavy atoms per canonical residue, in training-time order.
PROTEIN_HEAVY_ATOMS: dict[str, list[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK": ["N", "CA", "C", "O"],
}

PROTEIN_REF_POS: dict[str, dict[str, tuple[float, float, float]]] = {
    "ALA": {
        "N": (-0.01003183238208294, -1.2073018550872803, -1.0555061101913452),
        "CA": (-0.04190138354897499, 0.17447763681411743, -0.5729365348815918),
        "C": (1.2127548456192017, 0.4737588167190552, 0.19521640241146088),
        "O": (1.9390329122543335, 1.4484562873840332, -0.13759790360927582),
        "CB": (-1.276943325996399, 0.4288230538368225, 0.29937705397605896),
    },
    "ARG": {
        "N": (-2.0170421600341797, 0.6717798113822937, -1.1794233322143555),
        "CA": (-2.0503084659576416, -0.5735036730766296, -0.4097220301628113),
        "C": (-3.469440460205078, -1.0612813234329224, -0.2755832374095917),
        "O": (-3.8218462467193604, -2.1369943618774414, -0.8294969797134399),
        "CB": (-1.4193516969680786, -0.3735991418361664, 0.9852858781814575),
        "CG": (0.11878877878189087, -0.3112654983997345, 0.963895857334137),
        "CD": (0.6643245816230774, 1.0068185329437256, 0.3963329493999481),
        "NE": (2.1090238094329834, 1.0977025032043457, 0.6120952367782593),
        "CZ": (3.098905324935913, 0.3215920031070709, -0.09047172218561172),
        "NH1": (4.461230278015137, 0.3844667971134186, 0.34141138195991516),
        "NH2": (2.7856509685516357, -0.4166366159915924, -1.1148239374160767),
    },
    "ASN": {
        "N": (-0.7595629096031189, 0.7503494620323181, 1.1369825601577759),
        "CA": (-0.76087886095047, 0.23876343667507172, -0.23573364317417145),
        "C": (-1.9211044311523438, -0.6982439160346985, -0.42196929454803467),
        "O": (-2.677666187286377, -0.5753439664840698, -1.4223182201385498),
        "CB": (0.5504899024963379, -0.5078350305557251, -0.5390339493751526),
        "CG": (1.7250099182128906, 0.4264017939567566, -0.5778228640556335),
        "OD1": (1.9470350742340088, 1.1086392402648926, -1.613560438156128),
        "ND2": (2.57365345954895, 0.5730618834495544, 0.5608599781990051),
    },
    "ASP": {
        "N": (-1.8452696800231934, -1.2169504165649414, 0.19437327980995178),
        "CA": (-0.6379959583282471, -0.41974392533302307, 0.41681644320487976),
        "C": (-0.9431572556495667, 1.0356197357177734, 0.18555717170238495),
        "O": (-1.5183608531951904, 1.4045922756195068, -0.8739855885505676),
        "CB": (0.48594576120376587, -0.8970447778701782, -0.5209363698959351),
        "CG": (1.780342936515808, -0.19918935000896454, -0.2310730367898941),
        "OD1": (2.5202910900115967, -0.6044584512710571, 0.7049641013145447),
        "OD2": (2.1454880237579346, 0.9208861589431763, -0.9712985157966614),
    },
    "CYS": {
        "N": (0.0469963513314724, 1.190075159072876, -1.1607273817062378),
        "CA": (0.11344368755817413, -0.09400428831577301, -0.45952197909355164),
        "C": (-1.2652032375335693, -0.6832379698753357, -0.3594406247138977),
        "O": (-1.4631439447402954, -1.8851220607757568, -0.6826791763305664),
        "CB": (0.6919880509376526, 0.09034398198127747, 0.952482283115387),
        "SG": (2.4619927406311035, 0.5235707759857178, 0.9020372629165649),
    },
    "GLN": {
        "N": (-2.370004653930664, -0.9637529850006104, -0.7942749261856079),
        "CA": (-1.370002269744873, -0.6000258922576904, 0.2103111445903778),
        "C": (-1.7545503377914429, 0.7091967463493347, 0.8433493971824646),
        "O": (-1.8520662784576416, 0.7999289631843567, 2.0964975357055664),
        "CB": (0.02040259726345539, -0.5004461407661438, -0.44764479994773865),
        "CG": (1.1377512216567993, -0.28680720925331116, 0.582992434501648),
        "CD": (2.4745187759399414, -0.24800164997577667, -0.09364881366491318),
        "OE1": (3.1685523986816406, -1.2966246604919434, -0.1717153936624527),
        "NE2": (2.947425603866577, 0.9601329565048218, -0.6888364553451538),
    },
    "GLU": {
        "N": (-1.5850872993469238, -1.337684154510498, 0.9490851163864136),
        "CA": (-1.0560977458953857, 0.027459044009447098, 1.0306966304779053),
        "C": (-1.7741456031799316, 0.9664392471313477, 0.09259600937366486),
        "O": (-1.9012441635131836, 2.181349992752075, 0.402479350566864),
        "CB": (0.4706551432609558, 0.048803869634866714, 0.8114414811134338),
        "CG": (0.9133604764938354, -0.4219329059123993, -0.5830985307693481),
        "CD": (2.398822069168091, -0.3097084164619446, -0.7210537791252136),
        "OE1": (3.1389315128326416, -1.274524450302124, -0.39029765129089355),
        "OE2": (2.9647817611694336, 0.8781346082687378, -1.1732689142227173),
    },
    "GLY": {
        "N": (-1.3942985534667969, -0.39875128865242004, -0.3370324671268463),
        "CA": (-0.39974430203437805, 0.5488945245742798, 0.15242962539196014),
        "C": (0.9440054893493652, -0.10314033925533295, 0.19859643280506134),
        "O": (1.3352899551391602, -0.669218122959137, 1.2541258335113525),
    },
    "HIS": {
        "N": (-1.4532867670059204, -1.0689626932144165, 0.881072461605072),
        "CA": (-1.3396095037460327, 0.24797579646110535, 0.24960045516490936),
        "C": (-2.675257921218872, 0.6571555733680725, -0.30441102385520935),
        "O": (-3.1311378479003906, 1.8079776763916016, -0.06785715371370316),
        "CB": (-0.3041955828666687, 0.21721023321151733, -0.8885309100151062),
        "CG": (1.0887513160705566, 0.028941065073013306, -0.36419469118118286),
        "ND1": (1.840459942817688, 1.0411773920059204, 0.29804590344429016),
        "CD2": (1.780855417251587, -1.1011489629745483, -0.3814258575439453),
        "CE1": (2.9566943645477295, 0.4924798905849457, 0.6477115750312805),
        "NE2": (3.0280203819274902, -0.8751969337463379, 0.26084381341934204),
    },
    "ILE": {
        "N": (-0.7167549729347229, -1.5426139831542969, -0.9983330368995667),
        "CA": (-1.0636085271835327, -0.35169270634651184, -0.21393552422523499),
        "C": (-1.3896740674972534, 0.8142145276069641, -1.1164065599441528),
        "O": (-1.2377792596817017, 0.7302915453910828, -2.3656840324401855),
        "CB": (0.061667006462812424, 0.01599610224366188, 0.8057394623756409),
        "CG1": (1.502519965171814, -0.08899776637554169, 0.24154816567897797),
        "CG2": (-0.053174979984760284, -0.8521055579185486, 2.0702083110809326),
        "CD1": (1.7929610013961792, 0.899773120880127, -0.8863027691841125),
    },
    "LEU": {
        "N": (1.9657520055770874, -1.9763224124908447, -0.18391533195972443),
        "CA": (1.3077669143676758, -0.6677430868148804, -0.19492436945438385),
        "C": (1.9905058145523071, 0.24182087182998657, 0.7879968285560608),
        "O": (2.06896710395813, -0.07880014181137085, 2.0048046112060547),
        "CB": (-0.20306941866874695, -0.8093230128288269, 0.11243502795696259),
        "CG": (-0.9916267395019531, 0.5234957337379456, 0.06723011285066605),
        "CD1": (-2.4228057861328125, 0.29949337244033813, 0.573042094707489),
        "CD2": (-1.0282856225967407, 1.1250264644622803, -1.346014380455017),
    },
    "LYS": {
        "N": (2.4221372604370117, -0.6473312377929688, 0.6370573043823242),
        "CA": (2.0314927101135254, 0.2786507308483124, -0.4298512041568756),
        "C": (2.7168593406677246, 1.595757246017456, -0.20924785733222961),
        "O": (3.397681713104248, 2.116427421569824, -1.1332510709762573),
        "CB": (0.5018402934074402, 0.4873858690261841, -0.49062973260879517),
        "CG": (-0.25062066316604614, -0.7894009947776794, -0.9055535793304443),
        "CD": (-1.769762635231018, -0.5552700161933899, -1.040329933166504),
        "CE": (-2.576533555984497, -1.0221366882324219, 0.18493641912937164),
        "NZ": (-2.269151210784912, -0.24293844401836395, 1.3849012851715088),
    },
    "MET": {
        "N": (1.8903918266296387, -1.5252995491027832, -0.42638593912124634),
        "CA": (1.2630571126937866, -0.24417810142040253, -0.7626462578773499),
        "C": (2.30391001701355, 0.8367712497711182, -0.7254616618156433),
        "O": (2.465414524078369, 1.5928632020950317, -1.7207728624343872),
        "CB": (0.10567972809076309, 0.10861825942993164, 0.19741646945476532),
        "CG": (-1.0658042430877686, -0.8736631274223328, 0.08811883628368378),
        "SD": (-2.4557132720947266, -0.3332225978374481, 1.1461700201034546),
        "CE": (-3.265165090560913, 0.7033554911613464, -0.11588376015424728),
    },
    "PHE": {
        "N": (-2.8484435081481934, -1.525790810585022, 0.01789816841483116),
        "CA": (-1.591969609260559, -0.8545162677764893, 0.35214468836784363),
        "C": (-1.8900631666183472, 0.45833414793014526, 1.0232222080230713),
        "O": (-1.3424992561340332, 0.74432373046875, 2.121629476547241),
        "CB": (-0.760358452796936, -0.6342853307723999, -0.9257160425186157),
        "CG": (0.604112982749939, -0.07200468331575394, -0.6148118376731873),
        "CD1": (0.8468314409255981, 1.2480632066726685, -0.7146694660186768),
        "CD2": (1.6827683448791504, -0.9758077263832092, -0.1423054188489914),
        "CE1": (2.1801748275756836, 1.7875733375549316, -0.3744623064994812),
        "CE2": (2.888307809829712, -0.48277512192726135, 0.16804970800876617),
        "CZ": (3.149812936782837, 0.9656873941421509, 0.04440271109342575),
    },
    "PRO": {
        "N": (-0.836250364780426, -0.9899801015853882, 0.5561304688453674),
        "CA": (0.32722190022468567, -0.6164458394050598, -0.25072571635246277),
        "C": (1.6121541261672974, -1.1711241006851196, 0.31082412600517273),
        "O": (1.6127740144729614, -2.2771971225738525, 0.9156193733215332),
        "CB": (0.3248198926448822, 0.9028244018554688, -0.33368146419525146),
        "CG": (-1.1425083875656128, 1.2730128765106201, -0.2590600252151489),
        "CD": (-1.8495968580245972, 0.026575811207294464, 0.2681289613246918),
    },
    "SER": {
        "N": (0.674650251865387, 1.5018702745437622, -0.5367295145988464),
        "CA": (0.00013792862591799349, 0.4966467022895813, 0.28510504961013794),
        "C": (0.9941009879112244, -0.5374617576599121, 0.73505038022995),
        "O": (1.0545241832733154, -0.8683545589447021, 1.9495396614074707),
        "CB": (-1.1279288530349731, -0.1659376323223114, -0.5160963535308838),
        "OG": (-1.8135979175567627, -1.085249662399292, 0.28947514295578003),
    },
    "THR": {
        "N": (-1.325830340385437, -1.3728225231170654, 0.6882233023643494),
        "CA": (-0.5433306097984314, -0.16364754736423492, 0.41697052121162415),
        "C": (-1.294381856918335, 0.7077372074127197, -0.5549946427345276),
        "O": (-1.6939635276794434, 0.23654410243034363, -1.6540418863296509),
        "CB": (0.853203296661377, -0.5363803505897522, -0.14109353721141815),
        "OG1": (1.5220820903778076, -1.379003643989563, 0.7635167837142944),
        "CG2": (1.7225933074951172, 0.7054727077484131, -0.3651331067085266),
    },
    "TRP": {
        "N": (3.686030864715576, 0.7599999904632568, 0.496155709028244),
        "CA": (2.384092092514038, 0.09079249948263168, 0.5325262546539307),
        "C": (2.1113572120666504, -0.6121063232421875, -0.7733646035194397),
        "O": (1.796526312828064, -1.8323148488998413, -0.7775964140892029),
        "CB": (1.281521201133728, 1.1139036417007446, 0.8559791445732117),
        "CG": (-0.04292375594377518, 0.44645074009895325, 1.0942792892456055),
        "CD1": (-0.42329534888267517, -0.15470874309539795, 2.2227554321289062),
        "CD2": (-1.1023900508880615, 0.2158389836549759, 0.11529432237148285),
        "NE1": (-1.7030320167541504, -0.7665823101997375, 2.0595016479492188),
        "CE2": (-2.045644998550415, -0.4881173074245453, 0.710669219493866),
        "CE3": (-1.2173502445220947, 0.6102271676063538, -1.300106406211853),
        "CZ2": (-3.256009340286255, -0.9164394736289978, -0.00984987337142229),
        "CZ3": (-2.315925121307373, 0.2306906282901764, -1.9776310920715332),
        "CH2": (-3.3817875385284424, -0.5677337646484375, -1.3032053709030151),
    },
    "TYR": {
        "N": (-1.7900604009628296, -0.8409399390220642, 1.3180142641067505),
        "CA": (-1.913882851600647, 0.23552845418453217, 0.330669641494751),
        "C": (-3.347280740737915, 0.3588399887084961, -0.09830684959888458),
        "O": (-3.967811346054077, -0.6449354290962219, -0.5423302054405212),
        "CB": (-1.0093992948532104, 0.0004731413209810853, -0.8981552124023438),
        "CG": (0.4520410895347595, 0.021162061020731926, -0.5305932760238647),
        "CD1": (1.0992432832717896, 1.1877919435501099, -0.3579142987728119),
        "CD2": (1.1803174018859863, -1.253401279449463, -0.31122180819511414),
        "CE1": (2.5253450870513916, 1.1990256309509277, 0.029804613441228867),
        "CE2": (2.471151113510132, -1.240687608718872, 0.043534230440855026),
        "CZ": (3.180687665939331, 0.04672492295503616, 0.2214856892824173),
        "OH": (4.523719787597656, 0.0671030730009079, 0.5877485871315002),
    },
    "VAL": {
        "N": (0.5987519025802612, -1.569443702697754, -0.7379124760627747),
        "CA": (0.6014357209205627, -0.10503966361284256, -0.6336286664009094),
        "C": (1.8391697406768799, 0.4067850410938263, 0.06351757049560547),
        "O": (2.3952062129974365, -0.2666190266609192, 0.9731166958808899),
        "CB": (-0.694736897945404, 0.4259096384048462, 0.03581475466489792),
        "CG1": (-1.9276031255722046, 0.09515828639268875, -0.8172357082366943),
        "CG2": (-0.8938426971435547, -0.08640842139720917, 1.472349762916565),
    },
    "UNK": {
        "N": (0.0, 0.0, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (0.0, 0.0, 0.0),
        "O": (0.0, 0.0, 0.0),
    },
}

# Protonated nitrogens at physiological pH (matches CHARGED_ATOMS in the
# opensource constants for the protein subset).
PROTEIN_CHARGED_ATOMS: dict[tuple[str, str], int] = {
    ("LYS", "NZ"): 1,
    ("ARG", "NH2"): 1,
    ("HIS", "ND1"): 1,
}

# Only the elements that appear in canonical protein heavy atoms.
_PROTEIN_ELEMENT_TO_ATOMIC_NUM: dict[str, int] = {"C": 6, "N": 7, "O": 8, "S": 16}


def _encode_atom_name(name: str) -> list[int]:
    padded = name.ljust(4)[:4]
    return [ord(c) - 32 if c != " " else 0 for c in padded]


def prepare_protein_features(sequence: str) -> dict[str, Tensor]:
    """Featurize a single protein sequence for ESMFold2ExperimentalModel.forward.

    Returns the same keys with the same dtypes/shapes as
    ``ESMFold2InputBuilder.prepare_input(StructurePredictionInput(...))``
    restricted to a single-chain protein with no MSA, modifications,
    distogram conditioning, or covalent bonds. All tensors have a
    leading batch dim of 1; the caller is responsible for moving them
    to the model device.
    """
    if not sequence:
        raise ValueError("sequence must be non-empty")

    res_3letter = [PROTEIN_1TO3.get(c, "UNK") for c in sequence]
    L = len(sequence)

    token_atom_starts: list[int] = []
    atom_records: list[tuple[int, str, str, int, tuple[float, float, float]]] = []
    res_type_vals: list[int] = []
    input_id_vals: list[int] = []
    distogram_rep_atom_idx: list[int] = []

    atom_cursor = 0
    for t_idx, (letter, res_3) in enumerate(zip(sequence, res_3letter)):
        atom_names = PROTEIN_HEAVY_ATOMS[res_3]
        res_type = PROTEIN_RESIDUE_TO_RES_TYPE.get(res_3, PROTEIN_UNK_RES_TYPE)
        input_id = ESM_PROTEIN_VOCAB.get(letter, ESM_PROTEIN_VOCAB["X"])

        token_atom_starts.append(atom_cursor)
        for name in atom_names:
            charge = PROTEIN_CHARGED_ATOMS.get((res_3, name), 0)
            element = name[0]  # protein heavy atoms are all single-letter C/N/O/S
            ref_pos = PROTEIN_REF_POS[res_3][name]
            atom_records.append((t_idx, name, element, charge, ref_pos))
            atom_cursor += 1

        rep_name = "CB" if "CB" in atom_names else "CA"
        distogram_rep_atom_idx.append(
            token_atom_starts[t_idx] + atom_names.index(rep_name)
        )

        res_type_vals.append(res_type)
        input_id_vals.append(input_id)

    n_real_atoms = len(atom_records)
    n_atoms = math.ceil(n_real_atoms / 32) * 32 if n_real_atoms > 0 else 32

    ref_pos = torch.zeros(n_atoms, 3, dtype=torch.float32)
    ref_element = torch.zeros(n_atoms, dtype=torch.int64)
    ref_charge = torch.zeros(n_atoms, dtype=torch.int8)
    ref_atom_name_chars = torch.zeros(n_atoms, 4, dtype=torch.int64)
    ref_space_uid = torch.zeros(n_atoms, dtype=torch.int64)
    atom_attention_mask = torch.zeros(n_atoms, dtype=torch.bool)
    atom_to_token = torch.zeros(n_atoms, dtype=torch.int64)

    for i, (t_idx, name, element, charge, pos) in enumerate(atom_records):
        ref_pos[i] = torch.tensor(pos, dtype=torch.float32)
        ref_element[i] = _PROTEIN_ELEMENT_TO_ATOMIC_NUM[element]
        ref_charge[i] = charge
        ref_atom_name_chars[i] = torch.tensor(
            _encode_atom_name(name), dtype=torch.int64
        )
        ref_space_uid[i] = t_idx
        atom_attention_mask[i] = True
        atom_to_token[i] = t_idx

    token_index = torch.arange(L, dtype=torch.int64)
    residue_index = torch.arange(L, dtype=torch.int64)
    asym_id = torch.zeros(L, dtype=torch.int64)
    sym_id = torch.zeros(L, dtype=torch.int64)
    entity_id = torch.ones(L, dtype=torch.int64)
    mol_type = torch.full((L,), MOL_TYPE_PROTEIN, dtype=torch.int64)
    res_type = torch.tensor(res_type_vals, dtype=torch.int64)
    input_ids = torch.tensor(input_id_vals, dtype=torch.int64)
    token_bonds = torch.zeros(L, L, 1, dtype=torch.float32)
    token_attention_mask = torch.ones(L, dtype=torch.bool)
    distogram_atom_idx = torch.tensor(distogram_rep_atom_idx, dtype=torch.int64)

    # Single-sequence MSA: depth 1, row 0 is the sequence itself.
    msa = res_type.unsqueeze(0)
    msa_attention_mask = torch.ones(1, L, dtype=torch.bool)
    has_deletion = torch.zeros(1, L, dtype=torch.bool)
    deletion_value = torch.zeros(1, L, dtype=torch.float32)
    deletion_mean = torch.zeros(L, dtype=torch.float32)

    features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "sym_id": sym_id,
        "entity_id": entity_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "input_ids": input_ids,
        "token_bonds": token_bonds,
        "token_attention_mask": token_attention_mask,
        "ref_pos": ref_pos,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_attention_mask": atom_attention_mask,
        "atom_to_token": atom_to_token,
        "distogram_atom_idx": distogram_atom_idx,
        "msa": msa,
        "msa_attention_mask": msa_attention_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_mean,
    }
    return {k: v.unsqueeze(0) for k, v in features.items()}


# 0-32 res_type → 3-letter name (only protein indices 2-22 are populated).
_RES_TYPE_TO_3LETTER: dict[int, str] = {
    rt: three for three, rt in PROTEIN_RESIDUE_TO_RES_TYPE.items()
}
_RES_TYPE_TO_3LETTER[PROTEIN_UNK_RES_TYPE] = "UNK"

# Featurization keys that ``output_to_pdb`` reads off the forward output.
# ``infer_protein`` re-attaches them because ``forward`` does not echo them
# back; both ESMFold2 model classes share this list.
OUTPUT_TO_PDB_FEATURE_KEYS: tuple[str, ...] = (
    "res_type",
    "atom_to_token",
    "ref_atom_name_chars",
    "atom_attention_mask",
    "token_attention_mask",
    "residue_index",
)


def output_to_pdb(output: dict) -> str:
    """Convert an ESMFold2 protein forward output into a PDB string.

    Expects ``output`` to carry the featurization keys re-attached by
    ``infer_protein`` (``res_type``, ``atom_to_token``,
    ``ref_atom_name_chars``, ``atom_attention_mask``,
    ``token_attention_mask``, ``residue_index``) alongside the predicted
    ``sample_atom_coords`` and ``plddt``. Builds a 37-atom
    ``OFProtein`` (per-atom pLDDT in the b-factor column) and renders it
    with the OpenFold utilities shipped in ``transformers.models.esm``.
    """
    from transformers.models.esm.openfold_utils import OFProtein, to_pdb
    from transformers.models.esm.openfold_utils import residue_constants as rc

    coords = output["sample_atom_coords"]
    if coords.dim() == 4:
        coords = coords[:, 0]
    coords = coords.detach().cpu().numpy()[0]

    plddt = output["plddt"].detach().cpu().numpy()[0]
    atom_to_token = output["atom_to_token"].cpu().numpy()
    ref_chars = output["ref_atom_name_chars"].cpu().numpy()
    res_type = output["res_type"].cpu().numpy()
    token_mask = output["token_attention_mask"].cpu().numpy().astype(bool)
    atom_mask_in = output["atom_attention_mask"].cpu().numpy().astype(bool)
    residue_index_arr = output["residue_index"].cpu().numpy()

    if atom_to_token.ndim == 2:
        atom_to_token = atom_to_token[0]
        ref_chars = ref_chars[0]
        res_type = res_type[0]
        token_mask = token_mask[0]
        atom_mask_in = atom_mask_in[0]
        residue_index_arr = residue_index_arr[0]

    valid_tok = np.where(token_mask)[0]
    n_res = valid_tok.shape[0]

    aatype = np.full(n_res, rc.restype_order_with_x["X"], dtype=np.int64)
    for new_i, t in enumerate(valid_tok):
        rt = int(res_type[t])
        three = _RES_TYPE_TO_3LETTER.get(rt)
        if three is None or three == "UNK":
            aatype[new_i] = rc.restype_order_with_x["X"]
        else:
            one = rc.restype_3to1.get(three, "X")
            aatype[new_i] = rc.restype_order_with_x[one]

    atom_positions = np.zeros((n_res, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, 37), dtype=np.float32)
    b_factors = np.zeros((n_res, 37), dtype=np.float32)
    tok_to_new = {int(t): i for i, t in enumerate(valid_tok)}

    for a in range(atom_to_token.shape[0]):
        if not atom_mask_in[a]:
            continue
        tok = int(atom_to_token[a])
        if tok not in tok_to_new:
            continue
        new_i = tok_to_new[tok]
        name = "".join(
            chr(int(c) + 32) if int(c) != 0 else " " for c in ref_chars[a]
        ).strip()
        idx37 = rc.atom_order.get(name)
        if idx37 is None:
            continue
        atom_positions[new_i, idx37] = coords[a]
        atom_mask[new_i, idx37] = 1.0
        b_factors[new_i, idx37] = float(plddt[tok])

    pred = OFProtein(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index_arr[valid_tok].astype(np.int32) + 1,
        b_factors=b_factors,
    )
    return to_pdb(pred)
