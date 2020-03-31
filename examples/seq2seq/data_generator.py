#!/usr/bin/env python3

import collections
import regex as re
from faker.factory import Factory
from fire import Fire

SPACES = re.compile(r"\s+")
FAKE = Factory.create()
FAKE.seed(0)

CONSTANTS = {}

# ordinals
CONSTANTS["ordinal_dict"] = collections.OrderedDict([
    (31, "thirty first"),
    (30, "thirtieth"),
    (29, "twenty-ninth"),
    (28, "twenty-eighth"),
    (27, "twenty-seventh"),
    (26, "twenty-sixth"),
    (25, "twenty-fifth"),
    (24, "twenty-fourth"),
    (23, "twenty-third"),
    (22, "twenty-second"),
    (21, "twenty-first"),
    (20, "twentieth"),
    (19, "nineteenth"),
    (18, "eighteenth"),
    (17, "seventeenth"),
    (16, "sixteenth"),
    (15, "fifteenth"),
    (14, "fourteenth"),
    (13, "thirteenth"),
    (12, "twelfth"),
    (11, "eleventh"),
    (10, "tenth"),
    (9, "ninth"),
    (8, "eighth"),
    (7, "seventh"),
    (6, "sixth"),
    (5, "fifth"),
    (4, "fourth"),
    (3, "third"),
    (2, "second"),
    (1, "first"),
])

CONSTANTS["months"] = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}


def month_num_to_abbrev(month_num):
    return CONSTANTS["months"][month_num]


def ord(n):
    return str(n) + ("th" if 4 <= n % 100 <= 20 else {
        1: "st",
        2: "nd",
        3: "rd"
    }.get(n % 10, "th"))


def day_num_to_str(day_num):
    return (CONSTANTS["ordinal_dict"][day_num])


def truncate_year(year_num):
    return str(year_num)


def make_date():
    output = FAKE.date(pattern="%d/%m/%Y")
    day, month, year = map(int, output.split("/"))
    inp = " ".join(
        [day_num_to_str(day),
         month_num_to_abbrev(month),
         truncate_year(year)])
    return (inp, output)


def make_training_data(train_size=1000, dev_size=1000, test_size=1000):
    for output, size in zip(("train", "dev", "test"),
                            (train_size, dev_size, test_size)):
        with open("{}.txt".format(output), "w") as output_file:
            for _ in range(size):
                output_file.write(
                    "input: {}\noutput: {}\n".format(*make_date()))


if __name__ == "__main__":
    Fire(make_training_data)
