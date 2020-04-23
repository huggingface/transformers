import collections

from asrtoolkit.clean_formatting import clean_up
from faker import Faker
from fire import Fire


FAKER = Faker()


def make_name():
    name = FAKER.name()
    return (clean_up(name), name)


def make_training_data(train_size=20000, dev_size=1000, test_size=1000):
    for output, size in zip(("train", "dev", "test"), (train_size, dev_size, test_size)):
        with open("{}.txt".format(output), "w") as output_file:
            for _ in range(size):
                output_file.write("input: {}\noutput: {}\n".format(*make_name()))


if __name__ == "__main__":
    Fire(make_training_data)
