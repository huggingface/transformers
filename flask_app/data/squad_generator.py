import argparse
import json


def create_example_dict(context, id, questions):
    qas = []
    count = id
    for question in questions:
        qas.append({
                "answers": [{"answer_start": -1, "text": ""}],
                "id": count,
                "is_impossible": False,
                "question": question,
                }
        )
        count += 1

    return {
        "context": context,
        "qas": qas,
    }

def create_para_dict(example_dicts, title=""):
    if type(example_dicts) == dict:
        example_dicts = [example_dicts]
    return {"title": title,
            "paragraphs": example_dicts}

def convert_file_input_to_squad(input_file, output_file):
    with open(input_file, "r") as f:
        raw_text = f.read()
    return convert_text_input_to_squad(raw_text, output_file)

def validate_squad_input(input):
    paragraphs = input.split("\n\n")
    for p in paragraphs:
        p = p.split("\n")
        if len(p) < 3:
            return False
        else:
            questions = p[2:]
            for q in questions:
                if not q:
                    return False
    return True


def convert_text_input_to_squad(raw_text, output_file):

    raw_text = raw_text.strip()

    assert validate_squad_input(raw_text)

    paragraphs = raw_text.split("\n\n")

    squad_dict = {"data": []}
    count = 0

    for p in paragraphs:
        p = p.split("\n")
        title = p[0]
        context = p[1]
        questions = p[2:]


        squad_dict["data"].append(
            create_para_dict(
                create_example_dict(
                    context=context,
                    id=count,
                    questions=questions,
                ),
            title)
        )
        count += len(questions)

    with open(output_file, "w+") as f:
        json.dump(squad_dict, f)

    return squad_dict

def convert_context_and_questions_to_squad(context, questions, output_file):

    squad_dict = {"data": []}
    count = 0

    title = context.split("\n")[0]
    context = context[len(title):]
    questions = questions.split("\n")

    squad_dict["data"].append(
        create_para_dict(
            create_example_dict(
                context=context,
                id=count,
                questions=questions,
            ),
        title)
    )


    with open(output_file, "w+") as f:
        json.dump(squad_dict, f)

    return squad_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default=None, type=str, required=True,
                        help="File to convert to SQuAD format")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Output file location")
    args = parser.parse_args()

    convert_file_input_to_squad(args.file, args.output)


if __name__ == "__main__":
    main()
