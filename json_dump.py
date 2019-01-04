import json
import argparse




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input", default=None, type=str, required=True,
                        help="input file")
    parser.add_argument("--output", default=None, type=str, required=False,
                        help="output file")
    parser.add_argument("--silence", default=False, action='store_true', help="silence mode")                        
    args = parser.parse_args()

    with open(args.input) as data_file:
        data = json.load(data_file)

    with open(args.output, 'w', encoding='UTF-8') as file:
        file.write(json.dumps(data, ensure_ascii=False))
    if not args.silence:
        print('print data')
        print(data)
    print('ok')

if __name__ == "__main__":
    main()
