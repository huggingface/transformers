import argparse
import json
import subprocess


def get_runner_status(target_runners, token):
    offline_runners = []

    cmd = (
        f'curl -H "Accept: application/vnd.github+json" -H "Authorization: Bearer {token}"'
        " https://api.github.com/repos/huggingface/transformers/actions/runners"
    )
    output = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    o = output.stdout.decode("utf-8")
    status = json.loads(o)

    runners = status["runners"]
    for runner in runners:
        if runner["name"] in target_runners:
            if runner["status"] == "offline":
                offline_runners.append(runner)

    # save the result so we can report them on Slack
    with open("offline_runners.txt", "w") as fp:
        fp.write(json.dumps(offline_runners))

    if len(offline_runners) > 0:
        failed = "\n".join([x["name"] for x in offline_runners])
        raise ValueError(f"The following runners are offline:\n{failed}")


if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--target_runners",
        default=None,
        type=list_str,
        required=True,
        help="Comma-separated list of runners to check status.",
    )

    parser.add_argument(
        "--token", default=None, type=str, required=True, help="A token that has actions:read permission."
    )
    args = parser.parse_args()

    get_runner_status(args.target_runners, args.token)
