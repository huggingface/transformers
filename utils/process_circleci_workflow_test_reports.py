import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow_id', type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id

    command = f'curl -o workflow_jobs.json --location --request GET "https://circleci.com/api/v2/workflow/{workflow_id}/job" --header "Circle-Token: $CIRCLE_TOKEN"'
    os.system(command)

    os.system("tail -1000 workflow_jobs.json")
