import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow_id', type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id

    command = f'curl -o workflow_jobs.json --location --request GET "https://circleci.com/api/v2/workflow/{workflow_id}/job" --header "Circle-Token: $CIRCLE_TOKEN"'
    os.system(command)

    # os.system("tail -1000 workflow_jobs.json")

    with open("workflow_jobs.json") as fp:
        jobs = json.load(fp)

    # for each job, download artifacts
    for job in jobs:
        print(job)

        project_slug = f'gh/{os.environ["CIRCLE_PROJECT_USERNAME"]}/{os.environ["CIRCLE_PROJECT_REPONAME"]}'
        if job["name"].startswith("tests_"):
            url = f'https://circleci.com/api/v2/project/{project_slug}/{job["job_number"]}/artifacts'
            os.system(f'curl -o {job["name"]}_artifacts.json {url} --header "Circle-Token: $CIRCLE_TOKEN"')
            with open(f'{job["name"]}_artifacts.json') as fp:
                job_artifacts = json.load(fp)
                print(job_artifacts)

