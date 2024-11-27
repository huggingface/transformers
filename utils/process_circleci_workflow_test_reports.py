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
        jobs = json.load(fp)["items"]

    workflow_summary = {}
    # for each job, download artifacts
    for job in jobs:
        print(job)

        project_slug = f'gh/{os.environ["CIRCLE_PROJECT_USERNAME"]}/{os.environ["CIRCLE_PROJECT_REPONAME"]}'
        if job["name"].startswith("tests_"):

            os.system(f'mkdir {job["name"]}')

            url = f'https://circleci.com/api/v2/project/{project_slug}/{job["job_number"]}/artifacts'
            os.system(f'curl -o {job["name"]}_artifacts.json {url} --header "Circle-Token: $CIRCLE_TOKEN"')
            with open(f'{job["name"]}_artifacts.json') as fp:
                job_artifacts = json.load(fp)["items"]
                print(job_artifacts)

                job_test_summaries = {}
                for artifact in job_artifacts:
                    if artifact["path"].startswith("reports/") and artifact["path"].endswith("/summary_short.txt"):
                        node_index = artifact["node_index"]
                        url = artifact["url"]
                        fn = f'{job["name"]}/test_summary_{node_index}.txt'
                        # -L is necessary
                        command = f'curl -L -o {fn} {url} --header "Circle-Token: $CIRCLE_TOKEN"'
                        os.system(command)

                        # add each
                        with open(fn) as fp:
                            test_summary = fp.read()
                            job_test_summaries[node_index] = test_summary

                summary = {}
                for node_index, node_test_summary in job_test_summaries.items():
                    for line in node_test_summary.splitlines():
                        if line.startswith("PASSED "):
                            test = line[len("PASSED "):]
                            summary[test] = "passed"
                        elif line.startswith("FAILED "):
                            test = line[len("FAILED "):].split()[0]
                            summary[test] = "failed"
                # failed before passed
                summary = {k: v for k, v in sorted(summary.items(), key=lambda x: (x[1], x[0]))}
                workflow_summary[job["name"]] = summary

                # collected version
                with open(f'{job["name"]}/test_summary.json', "w") as fp:
                    json.dump(summary, fp, indent=4)
                # print(summary)

    new_workflow_summary = {}
    for job_name, job_summary in workflow_summary.items():
        for test, status in job_summary.items():
            if test not in new_workflow_summary:
                new_workflow_summary[test] = {}
            new_workflow_summary[test][job_name] = status

    for test, result in workflow_summary.items():
        workflow_summary[test] = sorted(result)
    workflow_summary = sorted(workflow_summary)

    with open(f'test_summary.json', "w") as fp:
        json.dump(workflow_summary, fp, indent=4)
