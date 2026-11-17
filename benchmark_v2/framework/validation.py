from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus


def validate_generation_outputs(
    outputs: dict[str, GenerationOutput], inputs: list[list[int]], num_return_sequences: int = 1
) -> None:
    """Validate results from a fresh, nonpersistent generate_batch manager before measuring or scoring."""
    if not inputs:
        raise ValueError("A benchmark workload must contain at least one request")
    # Fresh managers assign req_0, req_1, ... in input order, even when scheduling reorders prompts.
    expected = {}
    for index, prompt in enumerate(inputs):
        request_id = f"req_{index}"
        expected[request_id] = prompt
        for child in range(num_return_sequences - 1):
            expected[f"{request_id}__child#{child}"] = prompt
    missing = expected.keys() - outputs.keys()
    unexpected = outputs.keys() - expected.keys()
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete generation results: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for request_id, prompt in expected.items():
        output = outputs[request_id]
        if output.request_id != request_id or output.prompt_ids != prompt:
            raise RuntimeError(f"Generation result does not match input for {request_id}")
        if output.error is not None or output.status != RequestStatus.FINISHED:
            raise RuntimeError(f"Generation failed for {request_id}: status={output.status}, error={output.error}")
