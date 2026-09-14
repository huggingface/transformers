# SCQOS governed model-transition proof for Transformers

This document records an external Supreme Computation / SCQOS governance experiment performed against a pinned Hugging Face Transformers state.

It does **not** change Transformers runtime behavior. It demonstrates that authorization can be tied to one exact frozen transition, invalidated when that state changes, and re-issued only after a fresh qualification.

## Frozen target

- Target repository: `huggingface/transformers`
- Model: `google-bert/bert-base-uncased`
- Transformers revision: `9d13f8cf6e22f9cac94601eee286e768cc77cf46`
- Model weights SHA-256: `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`

## Qualification N

- Transition: `9dc93e28-c699-5593-b684-38849c1f4c4f`
- Decision: `PERMIT`
- Permit: `b6e34c5e-d21a-5c2f-9887-15718facb974`

## Adversarial change after qualification

The tokenizer configuration changed from `model_max_length: 512` to `model_max_length: 513`. That created transition `53e700a6-a625-55bc-b783-bb242bf8d91d`.

The old permit was reused against the mutated state. The live AWS DynamoDB conditional boundary blocked that reuse and SCQOS returned `HOLD`. Failed invariants were Time, Continuity, Genesis, Boundary, Reference, and Causality.

## Requalification N+1

After the mutated state was treated as a new transition and requalified:

- Transition: `27bddaa9-5e82-54f9-bf7f-ca6cf4ea3d9a`
- Decision: `PERMIT_N+1`
- Permit: `65e038ee-c191-56ff-80fc-a73c959cc5ea`

`PERMIT_N+1` is a new authorization bound to a new frozen transition identity. It is not a resurrection of `PERMIT_N`.

## Durable AWS evidence

- S3 bucket: `scqos-governance-evidence-us-east-1`
- S3 object key: `huggingface-transformers/20260910T193931Z/scqos-model-transition-proof.json`
- S3 version: `Rdbo0bBBDtKPQJgl4j2eWOfhcnpf9nSO`
- S3 SHA-256: `99a997329568dee3f3d6fea2aa7df572e34ba38a2da7604ce8adfed5d94f8825`
- DynamoDB receipts: `hf-transformers-permit-n-20260910T193931Z`, `hf-transformers-permit-n1-20260910T193931Z`

## Result

Change one frozen transition component after qualification and the old PERMIT cannot execute. Requalification produces a new transition identity and a new PERMIT.

In plain English: **permission belongs to the exact state that earned it. Change the governed state and the old permission is no longer valid.**

Nothing Executes Until It Proves Itself.
