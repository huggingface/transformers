# Troubleshooting

This is a document explaining how to deal with various issues on github-actions self-hosted CI. The entries may include actual solutions or pointers to Issues that cover those.

## GitHub Actions (self-hosted CI)

* Deepspeed

  - if jit build hangs, clear out `rm -rf ~/.cache/torch_extensions/` reference: https://github.com/huggingface/transformers/pull/12723
