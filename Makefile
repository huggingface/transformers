.PHONY: style

style:
	black --line-length 119 examples templates transformers utils
	isort --recursive examples templates transformers utils
