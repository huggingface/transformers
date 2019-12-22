.PHONY: style

style:
	black --line-length 119 examples templates tests transformers utils
	isort --recursive examples templates tests transformers utils
