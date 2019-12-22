.PHONY: style

style:
	black --line-length 119 examples templates tests src utils
	isort --recursive examples templates tests src utils
