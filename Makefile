install:
	pip uninstall -y transformers
	python setup.py install
	rm -rfv dist build
