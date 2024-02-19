selective_scan_commit := 2a3704fd47ba817b415627b06fd796b971fdc137

causal-conv1d:
	rm -rf causal-conv1d
	git clone https://github.com/Dao-AILab/causal-conv1d.git

build-causal-conv1d: causal-conv1d
	cd causal-conv1d/ && git checkout v1.1.1 # known latest working version tag
	cd causal-conv1d/ && CAUSAL_CONV1D_FORCE_BUILD=TRUE python setup.py build

install-causal-conv1d: build-causal-conv1d
	pip uninstall causal-conv1d -y || true
	cd causal-conv1d/ && pip install .

# selective-scan dependends on causal-conv1d
selective-scan: 
	rm -rf mamba
	git clone https://github.com/state-spaces/mamba.git mamba

build-selective-scan: selective-scan
	cd mamba/ && git fetch && git checkout $(selective_scan_commit)
	cd mamba && python setup.py build

install-selective-scan: install-causal-conv1d build-selective-scan 
	pip uninstall selective-scan-cuda -y || true
	cd mamba && pip install .

build-all: build-causal-conv1d build-selective-scan