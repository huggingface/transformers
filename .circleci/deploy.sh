cd docs

function deploy_doc(){
	echo "Creating doc at commit $1 and pushing to folder $2"
	git checkout $1
	if [ ! -z "$2" ]
	then
		if [ -d "$dir/$2" ]; then
			echo "Directory" $2 "already exists"
		else
			echo "Pushing version" $2
			make clean && make html && scp -r -oStrictHostKeyChecking=no _build/html $doc:$dir/$2
		fi
	else
		echo "Pushing master"
		make clean && make html && scp -r -oStrictHostKeyChecking=no _build/html/* $doc:$dir
	fi
}

deploy_doc "master"
deploy_doc "b33a385" v1.0.0
deploy_doc "fe02e45" v1.1.0
deploy_doc "89fd345" v1.2.0
deploy_doc "fc9faa8" v2.0.0
deploy_doc "3ddce1d" v2.1.1
deploy_doc "3616209" v2.2.0
deploy_doc "d0f8b9a" v2.3.0
deploy_doc "6664ea9" v2.4.0