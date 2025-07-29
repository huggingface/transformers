# create a patch but exclude the _scripts directory

BASE_REF=main-fork-44ee5d167

cd /root/repos/cohere-transformers-lint
git diff --binary $BASE_REF...HEAD -- . ':(exclude)_scripts' > _scripts/my_changes-$BASE_REF.patch
cd /root/repos/new-model-addition-cohere-july-2025
git apply --index --whitespace=nowarn ../cohere-transformers-lint/_scripts/my_changes.patch


