# التدريب الفعال على معالجات متعددة

عندما يكون التدريب على معالج واحد بطيئًا جدًا، يمكننا استخدام معالجات متعددة. يركز هذا الدليل على تمكين PyTorch-based DDP من التدريب الموزع على المعالج المركزي بكفاءة على [الحديد العاري](#usage-in-trainer) و [Kubernetes](#usage-with-kubernetes).

## Intel® oneCCL Bindings لـ PyTorch

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (مكتبة الاتصالات الجماعية) هي مكتبة للتدريب الموزع على التعلم العميق بكفاءة، وتنفذ عمليات جماعية مثل allreduce و allgather و alltoall. لمزيد من المعلومات حول oneCCL، يرجى الرجوع إلى [وثائق oneCCL](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) و [مواصفات oneCCL](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html).

تنفذ الوحدة النمطية `oneccl_bindings_for_pytorch` (`torch_ccl` قبل الإصدار 1.12) واجهة برمجة تطبيقات مجموعة العمليات C10D الخاصة بـ PyTorch ويمكن تحميلها ديناميكيًا كمجموعة عمليات خارجية ولا تعمل إلا على منصة Linux الآن

تحقق من مزيد من المعلومات التفصيلية لـ [oneccl_bind_pt](https://github.com/intel/torch-ccl).

### تثبيت Intel® oneCCL Bindings لـ PyTorch

تتوفر ملفات العجلة لإصدارات Python التالية:

| إصدار الملحق | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| 2.1.0             |            | √          | √          | √          | √           |
| 2.0.0             |            | √          | √          | √          | √           |
| 1.13.0            |            | √          | √          | √          | √           |
| 1.12.100          |            | √          | √          | √          | √           |
| 1.12.0            |            | √          | √          | √          | √           |

يرجى تشغيل `pip list | grep torch` للحصول على إصدار PyTorch الخاص بك.
```bash
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
حيث يجب أن يكون `{pytorch_version}` إصدار PyTorch الخاص بك، على سبيل المثال 2.1.0.
تحقق من المزيد من النهج لتثبيت [oneccl_bind_pt](https://github.com/intel/torch-ccl).
يجب أن تتطابق إصدارات oneCCL و PyTorch.

<Tip warning={true}>

oneccl_bindings_for_pytorch 1.12.0 عجلة مسبقة البناء لا تعمل مع PyTorch 1.12.1 (فهي لـ PyTorch 1.12.0)
يجب أن يعمل PyTorch 1.12.1 مع oneccl_bindings_for_pytorch 1.12.100

</Tip>

## مكتبة Intel® MPI
استخدم هذا التنفيذ MPI القائم على المعايير لتقديم رسائل مجموعات مرنة وفعالة وقابلة للتطوير على بنية Intel®. هذا المكون هو جزء من Intel® oneAPI HPC Toolkit.

يتم تثبيت oneccl_bindings_for_pytorch جنبًا إلى جنب مع مجموعة أدوات MPI. يجب تحديد مصدر البيئة قبل استخدامه.

لـ Intel® oneCCL >= 1.12.0
```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

لـ Intel® oneCCL الذي يقل إصدارها عن 1.12.0
```bash
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### تثبيت Intel® Extension لـ PyTorch
#### تثبيت Intel® Extension لـ PyTorch

يوفر Intel Extension لـ PyTorch (IPEX) تحسينات أداء لتدريب المعالج باستخدام كل من Float32 و BFloat16 (راجع قسم [معالج واحد](./perf_train_cpu) لمعرفة المزيد).


يأخذ الاستخدام التالي في Trainer "مثالًا على استخدام mpirun في مكتبة Intel® MPI.


## الاستخدام في Trainer
لتمكين التدريب الموزع متعدد المعالجات في Trainer مع backend ccl، يجب على المستخدمين إضافة **--ddp_backend ccl** في وسيطات الأمر.

دعونا نرى مثالاً مع [مثال الإجابة على الأسئلة](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)


يُمكّن الأمر التالي التدريب باستخدام عمليتين على عقدة Xeon واحدة، مع تشغيل عملية واحدة لكل مقبس. يمكن ضبط متغيرات OMP_NUM_THREADS/CCL_WORKER_COUNT للحصول على الأداء الأمثل.
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```
يُمكّن الأمر التالي التدريب باستخدام ما مجموعه أربع عمليات على اثنين من معالجات Xeon (node0 و node1، مع أخذ node0 كعملية رئيسية)، يتم تعيين ppn (العمليات لكل عقدة) إلى 2، مع تشغيل عملية واحدة لكل مقبس. يمكن ضبط متغيرات OMP_NUM_THREADS/CCL_WORKER_COUNT للحصول على الأداء الأمثل.

في node0، تحتاج إلى إنشاء ملف تكوين يحتوي على عناوين IP لكل عقدة (على سبيل المثال hostfile) وتمرير مسار ملف التكوين كوسيط.
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
الآن، قم بتشغيل الأمر التالي في node0 وسيتم تمكين **4DDP** في node0 و node1 مع دقة BF16 المختلطة تلقائيًا:
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```

## الاستخدام مع Kubernetes

يمكن نشر نفس مهمة التدريب الموزع من القسم السابق في مجموعة Kubernetes باستخدام
[مشغل Kubeflow PyTorchJob للتدريب](https://www.kubeflow.org/docs/components/training/pytorch/).

### الإعداد

يفترض هذا المثال أن لديك ما يلي:
* الوصول إلى مجموعة Kubernetes مع [Kubeflow المثبتة](https://www.kubeflow.org/docs/started/installing-kubeflow/)
* [`kubectl`](https://kubernetes.io/docs/tasks/tools/) مثبتة ومُهيأة للوصول إلى مجموعة Kubernetes
* [مطالبة حجم ثابتة (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) يمكن استخدامها
  لتخزين مجموعات البيانات وملفات النماذج. هناك خيارات متعددة لإعداد PVC بما في ذلك استخدام فئة تخزين NFS
  أو دلو تخزين سحابي.
* حاوية Docker تحتوي على نص برمجة التدريب الخاص بك وجميع التبعيات اللازمة لتشغيل النص. بالنسبة
  إلى وظائف التدريب الموزع على المعالج المركزي، يتضمن ذلك عادةً PyTorch وTransformers وIntel Extension لـ PyTorch وIntel
  oneCCL Bindings لـ PyTorch وOpenSSH للتواصل بين الحاويات.

الجزء التالي هو مثال على Dockerfile يستخدم صورة أساسية تدعم التدريب الموزع على المعالج المركزي ثم
يستخرج إصدار Transformers إلى دليل `/workspace`، بحيث يتم تضمين نصوص الأمثلة في الصورة:
```dockerfile
FROM intel/ai-workflows:torch-2.0.1-huggingface-multinode-py3.9

WORKDIR /workspace
WORKDIR /workspace

# قم بتنزيل واستخراج رمز المحولات
ARG HF_TRANSFORMERS_VER="4.35.2"
RUN mkdir transformers && \
    curl -sSL --retry 5 https://github.com/huggingface/transformers/archive/refs/tags/v${HF_TRANSFORMERS_VER}.tar.gz | tar -C transformers --strip-components=1 -xzf -
```
يجب بناء الصورة ونسخها إلى عقد المجموعة أو دفعها إلى سجل الحاويات قبل نشر PyTorchJob في المجموعة.

### ملف مواصفات PyTorchJob

يتم استخدام [Kubeflow PyTorchJob](https://www.kubeflow.org/docs/components/training/pytorch/) لتشغيل مهمة التدريب الموزع على المجموعة. يحدد ملف yaml لـ PyTorchJob معلمات مثل:
 * اسم PyTorchJob
 * عدد النسخ المتماثلة (العاملين)
 * نص Python ووسائطه التي سيتم استخدامها لتشغيل مهمة التدريب
 * أنواع الموارد (محدد العقدة وذاكرة الوصول العشوائي ووحدة المعالجة المركزية) اللازمة لكل عامل
 * الصورة/العلامة لحاوية Docker لاستخدامها
 * متغيرات البيئة
 * تركيب حجم ل PVC

يحدد تركيب الحجم مسارًا حيث سيتم تركيب PVC في الحاوية لكل بود عامل. يمكن استخدام هذا الموقع لمجموعة البيانات وملفات نقاط التفتيش والنماذج المحفوظة بعد اكتمال التدريب.
بالتأكيد، سأتبع تعليماتك وسأبدأ الترجمة من بعد التعليق الأول:

تعتبر القطعة أدناه مثالاً على ملف yaml لوظيفة PyTorchJob مع 4 عمال يقومون بتشغيل
[مثال الإجابة على السؤال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).
```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: transformers-pytorchjob
  namespace: kubeflow
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
    maxRestarts: 10
  pytorchReplicaSpecs:
    Worker:
      replicas: 4  # عدد وحدات worker
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <image name>:<tag>  # حدد صورة Docker التي سيتم استخدامها لوحدات worker
              imagePullPolicy: IfNotPresent
              command:
                - torchrun
                - /workspace/transformers/examples/pytorch/question-answering/run_qa.py
                - --model_name_or_path
                - "google-bert/bert-large-uncased"
                - --dataset_name
                - "squad"
                - --do_train
                - --do_eval
                - --per_device_train_batch_size
                - "12"
                - --learning_rate
                - "3e-5"
                - --num_train_epochs
                - "2"
                - --max_seq_length
                - "384"
                - --doc_stride
                - "128"
                - --output_dir
                - "/tmp/pvc-mount/output"
                - --no_cuda
                - --ddp_backend
                - "ccl"
                - --use_ipex
                - --bf16  # حدد --bf16 إذا كان عتادك يدعم bfloat16
              env:
              - name: LD_PRELOAD
                value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
              - name: TRANSFORMERS_CACHE
                value: "/tmp/pvc-mount/transformers_cache"
              - name: HF_DATASETS_CACHE
                value: "/tmp/pvc-mount/hf_datasets_cache"
              - name: LOGLEVEL
                value: "INFO"
              - name: CCL_WORKER_COUNT
                value: "1"
              - name: OMP_NUM_THREADS  # يمكن ضبطه للحصول على الأداء الأمثل
                value: "56"
              resources:
                limits:
                  cpu: 200  # قم بتحديث حدود/طلبات CPU والذاكرة بناءً على عقدك
                  memory: 128Gi
                requests:
                  cpu: 200  # قم بتحديث طلبات CPU والذاكرة بناءً على عقدك
                  memory: 128Gi
              volumeMounts:
              - name: pvc-volume
                mountPath: /tmp/pvc-mount
              - mountPath: /dev/shm
                name: dshm
          restartPolicy: Never
          nodeSelector:  # استخدم بشكل اختياري NodeSelector لتحديد أنواع العقد التي سيتم استخدامها لعمال
            node-type: spr
          volumes:
          - name: pvc-volume
            persistentVolumeClaim:
              claimName: transformers-pvc
          - name: dshm
            emptyDir:
              medium: Memory
```
لتشغيل هذا المثال، قم بتحديث yaml بناءً على نص البرنامج النصي للتدريب والعقد في عنقودك.

<Tip>
<Tip>

تُعرَّف حدود/طلبات موارد وحدة المعالجة المركزية (CPU) في ملف yaml بوحدات [وحدة المعالجة المركزية](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu)
حيث تعادل وحدة معالجة مركزية واحدة وحدة معالجة مركزية فعلية واحدة أو نواة افتراضية واحدة (اعتمادًا على ما إذا كان العقدة مضيفًا فعليًا
أو آلة افتراضية). يجب أن يكون مقدار حدود/طلبات وحدة المعالجة المركزية (CPU) والذاكرة المحددة في ملف yaml أقل من مقدار
سعة وحدة المعالجة المركزية/الذاكرة المتوفرة على جهاز واحد. عادة ما يكون من الجيد عدم استخدام السعة الكاملة للجهاز
من أجل ترك بعض الموارد لـ Kubelet وOS. من أجل الحصول على خدمة ["مضمونة"](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/#guaranteed)
[جودة الخدمة](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/) لوحدات العامل، قم بتعيين نفس
مقدار وحدة المعالجة المركزية والذاكرة لكل من حدود الموارد والطلبات.

</Tip>

### النشر

بعد تحديث مواصفات PyTorchJob بالقيم المناسبة لعنقودك ووظيفة التدريب، يمكن نشرها
في العنقود باستخدام:
```bash
kubectl create -f pytorchjob.yaml
```

يمكن بعد ذلك استخدام الأمر `kubectl get pods -n kubeflow` لعرض قائمة الوحدات في مساحة الاسم `kubeflow`. يجب أن ترى
وحدات العامل لوظيفة PyTorchJob التي تم نشرها للتو. في البداية، من المحتمل أن يكون لها حالة "Pending" أثناء
سحب الحاويات وإنشائها، ثم يجب أن تتغير الحالة إلى "Running".
```
NAME                                                     READY   STATUS                  RESTARTS          AGE
...
transformers-pytorchjob-worker-0                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-1                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-2                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-3                         1/1     Running                 0                 7m37s
...
```

يمكن عرض سجلات العامل باستخدام `kubectl logs -n kubeflow <اسم الوحدة>`. أضف `-f` لدفق السجلات، على سبيل المثال:
```bash
kubectl logs -n kubeflow transformers-pytorchjob-worker-0 -f
```

بعد اكتمال مهمة التدريب، يمكن نسخ النموذج المدرب من موقع التخزين أو وحدة التخزين المؤقتة للقرص الظاهري. عندما تنتهي
من الوظيفة، يمكن حذف مورد PyTorchJob من العنقود باستخدام `kubectl delete -f pytorchjob.yaml`.

## ملخص

غطى هذا الدليل تشغيل وظائف تدريب PyTorch الموزعة باستخدام وحدات معالجة مركزية متعددة على الخوادم العارية وعلى عنقود Kubernetes.
تستخدم كلتا الحالتين Intel Extension لـ PyTorch وIntel oneCCL Bindings لـ PyTorch لتحقيق الأداء الأمثل للتدريب، ويمكن استخدامها
كقالب لتشغيل حمل العمل الخاص بك على عدة عقد.