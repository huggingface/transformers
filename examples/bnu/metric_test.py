from datasets import list_metrics
from datasets import load_metric
from datasets import Metric
metrics_list = list_metrics()

print(len(metrics_list))
print(metrics_list)
metric = load_metric("glue", 'mrpc',cache_dir='./metric_dir')
