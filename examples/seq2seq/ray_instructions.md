### Instructions to use ray-tune

1) Install ray from source on your **local** machine

2) In `ray_tune_config.yaml` set
```yaml
file_mounts: {
    /home/ubuntu/transformers/: PATH_TO_YOUR_TRANSFORMERS
}
```

TODO: Finish these

```bash
export CFG=ray_tune_config.yaml
ray up $CFG

ray stop $CFG # when you are finished

```

### Modifications

[Doc](https://docs.ray.io/en/master/tune/tutorials/tune-distributed.html#pre-emptible-instances-cloud) for switching to GCP:
