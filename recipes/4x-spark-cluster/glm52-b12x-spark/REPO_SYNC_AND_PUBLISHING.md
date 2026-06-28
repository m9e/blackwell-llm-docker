# Repo sync and publishing notes

Date: 2026-06-27

## blackwell-llm-docker

Local path:

```text
/home/matt/code/blackwell-llm-docker
```

Upstream remote:

```text
https://github.com/local-inference-lab/blackwell-llm-docker.git
```

This repo was fetched and fast-forwarded to upstream `main` before adding the Spark GLM-5.2 recipe files.

Files intended for publication:

```text
recipes/4x-spark-cluster/glm52-b12x-spark/README.md
recipes/4x-spark-cluster/glm52-b12x-spark/COMMUNITY_HOWTO_GLM52_SPARK.md
recipes/4x-spark-cluster/glm52-b12x-spark/LOCALLAMA_POST.md
recipes/4x-spark-cluster/glm52-b12x-spark/NVIDIA_FORUM_POST.md
recipes/4x-spark-cluster/glm52-b12x-spark/MTP_ITERATIVE_DCP_DIAGNOSIS.md
recipes/4x-spark-cluster/glm52-b12x-spark/SOULKILLER_MEMORY_PRUNE_20260627.md
recipes/4x-spark-cluster/glm52-b12x-spark/*.env
recipes/4x-spark-cluster/glm52-b12x-spark/*.sh
vllm-spark-disable-dcp-mq-broadcaster.patch
```

Files intentionally excluded:

```text
logs/
corpus/
*.output.txt
__pycache__/
```

## vllm fork

Local path:

```text
/home/matt/code/vllm-dark-devotion
```

Upstream remote:

```text
https://github.com/local-inference-lab/vllm.git
```

Current working branch before publishing:

```text
codex/dark-devotion-release-20260622
```

Dirty files to publish as the Spark GLM-5.2 patch branch:

```text
vllm/distributed/parallel_state.py
vllm/model_executor/layers/sparse_attn_indexer.py
vllm/model_executor/models/deepseek_mtp.py
vllm/model_executor/models/glm4_moe_mtp.py
vllm/v1/attention/backends/mla/b12x_mla_sparse.py
vllm/v1/attention/backends/mla/indexer.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/sample/rejection_sampler.py
vllm/v1/spec_decode/llm_base_proposer.py
vllm/v1/spec_decode/step3p5.py
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu_worker.py
```

The public branch should be pushed to the `m9e/vllm` fork after committing.

## Other repos checked

```text
/home/matt/code/glm-5.2-sm120
/home/matt/code/sonusflow-spark-vllm-docker
/home/matt/code/spark-vllm-docker
```

`glm-5.2-sm120` and `sonusflow-spark-vllm-docker` were clean and up to date after fetch.

`spark-vllm-docker` is historical build-work context. It has local dirty files and is behind upstream. It should not be force-synced without first preserving those local build notes/changes.
