# GLM-5.2 Spark production freeze

- Date: 2026-06-25
- Production point: TP4 / DCP4 / MTP1 / 128K / fp8 KV / B12X sparse MLA / flashinfer_cutlass MoE
- Known-good model name: glm52-mtp1-dcp4-128k
- Known-good capacity: 132,096 KV tokens, 1.01x concurrency at 131,072 max_model_len
- Known-good perf: 14.35 tok/s on 512-token run; best hot sample 15.18 tok/s; quality smoke passed
- MTP2 experiment: same capacity, 14.44 tok/s hot 512-token run, not better than MTP1

## Current /v1/models at freeze time
{"object":"list","data":[{"id":"glm52-mtp2-dcp4-128k","object":"model","created":1782427705,"owned_by":"vllm","root":"/models","parent":null,"max_model_len":131072,"permission":[{"id":"modelperm-a659805de0f87bd0","object":"model_permission","created":1782427705,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
## Docker containers
NAMES           IMAGE                                            STATUS       PORTS
glm-dark-head   glm-darkdevotion-b12x:20260625-arm64-mtp1-trim   Up 6 hours   

## Image inspect: glm-darkdevotion-b12x:20260625-arm64-mtp1-trim
[
    {
        "Id": "sha256:1a32d0280c931554388cc8566ba44884a7f9733e08efc694b92e924733b1b914",
        "RepoTags": [
            "glm-darkdevotion-b12x:20260625-arm64-mtp1-trim"
        ],
        "RepoDigests": [],
        "Comment": "buildkit.dockerfile.v0",
        "Created": "2026-06-25T05:43:25.830219667-06:00",
        "Config": {
            "Env": [
                "PATH=/workspace/vllm:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "NVARCH=sbsa",
                "NVIDIA_REQUIRE_CUDA=cuda>=13.2 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=580,driver<581 brand=grid,driver>=580,driver<581 brand=tesla,driver>=580,driver<581 brand=nvidia,driver>=580,driver<581 brand=quadro,driver>=580,driver<581 brand=quadrortx,driver>=580,driver<581 brand=nvidiartx,driver>=580,driver<581 brand=vapps,driver>=580,driver<581 brand=vpc,driver>=580,driver<581 brand=vcs,driver>=580,driver<581 brand=vws,driver>=580,driver<581 brand=cloudgaming,driver>=580,driver<581 brand=unknown,driver>=590,driver<591 brand=grid,driver>=590,driver<591 brand=tesla,driver>=590,driver<591 brand=nvidia,driver>=590,driver<591 brand=quadro,driver>=590,driver<591 brand=quadrortx,driver>=590,driver<591 brand=nvidiartx,driver>=590,driver<591 brand=vapps,driver>=590,driver<591 brand=vpc,driver>=590,driver<591 brand=vcs,driver>=590,driver<591 brand=vws,driver>=590,driver<591 brand=cloudgaming,driver>=590,driver<591",
                "NV_CUDA_CUDART_VERSION=13.2.51-1",
                "CUDA_VERSION=13.2.0",
                "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
                "NVIDIA_VISIBLE_DEVICES=all",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "NV_CUDA_LIB_VERSION=13.2.0-1",
                "NV_NVTX_VERSION=13.2.20-1",
                "NV_LIBNPP_VERSION=13.1.0.44-1",
                "NV_LIBNPP_PACKAGE=libnpp-13-2=13.1.0.44-1",
                "NV_LIBCUSPARSE_VERSION=12.7.9.17-1",
                "NV_LIBCUBLAS_PACKAGE_NAME=libcublas-13-2",
                "NV_LIBCUBLAS_VERSION=13.3.0.5-1",
                "NV_LIBCUBLAS_PACKAGE=libcublas-13-2=13.3.0.5-1",
                "NVIDIA_PRODUCT_NAME=CUDA",
                "NV_CUDA_CUDART_DEV_VERSION=13.2.51-1",
                "NV_NVML_DEV_VERSION=13.2.51-1",
                "NV_LIBCUSPARSE_DEV_VERSION=12.7.9.17-1",
                "NV_LIBNPP_DEV_VERSION=13.1.0.44-1",
                "NV_LIBNPP_DEV_PACKAGE=libnpp-dev-13-2=13.1.0.44-1",
                "NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-13-2",
                "NV_LIBCUBLAS_DEV_VERSION=13.3.0.5-1",
                "NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-13-2=13.3.0.5-1",
                "NV_CUDA_NSIGHT_COMPUTE_VERSION=13.2.0-1",
                "NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-13-2=13.2.0-1",
                "LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
                "MAX_JOBS=8",
                "CMAKE_BUILD_PARALLEL_LEVEL=8",
                "NINJAFLAGS=-j8",
                "MAKEFLAGS=-j8",
                "DG_JIT_USE_NVRTC=1",
                "USE_CUDNN=1",
                "DEBIAN_FRONTEND=noninteractive",
                "PIP_BREAK_SYSTEM_PACKAGES=1",
                "VLLM_BASE_DIR=/workspace/vllm",
                "PIP_CACHE_DIR=/root/.cache/pip",
                "UV_CACHE_DIR=/root/.cache/uv",
                "UV_SYSTEM_PYTHON=1",
                "UV_BREAK_SYSTEM_PACKAGES=1",
                "UV_LINK_MODE=copy",
                "TORCH_CUDA_ARCH_LIST=12.1a",
                "FLASHINFER_CUDA_ARCH_LIST=12.1a",
                "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
                "TIKTOKEN_ENCODINGS_BASE=/workspace/vllm/tiktoken_encodings"
            ],
            "Entrypoint": [
                "/opt/nvidia/nvidia_entrypoint.sh"
            ],
            "WorkingDir": "/workspace/vllm",
            "Labels": {
                "local.kamiwaza.glm52.spark.base_image": "glm-darkdevotion-b12x:20260624-arm64-mtpfix6",
                "local.kamiwaza.glm52.spark.overlay": "kv-diagnostics-and-post-load-trim",
                "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
                "org.opencontainers.image.ref.name": "ubuntu",
                "org.opencontainers.image.version": "24.04"
            }
        },
        "Architecture": "arm64",
        "Os": "linux",
        "Size": 19900217514,
        "GraphDriver": {
            "Data": {
                "LowerDir": "/var/lib/docker/overlay2/gzp4sytiwzurduyixjfh1vc0k/diff:/var/lib/docker/overlay2/f6e6xo5y7xz3moqdkje6cm8zb/diff:/var/lib/docker/overlay2/sv6hhekdbll6ukiwgd8fhtrex/diff:/var/lib/docker/overlay2/qply5198a4lqewczdw1688qut/diff:/var/lib/docker/overlay2/30ma5ez16v677d3d2d0reytsu/diff:/var/lib/docker/overlay2/lc2b7gmb7di6kitniobbv91e6/diff:/var/lib/docker/overlay2/vut94l87xhdt2ex8j7qld1qwp/diff:/var/lib/docker/overlay2/o8hxfl69n5bzjtblc1qjmnpom/diff:/var/lib/docker/overlay2/sahzircofcwuw11cnczwapy4e/diff:/var/lib/docker/overlay2/vw3g59qw8dfjfoojs2r8e3m7e/diff:/var/lib/docker/overlay2/5xjani6op87uc6tze6u4i5ff9/diff:/var/lib/docker/overlay2/wenmhftxg636ib3io2usipvl4/diff:/var/lib/docker/overlay2/0pyfq8da6wjn1zfrowf52t6hg/diff:/var/lib/docker/overlay2/ydxuegffjyu4okpnoqqa4pgq9/diff:/var/lib/docker/overlay2/ow2ctob385mpfpxjl5tec532g/diff:/var/lib/docker/overlay2/eikp5erzijmee7d8phk93k0iw/diff:/var/lib/docker/overlay2/i5lrfwjfeeyyt95efqnbwi2p0/diff:/var/lib/docker/overlay2/ndfbxvvatxygjr4ztvme240wo/diff:/var/lib/docker/overlay2/jker9hybsgdqe1popmjpqmsq7/diff:/var/lib/docker/overlay2/792731e7122049de70306ce42f510e2731858cfc980744ee8644f22a5992e80c/diff:/var/lib/docker/overlay2/3a4db038e1c344a159c2d5613197013983ceaa965ea52132a9d24b4a7734193d/diff:/var/lib/docker/overlay2/5e4d10431e1ea2bd01b32964394a0b53cdf9b530fea1fbfff6b39bddbfcf2ba9/diff:/var/lib/docker/overlay2/4e4c3cba143f3e94bc889ce00a7fd1fe2c70411a884e7637ace4c14f29950426/diff:/var/lib/docker/overlay2/db05501cee3b9b0bc57136d21dbb5b7f1d220eb376161a80d02beab3a4a13964/diff:/var/lib/docker/overlay2/62867e012f9450e6746be5f017a86cdace9defe6f9515b6380bafb4a34da76dc/diff:/var/lib/docker/overlay2/8716d5ee6196f67e0d772d977a8c237589cdd79f02e04c17c852274be9cab813/diff:/var/lib/docker/overlay2/70ee0514570897594e1813896f2d60f731e9780f45ee655fc9dfd12de40248db/diff:/var/lib/docker/overlay2/7b744b335d2bbdb49494d630d3540ef7f82a085d34061c303d9b7ad06326baa0/diff:/var/lib/docker/overlay2/822e2cb0bd31267137796e41c8d0a1d1b7efcfcb4ba001766f415d39ba86877d/diff:/var/lib/docker/overlay2/f649df9d7dc81e2e61dbd76495fcb7758499aade27e3335b18ec076106cad23f/diff:/var/lib/docker/overlay2/3fb7324ea1772505bf3b9b595dbbea97f99801e819c4f70844cda7983ec967a8/diff:/var/lib/docker/overlay2/0efa2e3823606e16c5c6e974db143e286f715befe51587610def75d2c1cd9aaa/diff",
                "MergedDir": "/var/lib/docker/overlay2/h28p2eggktq29nte2j1ckgf4k/merged",
                "UpperDir": "/var/lib/docker/overlay2/h28p2eggktq29nte2j1ckgf4k/diff",
                "WorkDir": "/var/lib/docker/overlay2/h28p2eggktq29nte2j1ckgf4k/work"
            },
            "Name": "overlay2"
        },
        "RootFS": {
            "Type": "layers",
            "Layers": [
                "sha256:e5dae71ade4390c09123a86ada6c9bc64ac469d0495acae5b2216a627395050c",
                "sha256:a0aed9a26128a7a4c9b13b8c171b453f1a00808cb32e2075d6ba7675eac56913",
                "sha256:539926fe12daa7de499df42e20a693fca1c3a4db1e61a1ce25a32df3e8041d3e",
                "sha256:eb9ffcd8102a913d1344d73a2e8dae763707b7977dac130cdc028a4ca8d6959a",
                "sha256:26c4202e143a608a1a8358a67268d5098a00ff5b866ecfcd3d3b397480f48972",
                "sha256:ed65183a4b4095c6e03693f7394075fe244194c935d3bb20f1370a7a3669e7eb",
                "sha256:53f586ec0996b4fd2f18b6eb9f92a8c4137149bf8e71dd8a0ce47042355b12e5",
                "sha256:e67c3dff55b16af41294d759c61580b6ccb77a587737c56b9958240db43b0a84",
                "sha256:4cf29c1b59083414e65a3ebcab217fd108690d353034d075cc128928d4649119",
                "sha256:59a0bbf0604b6056f47220d7c9046ded5639b9e2a2b2fb3e5a635f17253cb4d0",
                "sha256:557d105fc8723855f95ece306f0c82ae021f9ee8cb67951d1962abd5d19985e9",
                "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef",
                "sha256:84fbad02f97c11c71c7aa671fb061bfbc1d7722284da054b2d52cc9c881fa88b",
                "sha256:98a709a135efec75ca02a1afdf7228088fe37be9f9ba95f50452b4fa8d7f7b91",
                "sha256:e3d59e7cbefe8b3dfe81457d0ccd1ad6f793955afb8d0b2fe9a9f4bad48e4ac2",
                "sha256:426c72c1c7f2da1e8d2a9d312e8245a74574e56683318ed5089e17c1f0ab968e",
                "sha256:481c0002ac15604137216900f3567a2eb4a70fd34bc53bdbfafcc5225be17f0f",
                "sha256:21ca0af270bb4dbdcd44e1d1bf116de914267aae684861588f957527cff25fff",
                "sha256:a50148a8692190d023051854d2666ad763611fec8dd2d66c0e9ff1c02fc044f7",
                "sha256:a4bed254e3651b8590fc8fcc2d25c6fe188ccd8d927dce7f591fd29ec3d2c913",
                "sha256:c125b489b286ef0eca5649101abb9f2c2891de28549f0b8a819d1b39ec008868",
                "sha256:87c1dc6c3cdc79e0989f04411c0b1521730be8cb3edf6aec32e83df4d8463647",
                "sha256:997b5072de8f8f9cc12ad567ce4fa531cc2dec861b3556d62d0a106377227fa7",
                "sha256:68fa64fd5c62975be9622a298c22eb9acd33bdd3a0aba7ebaac8457481b0205d",
                "sha256:c8e73a471eb502b61834d09267588c50c6cdd3f457e9dbd2506ab2360213c1c0",
                "sha256:972ef1c94156a638b8ddd7dd4f4e910f08170dba53449c2c4273c4f1ce45d988",
                "sha256:5a320a714ebe95267f22aa23c3ed8b34ee95cbbedf72878aa0c9b2a375d862a2",
                "sha256:320bb232ead70588a248f00bc8c34e889cdef8bef90f7a09ae521293a9871171",
                "sha256:61bb07ff0359c1974db6a01672078617492284ece2a18cce17e2704ec120051f",
                "sha256:febf5f69ee21993714a37b4e68b76626beda3890b57b3b78732a7039f02b8811",
                "sha256:340440f8729e4ac77d2bbf2a0e82192cc6622eb8ef82e283f6da75f3d1d852f8",
                "sha256:3be54b0b09bfc2b8ffc6f741c7c337c9526b60db9dd9555b5bb1a5a2abfad2e7",
                "sha256:617520c2b479715028025aeef9e9838cc49a5488a57dc4f7aef476ee10b89b4b"
            ]
        },
        "Metadata": {
            "LastTagTime": "2026-06-25T05:44:19.141255169-06:00"
        }
    }
]
