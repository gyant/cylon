build-cpu:
	cargo build --workspace

run-cpu:
	cargo run --bin cylon

build-mac:
	cargo build --features metal --workspace

run-mac:
	cargo run --bin cylon --features metal

build-docker-cpu:
	docker build --tag registry.gyant.internal/cylon:cpu \
	--build-arg build_base_image=ubuntu:24.04 \
	--build-arg runtime_base_image=ubuntu:24.04 \
	.

build-docker-cuda:
	docker build --tag registry.gyant.internal/cylon:cuda \
	--build-arg candle_feature=cuda \
	--build-arg cuda_compute_cap=89 \
	. 

build-docker-cuda-blackwell:
	docker build --tag registry.gyant.internal/cylon:cuda \
	--build-arg candle_feature=cuda \
	--build-arg cuda_compute_cap=90 \
	. 

build-docker-cudnn:
	docker build --tag registry.gyant.internal/cylon:cudnn \
	--build-arg candle_feature=cudnn,flash-attn \
	. 

build-docker-cudnn-blackwell:
	docker build --tag registry.gyant.internal/cylon:cudnn-blackwell \
	--build-arg candle_feature=cudnn,flash-attn \
	--build-arg cuda_compute_cap=90 \
	. 

run-docker-cpu:
	docker run --rm -it registry.gyant.internal/cylon:cpu /bin/bash

run-docker-cuda:
	docker run --rm -it --gpus all registry.gyant.internal/cylon:cuda /bin/bash

run-docker-cudnn:
	docker run --rm -it --gpus all registry.gyant.internal/cylon:cudnn /bin/bash

run-docker-metal:
	docker run --rm -it --gpus all registry.gyant.internal/cylon:metal /bin/bash
