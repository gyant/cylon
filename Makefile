build:
	cargo build --features metal --workspace

run:
	cargo run --features metal --workspace

build-docker:
	docker build --tag registry.gyant.internal/cylon:cpu .

build-docker-metal:
	docker build --tag registry.gyant.internal/cylon:metal --build-arg candle_feature=metal .

build-docker-cuda:
	docker build --tag registry.gyant.internal/cylon:cuda --build-arg candle_feature=cuda . 

build-docker-cudnn:
	docker build --tag registry.gyant.internal/cylon:cudnn --build-arg candle_feature=cudnn . 

build-docker-cudnn-blackwell:
	docker build --tag registry.gyant.internal/cylon:cudnn-blackwell --build-arg candle_feature=cudnn --build-arg cuda_compute_cap=90 . 

run-docker-cuda:
	docker run --rm --gpus all registry.gyant.internal/cylon:cuda /bin/bash

run-docker-cudnn:
	docker run --rm --gpus all registry.gyant.internal/cylon:cudnn /bin/bash

run-docker-metal:
	docker run --rm --gpus all registry.gyant.internal/cylon:metal /bin/bash

run-docker-cpu:
	docker run --rm --gpus all registry.gyant.internal/cylon:cpu /bin/bash

