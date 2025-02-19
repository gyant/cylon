build:
	cargo build --features metal

run:
	cargo run --features metal

build-docker:
	docker build --tag registry.gyant.internal/cylon:cpu .

build-docker-metal:
	docker build --tag registry.gyant.internal/cylon:metal --build-arg candle_feature=metal .

build-docker-cuda:
	docker build --tag registry.gyant.internal/cylon:cuda --build-arg candle_feature=cuda . 

build-docker-cudnn:
	docker build --tag registry.gyant.internal/cylon:cudnn --build-arg candle_feature=cudnn . 

run-docker-cuda:
	docker run --rm --gpus all registry.gyant.internal/cylon:cuda

run-docker-cudnn:
	docker run --rm --gpus all registry.gyant.internal/cylon:cudnn

