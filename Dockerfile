ARG build_base_image=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG runtime_base_image=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

FROM $build_base_image AS build

RUN apt update && \
  apt upgrade -y && \
  apt install -y \
  curl \
  protobuf-compiler \
  libprotobuf-dev \
  build-essential \
  pkg-config \
  libssl-dev

ENV PATH=$PATH:/root/.cargo/bin

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable

WORKDIR /app

COPY . .

ARG candle_feature=default
ARG cuda_compute_cap=89
ARG cargo_build_jobs=4

RUN case "$candle_feature" in \
  *cuda* | *cudnn* ) \
    CUDA_COMPUTE_CAP=$cuda_compute_cap \
    cargo build --features $candle_feature -j $cargo_build_jobs --workspace --release;; \
  * ) \
    cargo build --features $candle_feature -j $cargo_build_jobs --workspace --release;; \
  esac

FROM $runtime_base_image

WORKDIR /app

COPY --from=build /app/target/release/cylon /app/cylon

EXPOSE 8080

ENV CYLON_LISTEN_ADDRESS=0.0.0.0

CMD ["/app/cylon"]
