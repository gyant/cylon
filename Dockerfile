FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build

RUN apt update && \
  apt upgrade -y && \
  apt install -y \
  curl \
  protobuf-compiler \
  libprotobuf-dev

ENV PATH=$PATH:/root/.cargo/bin

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable

WORKDIR /app

COPY Cargo.toml ./Cargo.toml
COPY build.rs ./build.rs
COPY proto ./proto
COPY src ./src

ARG candle_feature=default
ARG cuda_compute_cap=89
ARG cargo_build_jobs=4

RUN if [ "$candle_feature" = "cuda" ] || [ "$candle_feature" = "cudnn" ]; then \
  CUDA_COMPUTE_CAP=$cuda_compute_cap \
  cargo build --features $candle_feature -j $cargo_build_jobs --workspace --release; \
  else \
  cargo build --features $candle_feature -j $cargo_build_jobs --workspace --release; \
  fi

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

COPY --from=build /app/target/release/cylon /app/cylon

CMD ["/app/cylon"]
