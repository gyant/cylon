FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS build

RUN apt update && \
  apt upgrade -y && \
  apt install -y curl

ENV PATH=$PATH:/root/.cargo/bin

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable

WORKDIR /app

COPY Cargo.toml ./Cargo.toml
COPY src ./src

ARG candle_feature=default
ARG cuda_compute_cap=89

RUN if [ "$candle_feature" = "cuda" ] || [ "$candle_feature" = "cudnn" ]; then \
  CUDA_COMPUTE_CAP=$cuda_compute_cap cargo build --features $candle_feature --release; \
  else \
  cargo build --features $candle_feature --release; \
  fi

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

COPY --from=build /app/target/release/cylon /app/cylon

CMD ["/app/cylon"]
