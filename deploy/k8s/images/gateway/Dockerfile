# syntax=docker/dockerfile:1
# Gateway image for the k3d phase: the same Rust binary that runs natively,
# built for linux/arm64 inside the container (Metal stays OUT of containers —
# only the control plane is kubernetes-native).
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY apps ./apps
RUN cargo build --release -p gateway

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/gateway /usr/local/bin/gateway
ENV GATEWAY_ADDR=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/gateway"]
