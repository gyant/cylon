#!/bin/bash
for i in {1..5}; do
    grpcurl -plaintext -import-path ./proto -proto cylon.proto -d '{"prompt": "In one word, what color is the sky?"}' '127.0.0.1:8080' cylon.Agent/RunInference &
done
wait
