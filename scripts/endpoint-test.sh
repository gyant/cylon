#!/bin/bash
for i in {1..5}; do
    grpcurl -plaintext -import-path ./proto -proto cylon.proto -d '{"messages": [{"role": "user", "content": "In one word, what color is the sky?"}]}' '127.0.0.1:8080' cylon.CylonApi/InferenceRun &
done
wait
