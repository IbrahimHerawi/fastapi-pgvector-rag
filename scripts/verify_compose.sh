#!/usr/bin/env sh
set -eu

echo "==> Validating docker-compose.yml"
docker compose config > /dev/null

echo "==> Building images"
docker compose build

echo "Compose build verification passed."
