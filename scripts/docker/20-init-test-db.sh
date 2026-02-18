#!/bin/sh
set -eu

postgres_user="${POSTGRES_USER:-postgres}"
test_db="${POSTGRES_TEST_DB:-rag_test}"

exists="$(
  psql -U "$postgres_user" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$test_db'"
)"

if [ "$exists" = "1" ]; then
  echo "Database '$test_db' already exists; skipping."
  exit 0
fi

psql -U "$postgres_user" -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"$test_db\";"
echo "Database '$test_db' created."
