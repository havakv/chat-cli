#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"


version="$(cat .python-version | rg -o '[0-9]*\.[0-9]*\.[0-9]*' | sed 's/\.[0-9]*$//')"
echo "python$version"
target_dir=target/release
mkdir -p "$target_dir"

# build the executable zipp app
rye run shiv . -p "/usr/bin/env python$version" -c main -o "$target_dir/chat-cli"


