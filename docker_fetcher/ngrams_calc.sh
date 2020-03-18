#!/usr/bin/env bash

((n=${1:-0})) || exit 1

declare -A ngrams

while read -ra line; do
        for ((i = 0; i < ${#line[@]}; i++)); do
                ((ngrams[${line[@]:i:n}]++))
        done
done

for i in "${!ngrams[@]}"; do
        printf '%d\t%s\n' "${ngrams[$i]}" "$i"
done
