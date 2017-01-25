#!/bin/bash
# 1st parameter - player

for i in $(seq 1 5); do
  echo benchmarks for $1

  echo running 50
  ./run_ai -m test -n 50 -1 $1 -2 $1 >> tests/data-$1;

  echo running 100
  ./run_ai -m test -n 100 -1 $1 -2 $1 >> tests/data-$1;

  for i in $(seq 200 200 1000); do
    echo running $i;
    ./run_ai -m test -n $i -1 $1 -2 $1 >> tests/data-$1;
  done;

  for i in $(seq 2000 2000 10000); do
    echo running $i;
    ./run_ai -m test -n $i -1 $1 -2 $1 >> tests/data-$1;
  done;

  for i in $(seq 20000 20000 100000); do
    echo running $i;
    ./run_ai -m test -n $i -1 $1 -2 $1 >> tests/data-$1;
  done;

  echo running 200000
  ./run_ai -m test -n 200000 -1 $1 -2 $1 >> tests/data-$1;
done;

