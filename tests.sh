#!/bin/bash
# 1st parameter - player

mkdir -p tests

function run_test {
    bin/release/run_ai -m playout_test -n $1 -1 $2 -2 $2 >> tests/data-$2
}

for i in $(seq 1 5); do
  echo benchmarks for $1

  echo running 50
  run_test 50 $1

  echo running 100
  run_test 100 $1

  for i in $(seq 200 200 1000); do
    echo running $i
    run_test $i $1
  done;

  for i in $(seq 2000 2000 10000); do
    echo running $i
    run_test $i $1
  done;

  for i in $(seq 20000 20000 100000); do
    echo running $i
    run_test $i $1
  done;

  echo running 200000
  run_test 200000 $1
done;

