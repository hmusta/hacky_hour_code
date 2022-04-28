#!/bin/bash

make

MAXTHREADS=64
echo "Strong scaling"
for i in 1 2 4 8 16 32 64; do ./atomic_increment $i ${MAXTHREADS}00000; done | tee atomic_increment_strong.tsv
echo
echo "Weak scaling"
for i in 1 2 4 8 16 32 64; do ./atomic_increment $i ${i}00000; done | tee atomic_increment_weak.tsv

echo "Strong scaling"
for i in 1 2 4 8 16 32 64; do ./fixed_size_queue $i ${MAXTHREADS}00000; done | tee fixed_size_queue_strong.tsv
echo
echo "Weak scaling"
for i in 1 2 4 8 16 32 64; do ./fixed_size_queue $i ${i}00000; done | tee fixed_size_queue_weak.tsv
echo
