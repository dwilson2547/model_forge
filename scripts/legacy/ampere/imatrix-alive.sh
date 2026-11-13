#!/bin/bash
stat -c '%y' ~/m0/baseline-27b/imatrix-27b-bf16.dat | cut -c12-19
p=$(pgrep -x llama-imatrix)
u1=$(awk '{print $14}' /proc/$p/stat)
sleep 40
u2=$(awk '{print $14}' /proc/$p/stat)
echo "utime delta over 40s: $((u2-u1)) jiffies (full-rate ref ~5120)"
