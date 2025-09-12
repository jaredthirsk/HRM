#!/bin/bash

# Monitor the ACT steps experiment progress
# Run this in a separate terminal to watch results as they come in

echo "Monitoring ACT Steps Experiment"
echo "==============================="
echo "Press Ctrl+C to stop monitoring"
echo ""

# Find the most recent results file
results_dir="results/act_steps_experiment"
latest_file=$(ls -t "$results_dir"/results_*.csv 2>/dev/null | head -1)

if [ -z "$latest_file" ]; then
    echo "No results file found yet. Waiting for experiment to start..."
    while [ -z "$latest_file" ]; do
        sleep 2
        latest_file=$(ls -t "$results_dir"/results_*.csv 2>/dev/null | head -1)
    done
fi

echo "Monitoring: $latest_file"
echo ""

# Watch the file for changes
watch -n 2 "echo 'ACT Steps Experiment Progress'; echo '============================'; cat '$latest_file' | column -t -s','; echo ''; echo 'File: $latest_file'; echo 'Updated: \$(date)'"