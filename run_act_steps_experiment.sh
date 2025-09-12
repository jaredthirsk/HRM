#!/bin/bash

# ACT Steps Experiment for 9x9 Sudoku - Fixed version
# Tests how model performance varies with different maximum ACT steps

echo "Starting ACT Steps Experiment (Fixed)"
echo "======================================"
echo "Model: aspiring-parakeet (step_12221)"
echo "Dataset: sudoku-extreme-1k-aug-1000"
echo ""

# Create results directory
mkdir -p results/act_steps_experiment
results_csv="results/act_steps_experiment/results_$(date +%Y%m%d_%H%M%S).csv"
temp_output="/tmp/act_eval_output.txt"

echo "Steps,Accuracy,ExactAccuracy,Loss,ActualSteps,Timestamp" > "$results_csv"
echo "Results will be saved to: $results_csv"
echo ""

for steps in 1 2 4 8 16 32 64
do
    echo "================================================"
    echo "Testing with halt_max_steps=${steps}..."
    
    # Update config with new step count
    export steps="${steps}"
    envsubst < "checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 aspiring-parakeet/all_config.yaml.tmpl" > "checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 aspiring-parakeet/all_config.yaml"
    
    # Run evaluation with DISABLE_COMPILE and save full output
    echo "  Running evaluation..."
    DISABLE_COMPILE=1 python3 evaluate.py \
        checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 aspiring-parakeet/step_12221" \
        data_path=data/sudoku-extreme-1k-aug-1000 > "$temp_output" 2>&1
    
    # Check if evaluation succeeded
    if [ $? -ne 0 ]; then
        echo "  ERROR: Evaluation failed for steps=${steps}"
        tail -10 "$temp_output"
        continue
    fi
    
    # Extract metrics from output
    # Looking for lines like: "all {'accuracy': 0.633, 'exact_accuracy': 0.005, 'lm_loss': 0.829, 'steps': 8.0}"
    accuracy=$(grep "^all {" "$temp_output" | grep -oP "'accuracy': \K[0-9.]+")
    exact_acc=$(grep "^all {" "$temp_output" | grep -oP "'exact_accuracy': \K[0-9.]+")
    loss=$(grep "^all {" "$temp_output" | grep -oP "'lm_loss': \K[0-9.]+")
    actual_steps=$(grep "^all {" "$temp_output" | grep -oP "'steps': \K[0-9.]+")
    
    # Default values if extraction fails
    accuracy=${accuracy:-"0.0"}
    exact_acc=${exact_acc:-"0.0"}
    loss=${loss:-"999"}
    actual_steps=${actual_steps:-"$steps"}
    
    # Save results immediately with timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${steps},${accuracy},${exact_acc},${loss},${actual_steps},${timestamp}" >> "$results_csv"
    
    # Display current result
    echo "  Max Steps: ${steps}"
    echo "  Accuracy: ${accuracy}"
    echo "  Exact Accuracy: ${exact_acc}"
    echo "  Loss: ${loss}"
    echo "  Actual Steps Used: ${actual_steps}"
    echo "  Saved at: ${timestamp}"
    echo ""
    
    # Show all results so far
    echo "Results so far:"
    echo "---------------"
    cat "$results_csv" | column -t -s','
    echo ""
done

echo "======================================"
echo "Experiment complete!"
echo "Results saved to: $results_csv"
echo ""
echo "Final Summary:"
cat "$results_csv" | column -t -s','

# Clean up
rm -f "$temp_output"