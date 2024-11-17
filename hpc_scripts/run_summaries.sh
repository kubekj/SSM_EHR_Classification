#!/bin/bash

# Function to print section headers
print_header() {
    echo "----------------------------------------"
    echo "$1"
    echo "----------------------------------------"
}

# Function to generate summary for a specific results directory
generate_summary() {
    local dir=$1
    local model_name=$(basename "$dir")

    print_header "Generating summary for $model_name"

    if [ -d "$dir" ]; then
        python hpc_scripts/get_summary.py --dir "$dir"
        if [ $? -eq 0 ]; then
            echo "Successfully generated summary for $model_name"
        else
            echo "Error generating summary for $model_name"
        fi
    else
        echo "Directory $dir does not exist!"
    fi
    echo
}

# Main script
main() {
    # List of results directories to process
    local dir="baseline_outputs"
    RESULTS_DIRS=(
        "$dir/grud_output"
        "$dir/ipnets_output"
        "$dir/seft_output"
        "$dir/transformer_output"
    )

    # Process each results directory
    for dir in "${RESULTS_DIRS[@]}"; do
        generate_summary "$dir"
    done

    print_header "All summaries generated!"
}

# Run main function
main