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
        python generate_summary.py --dir "$dir"
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
    RESULTS_DIRS=(
        "grud_output"
        "ipnets_output"
        "seft_output"
        "transformer_output"
    )

    # Process each results directory
    for dir in "${RESULTS_DIRS[@]}"; do
        generate_summary "$dir"
    done

    print_header "All summaries generated!"
}

# Run main function
main