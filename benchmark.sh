#!/bin/bash

# Function to print system information
print_system_info() {
    echo "System Information:"
    echo "Processor: $(uname -p)"
    echo "RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    else
        echo "No GPU detected"
    fi
    echo ""
}

# Function to generate payload based on number of inputs
generate_payload() {
    local base_input=$1
    local num=$2
    local service=$3

    inputs=()
    for ((i=1; i<=num; i++)); do
        inputs+=("\"$base_input\"")
    done

    joined_inputs=$(IFS=,; echo "${inputs[*]}")

    case $service in
        "embedding")
            echo "{\"input\": [$joined_inputs], \"model\": \"jinaai/jina-embeddings-v2-base-en\"}"
            ;;
        "splade")
            echo "{\"inputs\": [$joined_inputs]}"
            ;;
        "reranking")
            local query="This is a test query for reranking."
            local doc1="This is the first document for reranking."
            local doc2="This is the second document for reranking."
            local doc3="This is the third document for reranking."
            echo "{\"query\": \"$query\", \"texts\": [$joined_inputs], \"truncate\": false}"
            ;;
    esac
}

# Function to run a single request
run_single_request() {
    local url=$1
    local payload=$2
    curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$url" > /dev/null
}

# Function to run benchmark
run_benchmark() {
    local url=$1
    local base_input=$2
    local service_name=$3
    local number=$4
    local repeat=$5
    local async=$6
    local service_type=$7

    echo "$service_name:"
    total_time=0

    for i in $(seq 1 $repeat); do
        payload=$(generate_payload "$base_input" $number "$service_type")
        if [ $async -gt 1 ]; then
            # Run requests in parallel
            time_output=$( { time (
                for j in $(seq 1 $async); do
                    run_single_request "$url" "$payload" &
                done
                wait
            ); } 2>&1 )
        else
            # Run requests sequentially
            time_output=$( { time run_single_request "$url" "$payload"; } 2>&1 )
        fi

        real_time=$(echo "$time_output" | grep real | awk '{print $2}')
        minutes=$(echo $real_time | cut -d'm' -f1)
        seconds=$(echo $real_time | cut -d'm' -f2 | sed 's/s//')
        time_in_seconds=$(echo "$minutes * 60 + $seconds" | bc)

        total_time=$(echo "$total_time + $time_in_seconds" | bc)
    done

    avg_time=$(echo "scale=3; ($total_time / $repeat) * 1000" | bc)
    echo "  Average Time: $avg_time ms per request"
    echo "  Number of inputs per request: $number"
    echo "  Number of repeats: $repeat"
    echo "  async size: $async"
    echo "  Total requests: $repeat"

    if (( $(echo "$avg_time > 1000" | bc -l) )); then
        echo "  Warning: Average time exceeds 1000 ms"
    fi
}

# Set variables
embedding_url="http://localhost:6000/embeddings"
splade_doc_url="http://localhost:4000/embed_sparse"
splade_query_url="http://localhost:5000/embed_sparse"
reranking_url="http://localhost:8000/rerank"
number=1
repeat=1
async=2  # New parameter for async size

# Print system information
print_system_info

# Test embedding service
embedding_input="This is a test sentence for embedding."
run_benchmark "$embedding_url" "$embedding_input" "Embedding Service" $number $repeat $async "embedding"

# Test SPLADE doc service
splade_doc_input="This is a test sentence for SPLADE document embedding."
run_benchmark "$splade_doc_url" "$splade_doc_input" "SPLADE Doc Service" $number $repeat $async "splade"

# Test SPLADE query service
splade_query_input="This is a test query for SPLADE."
run_benchmark "$splade_query_url" "$splade_query_input" "SPLADE Query Service" $number $repeat $async "splade"

# Test reranking service
reranking_input="This is a test document for reranking."
run_benchmark "$reranking_url" "$reranking_input" "Reranking Service" $number $repeat $async "reranking"