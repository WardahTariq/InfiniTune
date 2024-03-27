#!/bin/bash

# Define the base file name
base_name1="knobs"
base_name2="latency"

# Get the current timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Initialize the counter
counter=1

# Construct the file name with timestamp and counter
knobs_file="${base_name1}_${counter}_${timestamp}.log"
latency_file="${base_name2}_${counter}_${timestamp}"

# Find the next available file name
while [ -e "$file_name" ]; do
    ((counter++))
    knobs_file="${base_name1}_${counter}_${timestamp}.log"
    latency_file="${base_name2}_${counter}_${timestamp}" 
done

# Create the file
touch "$knobs_file"
touch "$latency_file"

# Optionally, you can append some content to the file
echo "This is a log file created at $timestamp" >> "$knobs_file"
echo "This is a log file created at $timestamp" >> "$latency_file"

cp ../tmpLog/Training_w.log "$knobs_file"
cp ../tmpLog/TraininglogFile "$latency_file"

