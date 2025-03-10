#!/bin/bash

# Parse options
while getopts "h:" opt; do
    case ${opt} in
        h)
            HELLO_VALUE=$OPTARG
            ;;
    esac
done

# If no argument is passed for -h, show usage
if [ -z "$HELLO_VALUE" ]; then
    echo "Error: -h option is required"
    exit
fi

# Print the value for -h option
echo "Hello value: $HELLO_VALUE"

# Sleep for 60 seconds
sleep 10
