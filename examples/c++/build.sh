#!/bin/bash

hostname=`hostname -f`
if [[ $hostname == *"aurora"* ]]; then
  echo Building for Aurora ...
  icpx -o mpi_example mpi_example.cpp -lmpi
  icpx -o serial_example serial_example.cpp
elif [[ $hostname == *"polaris"* ]]; then
  echo Building for Polaris ...
else
  echo No pre-defined build instructions for host $hostname 
fi
