#!/bin/bash

directory_name="sampleenv"

if [ -d $directory_name ]
then
    echo "Directory already exists"
else
    mkdir $directory_name
fi