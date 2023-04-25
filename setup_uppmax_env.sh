#!/bin/bash

# create folder for storing fractures and trainind data
mkdir data
mkdir fractures

# load python
module load python3/3.9.5

# create virtual environment
python3 -m venv venv

# activate virtual environment
source venv/bin/activate

pip install -r requirements_uppmax.txt

