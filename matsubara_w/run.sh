#!/bin/bash
find . -name "*.pdf" -type f -delete
find . -name "*.png" -type f -delete
find . -name "*.txt" -type f -delete
python "main.py"