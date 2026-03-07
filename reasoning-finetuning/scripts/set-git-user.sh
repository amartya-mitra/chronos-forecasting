#!/bin/bash

# Configuration
MY_NAME="Amartya Mitra"
MY_EMAIL="amitr003@ucr.edu"

# Apply settings
git config --global user.name "$MY_NAME"
git config --global user.email "$MY_EMAIL"

# Verification
echo "Git identity updated to:"
git config --global user.name
git config --global user.email