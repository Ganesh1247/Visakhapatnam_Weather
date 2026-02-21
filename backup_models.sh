#!/bin/bash

# Create backup directory
mkdir -p models_backup

# Copy models
echo "Backing up models from models/ to models_backup/..."
cp -rv models/* models_backup/

echo "Backup complete."
