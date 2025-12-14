#!/bin/bash

# Script to push changes to GitHub
# Usage: ./push_to_github.sh [commit message]

# Get commit message from argument or use default
COMMIT_MSG="${1:-Update project files}"

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Error: Git repository not initialized. Run 'git init' first."
    exit 1
fi

# Check if remote is set
if ! git remote get-url origin &>/dev/null; then
    echo "Setting up remote repository..."
    git remote add origin git@github.com:chiamin/TensorFluidDynamics.git
fi

# Add all files (respecting .gitignore)
echo "Adding files to staging area..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit."
    exit 0
fi

# Commit changes
echo "Committing changes with message: '$COMMIT_MSG'"
git commit -m "$COMMIT_MSG"

# Get current branch name
BRANCH=$(git branch --show-current)

# Push to GitHub
echo "Pushing to GitHub (branch: $BRANCH)..."
git push -u origin "$BRANCH"

if [ $? -eq 0 ]; then
    echo "Successfully pushed to GitHub!"
else
    echo "Error: Failed to push to GitHub. Please check your SSH keys and repository access."
    exit 1
fi
