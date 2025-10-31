#!/bin/bash

# GitHub Setup Script for BTP Project
# This script helps you push your project to GitHub

echo "=========================================="
echo "BTP Project - GitHub Setup"
echo "=========================================="
echo ""

# Step 1: Check if git is installed
if ! command -v git &> /dev/null
then
    echo "❌ Git is not installed. Please install git first:"
    echo "   brew install git"
    exit 1
fi

echo "✓ Git is installed: $(git --version)"
echo ""

# Step 2: Initialize git repository
echo "Step 1: Initializing git repository..."
git init
echo "✓ Git repository initialized"
echo ""

# Step 3: Configure git (if needed)
echo "Step 2: Configuring git..."
echo "Please enter your name (for git commits):"
read -p "Name: " git_name
git config user.name "$git_name"

echo "Please enter your email (for git commits):"
read -p "Email: " git_email
git config user.email "$git_email"

echo "✓ Git configured"
echo "  Name: $git_name"
echo "  Email: $git_email"
echo ""

# Step 4: Add remote repository
echo "Step 3: Adding GitHub repository..."
echo ""
echo "First, create a new repository on GitHub:"
echo "1. Go to: https://github.com/new"
echo "2. Name it something like: BTP-IoT-Healthcare-Monitoring"
echo "3. DO NOT initialize with README (we have one)"
echo "4. Create the repository"
echo "5. Copy the repository URL (https://github.com/USERNAME/REPO.git)"
echo ""
echo "Please enter your GitHub repository URL:"
read -p "URL: " repo_url

git remote add origin "$repo_url"
echo "✓ Remote repository added: $repo_url"
echo ""

# Step 5: Stage files
echo "Step 4: Staging files..."
git add .
echo "✓ Files staged"
echo ""

# Step 6: Show what will be committed
echo "Files to be committed:"
git status --short
echo ""

# Step 7: Create commit
echo "Step 5: Creating initial commit..."
git commit -m "Initial commit: BTP IoT Healthcare Monitoring System

- Data integration pipeline completed
- Comprehensive preprocessing framework
- MLP implementation (9-class posture classification)
- Complete documentation and reports
- Organized directory structure
- Test accuracy: 11.26% (vitals only)
- Next: IMU integration planned for 70-85% accuracy"

echo "✓ Commit created"
echo ""

# Step 8: Prepare for push
echo "Step 6: Preparing to push to GitHub..."
git branch -M main
echo "✓ Branch renamed to 'main'"
echo ""

# Step 9: Instructions for pushing
echo "=========================================="
echo "Ready to Push!"
echo "=========================================="
echo ""
echo "To push to GitHub, run:"
echo ""
echo "  git push -u origin main"
echo ""
echo "You'll be prompted for credentials:"
echo "  Username: your GitHub username"
echo "  Password: your Personal Access Token (NOT your GitHub password)"
echo ""
echo "If you don't have a Personal Access Token:"
echo "1. Go to: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Select 'repo' scope"
echo "4. Copy the token and use it as password"
echo ""
echo "After pushing, your project will be at:"
echo "  $repo_url"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
