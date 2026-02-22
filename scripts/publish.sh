#!/bin/bash
set -e

# Configuration - Replace with your bucket name
BUCKET_NAME="vkp-agent-core-v2"

echo "🚀 Starting build and publish process..."

# 1. Ensure s3pypi is installed in the environment
echo "📦 Ensuring s3pypi is available..."
uv pip install s3pypi

# 2. Clean previous builds
rm -rf dist/

# 3. Build all packages in the workspace
echo "🏗️ Building packages..."
uv build --all-packages

# 4. Upload to S3
# Standard s3pypi usage: s3pypi upload <files> --bucket <bucket-name>
echo "📤 Uploading to S3 bucket: $BUCKET_NAME..."
uv run s3pypi upload dist/* --bucket "$BUCKET_NAME" --force

echo "✅ Done! You can now install your packages using:"
echo "pip install <package-name> --extra-index-url https://$BUCKET_NAME.s3.amazonaws.com/"
