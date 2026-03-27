#!/bin/bash
echo "🚀 Starting Chatway Prototype Deployment Script..."

# 1. Enforce Env Credentials
if [ ! -f .env ]; then
    echo "⚠️  Wait! '.env' file not found!"
    echo "Creating one from '.env.example'..."
    cp .env.example .env
    echo "❌ Please update the '.env' file with your GOOGLE_API_KEY before deploying, then run this script again."
    exit 1
fi

# 2. Check for FAISS index (warn if they haven't ingested a PDF yet)
if [ ! -d "faiss_index" ]; then
    echo "⚠️  'faiss_index' directory not found. Please note you must run 'python ingest.py' locally or inside the container to build your knowledge base."
fi

# 3. Pull latest changes & spin up docker containers
echo "📦 Building and deploying with Docker Compose..."
docker compose up --build -d

echo ""
echo "✅ Deployment successful! Your chat app is now securely running on http://localhost:5000"
echo "👉 To view real-time logs, execute: docker compose logs -f"
