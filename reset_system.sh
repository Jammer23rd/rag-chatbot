#!/bin/bash

# Reset RAG System
echo "🔄 Resetting RAG system..."
echo "-----------------------------"

# Delete database
echo "🧹 Cleaning vector database..."
rm -rf db/*

# Clear documents
echo "📄 Removing processed documents..."
rm -rf documents/*

# Remove logs
echo "🗑️  Clearing chat logs..."
rm -rf logs/*

# Recreate necessary directories
echo "📂 Recreating folder structure..."
mkdir -p documents logs db

echo "✅ Reset complete! System is ready for new documents."
echo "   1. Add new files to documents/ folder"
echo "   2. Run: python load_and_split_docs.py"
