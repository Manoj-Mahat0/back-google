#!/usr/bin/env python3
"""
Quick start script for the Indoor Navigation backend
Run from project root: python backend/scripts/run_backend.py
Or from backend dir: python scripts/run_backend.py
"""

import subprocess
import sys
import os

def main():
    print("🏢 Starting Indoor Navigation Backend...")
    
    # Determine backend directory (go up from scripts to backend)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    
    # Change to backend directory
    os.chdir(backend_dir)
    print(f"📂 Working directory: {backend_dir}")
    
    # Check if virtual environment exists
    venv_path = 'venv'
    if not os.path.exists(venv_path):
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', venv_path])
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_path, 'Scripts', 'pip')
        python_path = os.path.join(venv_path, 'Scripts', 'python')
    else:  # Unix/Linux/macOS
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    # Install dependencies
    print("📚 Installing dependencies...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'])
    
    # Start the server
    print("🚀 Starting FastAPI server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([python_path, '-m', 'uvicorn', 'main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'])
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")

if __name__ == "__main__":
    main()