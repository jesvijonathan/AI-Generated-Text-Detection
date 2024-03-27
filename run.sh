#!/bin/bash

# Make sure the project is fully set up, this is just an run script to automate the project
# sh run.sh

# paths and other configurable elements
PROJECT_DIR='C:\Users\Jesvi Jonathan\Documents\github'
FRONTEND_DIR="$PROJECT_DIR\Snitch-GPT-Frontend"
BACKEND_DIR="$PROJECT_DIR\AI-Generated-Text-Detection"
PORT1=5173
PORT2=5000

cleanup_and_exit() {
  echo "Stopping AI Detective and Chat Bot..."
  kill -TERM -$node_pid -$python_pid
  echo "Node.js process (PID $node_pid) and Python process (PID $python_pid) have been terminated."
  exit 0
}

start_python_app() {
  python app.py &
  python_pid=$!
  echo "Python process started (PID $python_pid)"
  Start http://localhost:$PORT2/
}

cd "$FRONTEND_DIR"
npm run dev &
node_pid=$!
echo "Node.js process started (PID $node_pid)"

start http://localhost:$PORT1/
sleep 3

echo ""
echo "AI Detective is running in the background."
echo "Please open http://localhost:$PORT1/ in your browser."
echo ""

cd "$BACKEND_DIR"

# Activate the virtual environment
source "nig/Scripts/activate"

echo "Environment activated."

echo "Running Chat Bot in the background."

echo "Press 'x' to stop the Chat Bot."

start_python_app

while true; do
  read -rsn1 key
  if [ "$key" == "x" ]; then
    cleanup_and_exit
  fi
done

echo "Script Ended"