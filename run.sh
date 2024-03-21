#!/bin/bash

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
}

cd 'C:\Users\Jesvi Jonathan\Documents\github\Snitch-GPT-Frontend' 
npm run dev &
node_pid=$!
echo "Node.js process started (PID $node_pid)"

# start http://localhost:5173/

sleep 3

echo ""
echo "AI Detective is running in the background."
echo "Please open http://localhost:5173/ in your browser."
echo ""

cd 'C:\Users\Jesvi Jonathan\Documents\github\AI-Generated-Text-Detection'


source "nig/Scripts/activate"
# source env/Scripts/activate
# .\nig\Scripts\activate


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