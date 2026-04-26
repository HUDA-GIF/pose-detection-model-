# launcher.py
# Clean entry point for the packaged exe
import os
import sys

# Make sure the app can find its files when running as exe
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    base_path = sys._MEIPASS
    os.chdir(os.path.dirname(sys.executable))
else:
    # Running as normal Python script
    base_path = os.path.dirname(os.path.abspath(__file__))

import webbrowser
import threading
import time

def open_browser():
    """Opens browser after server starts"""
    time.sleep(2.5)
    webbrowser.open("http://localhost:5000")

# Open browser automatically
threading.Thread(target=open_browser, daemon=True).start()

# Start the Flask app
from app import app
print("\n🚀 AI Fitness Coach starting...")
print("   Browser will open automatically...")
print("   Press Ctrl+C to stop\n")
app.run(debug=False, threaded=True, use_reloader=False)