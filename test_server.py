"""Quick server launch test"""
import sys, os
sys.path.insert(0, ".")

# Apply patch before anything
from gradio_client import utils as gcu
_orig = gcu._json_schema_to_python_type
def _patched(schema, defs):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig(schema, defs)
gcu._json_schema_to_python_type = _patched

from ui.app import create_ui
import threading, urllib.request, time

app = create_ui()
print("App created, launching server...")

result = {"status": "unknown"}

def curl():
    time.sleep(8)
    for attempt in range(3):
        try:
            req = urllib.request.Request("http://127.0.0.1:7860")
            resp = urllib.request.urlopen(req, timeout=10)
            content = resp.read()
            result["status"] = f"HTTP {resp.status}: {len(content)} bytes"
            if b"wrapper" in content[:500] or b"gradio" in content[:500].lower():
                result["status"] += " - GRADIO PAGE OK"
            print(result["status"])
            return
        except Exception as e:
            print(f"Attempt {attempt+1}: {e}")
            time.sleep(3)

t = threading.Thread(target=curl, daemon=True)
t.start()
app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
print("Server stopped")
