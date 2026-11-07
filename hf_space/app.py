"""Hugging Face Space launcher for the Hybrid Smartphone ISP.

The full project (code, ONNX models, examples) lives on GitHub. At start-up this
script downloads that repository and runs its Gradio app (app.py), so the Space
itself only needs this file, requirements.txt and README.md.
"""
import io
import os
import runpy
import sys
import urllib.request
import zipfile

# ZeroGPU Spaces refuse to start without at least one @spaces.GPU function. The app runs on CPU
# (ONNX Runtime), so this no-op only satisfies that check; on CPU hardware `spaces` is simply skipped.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except Exception:
    pass

REPO_ZIP = "https://codeload.github.com/spandana13-code/hybrid-isp/zip/refs/heads/main"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "hybrid-isp-main")

if not os.path.exists(os.path.join(CODE, "app.py")):
    print("Downloading project from GitHub ...", flush=True)
    with urllib.request.urlopen(REPO_ZIP, timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(HERE)

os.chdir(CODE)
sys.path.insert(0, CODE)
runpy.run_path(os.path.join(CODE, "app.py"), run_name="__main__")
