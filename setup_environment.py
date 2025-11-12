import os
import platform
import subprocess
import sys
import venv

# === CONFIG ===
ENV_NAME = "tracking_env"
REQUIREMENTS = """
ultralytics==8.1.34
deep-sort-realtime==1.3.2
opencv-python==4.9.0.80
pandas==2.2.1
numpy==1.24.4
matplotlib==3.8.2
streamlit==1.35.0
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
"""

# === CREATE VIRTUAL ENVIRONMENT ===
print("🚀 Membuat virtual environment...")
if not os.path.exists(ENV_NAME):
    venv.create(ENV_NAME, with_pip=True)
    print(f"✅ Environment '{ENV_NAME}' berhasil dibuat.")
else:
    print(f"ℹ️ Environment '{ENV_NAME}' sudah ada, lanjut ke instalasi...")

# === DETECT OS ===
is_windows = platform.system().lower() == "windows"
activate_cmd = (
    f"{ENV_NAME}\\Scripts\\activate"
    if is_windows
    else f"source {ENV_NAME}/bin/activate"
)

# === CREATE TEMP REQUIREMENTS FILE ===
req_file = "requirements_tmp.txt"
with open(req_file, "w") as f:
    f.write(REQUIREMENTS.strip())

# === INSTALL DEPENDENCIES ===
print("\n📦 Menginstal library yang diperlukan...\n")
pip_exe = os.path.join(ENV_NAME, "Scripts" if is_windows else "bin", "pip")

subprocess.run([pip_exe, "install", "--upgrade", "pip"])
subprocess.run([pip_exe, "install", "-r", req_file])

# === CLEANUP ===
os.remove(req_file)

# === SUCCESS MESSAGE ===
print("\n✅ Setup selesai!")
print(f"Untuk mulai menggunakan environment ini, jalankan perintah berikut:\n")

if is_windows:
    print(f"  {activate_cmd}")
else:
    print(f"  source {activate_cmd}")

print("\nSetelah aktif, jalankan dashboard dengan:\n")
print("  streamlit run dashboard_streamlit.py\n")
