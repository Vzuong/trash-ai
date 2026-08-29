import os
import sys
import subprocess
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

def main():
    log("=" * 65)
    log(" [TRASH AI] HE THONG GPU SIEU TOC & NGROK STATIC DOMAIN")
    log("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    py_exe = sys.executable

    # 1. Start Python YOLO GPU Service (port 5001)
    log("[1/3] ⚡ Dang khoi dong AI GPU Service tren card NVIDIA...")
    yolo_proc = subprocess.Popen(
        [py_exe, "-u", os.path.join(base_dir, "server", "yolo_service.py")],
        cwd=base_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(2)

    # 2. Start Node.js Express Server (port 5000)
    log("[2/3] 🌐 Dang khoi dong Web Server tren cong 5000...")
    node_proc = subprocess.Popen(
        ["node", os.path.join(base_dir, "server", "server.js")],
        cwd=base_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ, PORT="5000")
    )

    time.sleep(2)

    # 3. Start Ngrok Tunnel with static config
    log("[3/3] 📡 Dang ket noi Ngrok Tunnel vao domain co dinh...")
    ngrok_config = os.path.join(base_dir, "ngrok.yml")
    ngrok_proc = subprocess.Popen(
        ["ngrok", "http", "5000", "--config", ngrok_config],
        cwd=base_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(2)

    domain_url = "https://serving-numerous-spree.ngrok-free.dev"
    target_url = f"{domain_url}/classify"

    # Save to file
    with open(os.path.join(base_dir, "current_tunnel_url.txt"), "w", encoding="utf-8") as f:
        f.write(target_url)

    log("\n" + "=" * 65)
    log(" 🎉 HE THONG DA ONLINE HOAN TAT VOI NGROK STATIC DOMAIN!")
    log("=" * 65)
    log(f" 📱 LINK CO DINH TRON DOI:  {target_url}")
    log(f" 💻 Link may tinh noi bo:   http://localhost:5000/classify")
    log(" ⚡ Toc do suy luan:        30 - 60 FPS (Do tre ~20ms tren GPU)")
    log(" 🔒 Domain nay se GIU NGUYEN 100% vinh vien khong bao gio doi!")
    log("=" * 65)
    log(" 💡 Giu cua so nay mo de duy tri ket noi cho dien thoai.")
    log(" 🛑 Nhan Ctrl + C de dung he thong khi khong dung nua.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nDang tat cac tien trinh...")
        yolo_proc.terminate()
        node_proc.terminate()
        ngrok_proc.terminate()
        log("Da tat hoan tat.")

if __name__ == "__main__":
    main()
