# VPS Setup Guide — Vultr Deployment

## 🖥 Recommended VPS Specification

| Spec | Minimum | Recommended |
|------|---------|-------------|
| **OS** | Ubuntu 22.04 LTS | Windows Server 2022 |
| **CPU** | 2 vCPU | 4 vCPU (High Frequency) |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 50 GB SSD | 100 GB NVMe |
| **Location** | ใกล้ Broker Server | London / New York |

> [!IMPORTANT]
> ถ้าใช้ **MT5** ต้องรันบน **Windows Server** หรือติดตั้ง Wine บน Ubuntu
> แนะนำ **Windows Server 2022** เพื่อลด complexity ในการเชื่อมต่อ MT5

---

## Option A: Windows Server (แนะนำ)

### 1. Provision Vultr VPS
```
- เลือก Cloud Compute → High Frequency
- OS: Windows Standard 2022
- Plan: 4 vCPU / 8 GB RAM / 100 GB NVMe
- Location: เลือกตามตำแหน่ง Broker Server
```

### 2. Connect via RDP
```powershell
# ใช้ Remote Desktop Connection เชื่อมต่อด้วย IP จาก Vultr Dashboard
```

### 3. Install Python 3.11+
```powershell
# Download from python.org
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "python-installer.exe"
.\python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
```

### 4. Install MT5 Terminal
```
- Download MetaTrader 5 จาก Broker
- ติดตั้งและ Login ด้วย Account credentials
- เปิดให้รัน background: Tools → Options → Expert Advisors → Allow DLL imports
```

### 5. Clone & Setup Project
```powershell
cd C:\
git clone <repo-url> FxUltimatBot6159
cd FxUltimatBot6159
pip install -r requirements.txt
pip install -e .
```

### 6. Configure
```powershell
# แก้ไข config/default.yaml
notepad config\default.yaml
# ใส่: mt5.login, mt5.password, mt5.server, mt5.path
```

### 7. Test Run
```powershell
# ทดสอบ connection
python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.account_info())"

# Run paper trading
python scripts\live.py --paper
```

### 8. Auto-Start (Task Scheduler)
```powershell
# สร้าง Scheduled Task ให้รัน bot ตอนเปิดเครื่อง
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "scripts\live.py" -WorkingDirectory "C:\FxUltimatBot6159"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "FxBot" -Action $action -Trigger $trigger -RunLevel Highest
```

---

## Option B: Ubuntu + Wine

### 1. Provision
```
- OS: Ubuntu 22.04 LTS
- Same specs as above
```

### 2. Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git wget

# Install Wine for MT5
sudo dpkg --add-architecture i386
sudo apt install -y wine64 wine32 winetricks

# Setup Wine
winetricks dotnet48
```

### 3. Install MT5 on Wine
```bash
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
wine mt5setup.exe
```

### 4. Setup Python
```bash
git clone <repo-url> ~/FxUltimatBot6159
cd ~/FxUltimatBot6159
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 5. Create systemd Service
```bash
sudo tee /etc/systemd/system/fxbot.service << 'EOF'
[Unit]
Description=FxUltimatBot6159 AI Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/FxUltimatBot6159
ExecStart=/home/$USER/FxUltimatBot6159/venv/bin/python scripts/live.py
Restart=always
RestartSec=10
Environment=DISPLAY=:0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable fxbot
sudo systemctl start fxbot
sudo systemctl status fxbot
```

### 6. Log Monitoring
```bash
# View bot logs
tail -f ~/FxUltimatBot6159/logs/bot.log

# View trade journal
tail -f ~/FxUltimatBot6159/logs/trade_journal.log

# View systemd logs
journalctl -u fxbot -f
```

---

## 📊 Monitoring & Maintenance

### Health Check Commands
```bash
# Check if bot is running
ps aux | grep live.py

# Check log for errors
grep -i "error\|critical" logs/bot.log | tail -20

# Check trade activity
grep "ORDER" logs/trade_journal.log | tail -10
```

### Log Rotation
Logs auto-rotate at 10MB (5 backup files). For additional rotation:
```bash
sudo tee /etc/logrotate.d/fxbot << 'EOF'
/home/$USER/FxUltimatBot6159/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
}
EOF
```

---

## ⚠ Security Checklist

- [ ] Change default VPS password
- [ ] Enable firewall (only allow RDP/SSH from your IP)
- [ ] Store MT5 credentials in environment variables, not config files
- [ ] Enable 2FA on Vultr account
- [ ] Set up email alerts for bot errors
