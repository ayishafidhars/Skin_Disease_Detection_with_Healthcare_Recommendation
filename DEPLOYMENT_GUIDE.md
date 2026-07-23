# 🚀 DEPLOYMENT_GUIDE.md - Complete Deployment Instructions

## AI-Based Intelligent System for Skin Disease Detection

### Purpose
This guide provides step-by-step instructions for deploying the Skin Disease Detection system to production environments.

### Deployment Options
1. ✅ Local Development
2. ✅ Local Production (Single Machine)
3. ✅ Cloud Deployment (Azure/AWS/GCP)
4. ✅ Docker Container
5. ✅ Kubernetes Cluster

---

## Prerequisites

Regardless of deployment method, you need:

- ✅ Python 3.8+ installed
- ✅ Git access to repository
- ✅ Google OAuth credentials configured
- ✅ 4GB RAM minimum
- ✅ 200 MB disk space
- ✅ Internet connection

---

## Option 1: Local Development Deployment

### Perfect For:
- Personal testing
- Development/debugging
- Demonstration purposes

### Steps:

**1. Clone Repository**
```bash
git clone <repository-url>
cd AI-Based-Intelligent-System-for-Skin-Disease-Detection...
```

**2. Create Virtual Environment**
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r SkinDisease/requirements.txt
```

**4. Configure Streamlit**
```bash
# Create Streamlit config
mkdir SkinDisease/.streamlit
```

Create file: `SkinDisease/.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FF6E6C"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true

[logger]
level = "debug"
```

**5. Configure Secrets**
Create file: `SkinDisease/.streamlit/secrets.toml`
```toml
[auth.google]
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
redirect_uri = "http://localhost:8501"
```

**6. Run Application**
```bash
cd SkinDisease
streamlit run app.py
```

Application available at: `http://localhost:8501`

---

## Option 2: Local Production Deployment

### Perfect For:
- Small organization (< 50 users)
- On-premise deployment
- Single server setup

### Hardware Requirements:
- CPU: 4+ cores
- RAM: 8GB+ (16GB recommended)
- Storage: 500GB (for logs, backups)
- OS: Windows Server / Linux Server

### Steps:

**1. System Preparation**

Windows Server:
```powershell
# Install Python
# Download from github.com/python/cpython releases

# Install Nginx (optional reverse proxy)
choco install nginx

# Create application directory
mkdir "C:\Applications\SkinDisease"
```

Linux Server:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & dependencies
sudo apt install python3.10 python3-pip nginx

# Create directory
mkdir /opt/skindisease
```

**2. Clone & Setup**
```bash
cd /opt/skindisease  # or C:\Applications\SkinDisease
git clone <repository-url>
cd AI-Based-Intelligent-System-...

python3 -m venv .venv
source .venv/bin/activate
pip install -r SkinDisease/requirements.txt
```

**3. Configure Secrets**
Create environment variables file:

Windows: `.env`
```
GOOGLE_CLIENT_ID=your_id
GOOGLE_CLIENT_SECRET=your_secret
DATABASE_PATH=/data/users.db
```

Linux: `/etc/skindisease/.env`
```bash
export GOOGLE_CLIENT_ID="your_id"
export GOOGLE_CLIENT_SECRET="your_secret"
export DATABASE_PATH="/data/users.db"
```

**4. Setup Reverse Proxy (Nginx)**

File: `/etc/nginx/sites-available/skindisease`
```nginx
upstream streamlit {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    location / {
        proxy_pass http://streamlit;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable Nginx site:
```bash
cd /etc/nginx/sites-enabled
ln -s ../sites-available/skindisease .
sudo systemctl restart nginx
```

**5. Create Systemd Service (Linux)**

File: `/etc/systemd/system/skindisease.service`
```ini
[Unit]
Description=Skin Disease Detection System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/skindisease
Environment="PATH=/opt/skindisease/.venv/bin"
ExecStart=/opt/skindisease/.venv/bin/streamlit run SkinDisease/app.py --server.port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable skindisease
sudo systemctl start skindisease
sudo systemctl status skindisease
```

**6. Setup SSL Certificate**

Using Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d your-domain.com
```

**7. Database Backup Automation**

Create backup script: `/opt/skindisease/backup.sh`
```bash
#!/bin/bash
DB_PATH="/opt/skindisease/SkinDisease/users.db"
BACKUP_DIR="/backup/skindisease"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
cp $DB_PATH $BACKUP_DIR/users_$DATE.db

# Keep last 30 days only
find $BACKUP_DIR -name "*.db" -mtime +30 -delete

# Zip backups
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz -C $BACKUP_DIR users_$DATE.db
```

Add to crontab (backup every 6 hours):
```bash
crontab -e
# Add: 0 */6 * * * /opt/skindisease/backup.sh
```

**8. Monitoring & Logging**

Configure logging:
```python
# In SkinDisease/app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='logs/app.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Access Application:
- URL: `https://your-domain.com`
- Port: 443 (HTTPS)

---

## Option 3: Docker Container Deployment

### Perfect For:
- Easy scaling
- Consistent environments
- Cloud deployment

### Steps:

**1. Create Dockerfile**

File: `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r SkinDisease/requirements.txt

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "SkinDisease/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**2. Build Docker Image**
```bash
docker build -t skindisease:latest .
```

**3. Run Container**
```bash
docker run -p 8501:8501 \
  -e GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID \
  -e GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET \
  -v /data/users.db:/app/SkinDisease/users.db \
  skindisease:latest
```

**4. Docker Compose (Multi-container)**

File: `docker-compose.yml`
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID}
      GOOGLE_CLIENT_SECRET: ${GOOGLE_CLIENT_SECRET}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: always

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - app
```

Run:
```bash
docker-compose up -d
```

---

## Option 4: Cloud Deployment (Azure)

### Using Azure App Service:

**1. Create App Service**
```bash
az group create --name skindisease-rg --location eastus
az appservice plan create --name skindisease-plan --resource-group skindisease-rg
az webapp create --resource-group skindisease-rg --name skindisease-app --plan skindisease-plan --runtime "python|3.10"
```

**2. Deploy Code**
```bash
cd C:\path\to\project
az webapp deployment source config-zip --resource-group skindisease-rg --name skindisease-app --src app.zip
```

**3. Configure Environment Variables**
```bash
az webapp config appsettings set \
  --name skindisease-app \
  --resource-group skindisease-rg \
  --settings \
    GOOGLE_CLIENT_ID="your-id" \
    GOOGLE_CLIENT_SECRET="your-secret"
```

**4. Setup Custom Domain**
```bash
az webapp config hostname add \
  --resource-group skindisease-rg \
  --webapp-name skindisease-app \
  --hostname your-domain.com
```

---

## Post-Deployment Checklist

### Security
- [ ] Configure HTTPS/SSL certificates
- [ ] Set strong database passwords
- [ ] Enable firewall rules
- [ ] Configure backup encryption
- [ ] Enable audit logging
- [ ] Set file permissions (chmod 600 for .db)

### Performance
- [ ] Test under load
- [ ] Configure caching
- [ ] Monitor response times
- [ ] Setup CDN for static files
- [ ] Monitor CPU/memory usage

### Reliability
- [ ] Setup automated backups
- [ ] Test backup recovery
- [ ] Enable monitoring alerts
- [ ] Document recovery procedures
- [ ] Setup redundancy/failover

### Maintenance
- [ ] Schedule regular updates
- [ ] Plan dependency updates
- [ ] Document admin procedures
- [ ] Create runbooks
- [ ] Train operations team

---

## Monitoring & Maintenance

### Key Metrics to Monitor:
- Application uptime
- API response time
- Error rates
- Database size
- Disk space
- Memory usage
- CPU usage

### Tools to Use:
- Datadog, New Relic, Prometheus
- ELK Stack (logging)
- Grafana (visualization)

---

## Troubleshooting Deployment

### App won't start
```bash
# Check logs
journalctl -u skindisease -n 50
# or
docker logs <container-id>
```

### Port already in use
```bash
# Linux/Mac
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Database locked
```bash
# Restart application
sudo systemctl restart skindisease
```

### Out of memory
- Increase VM/container memory
- Enable swap
- Optimize model inference

---

## Rollback Procedure

If deployment fails:

```bash
# Git rollback
git revert HEAD
git push

# Docker rollback
docker run tagsof previous-version
```

---

## Version Control

Keep track of deployments:
```bash
# Tag releases
git tag -a v1.0.0 -m "Production release"
git push origin v1.0.0

# View release history
git tag -l
```

---

**Document Version**: 1.0  
**Last Updated**: March 2026  
**Maintained By**: DevOps Team
