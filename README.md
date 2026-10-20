# AI Cyber Guard Service (Standalone 24/7 Microservice)

ระบบสมองกล AI ตรวจจับภัยคุกคามทางไซเบอร์แบบ Real-time (**Brute-Force Login**, **Credential Stuffing**, **DDoS / High-RPS Flood**, และ **Port / Path Scanning**) ด้วยโมเดลคู่ (`RandomForestClassifier` + `IsolationForest`) จากไฟล์ `cyber_ai_model.pkl`

---

## ไฟล์ใน Repository นี้

- `cyber_ai_model.pkl` — ไฟล์สมองกล AI ที่เทรนเสร็จเรียบร้อยแล้ว (Accuracy 99.27%)
- `ai_guard_service.py` — FastAPI Real-time AI Engine (`POST /predict`, `GET /health`, `POST /retrain`)
- `train_ai.py` — สคริปต์สกัด Feature จาก URL/Path และเทรนโมเดลทั้ง 2 ตัว
- `requirements.txt` — รายการไลบรารี Python
- `Dockerfile` & `docker-compose.yml` — สำหรับรันบน Server / VPS / Cloud ตลอด 24 ชั่วโมง

---

## วิธีนำไปรันบน Server / Cloud ตลอด 24 ชั่วโมง

### วิธีที่ 1: รันด้วย Docker บน VPS / Public Server
```bash
git clone https://github.com/Licnicez/ai-cyber-guard-service.git
cd ai-cyber-guard-service
docker compose up -d --build
```
- เช็กสถานะ: `http://<SERVER_IP>:8001/health`
- ยิงตรวจจับ: `POST http://<SERVER_IP>:8001/predict`

### วิธีที่ 2: Deploy บน Render.com หรือ Railway.app (ฟรี HTTPS URL ตลอด 24 ชม.)
1. ไปที่ [Render.com](https://render.com) หรือ [Railway.app](https://railway.app)
2. กด **New Web Service** -> เลือกเชื่อมต่อกับ GitHub Repository นี้ (`ai-cyber-guard-service`)
3. เลือก Runtime เป็น **Docker** (หรือ Python: `uvicorn ai_guard_service:app --host 0.0.0.0 --port $PORT`)
4. เมื่อ Deploy เสร็จ จะได้ลิงก์ HTTPS ถาวร เช่น `https://ai-cyber-guard-service.onrender.com/predict` นำไปใส่ใน `AI_GUARD_URL` ของเว็บ Next.js ได้ทันที!
