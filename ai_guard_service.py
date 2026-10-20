"""
ai_guard_service.py
Real-time AI Engine API (FastAPI) ที่โหลดโมเดลจากไฟล์ cyber_ai_model.pkl
เพื่อทำหน้าที่เป็นสมองกลรับค่า Telemetry จากเว็บแอปมาประมวลผลด้วยโมเดลทั้ง 2 ตัว:
  1. RandomForestClassifier (จำแนกชนิดการโจมตี + ความมั่นใจ Confidence)
  2. IsolationForest (ตรวจจับพฤติกรรมผิดปกติแบบ Unsupervised Anomaly Score)
และส่งผลการตัดสินใจ (ALLOW, CHALLENGE_CAPTCHA, RATE_LIMIT_OR_BLOCK, BLOCK_IP) กลับภายในระดับมิลลิวินาที
"""

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from train_ai import (
    BOT_UA_REGEX,
    DEFAULT_CSV_PATH,
    DEFAULT_MODEL_PATH,
    SPECIAL_CHARS_REGEX,
    SUSPICIOUS_PATH_REGEX,
    train_and_save_models,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
CSV_PATH = Path(os.getenv("CSV_PATH", DEFAULT_CSV_PATH))
WEB_CSV_URL = os.getenv("WEB_CSV_URL", "")
AI_GUARD_API_KEY = os.getenv("AI_GUARD_API_KEY", "ai-cyber-secret-key-2026")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """ตรวจสอบ X-API-Key เมื่อเปิด REQUIRE_API_KEY=true สำหรับใช้งานข้าม Server ผ่านอินเทอร์เน็ต."""
    if REQUIRE_API_KEY and x_api_key != AI_GUARD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


class TelemetryPayload(BaseModel):
    client_ip: str = Field(default="127.0.0.1")
    method: str = Field(default="GET")
    path: str = Field(default="/")
    status_code: int = Field(default=200)
    latency_ms: float = Field(default=1.0)
    payload_size: int = Field(default=0)
    user_agent: str = Field(default="-")
    rps_1s: int = Field(default=1)
    rpm_10s: int = Field(default=1)
    rpm_60s: int = Field(default=1)
    failed_auth_60s: int = Field(default=0)
    unique_users_60s: int = Field(default=0)
    unique_paths_60s: int = Field(default=1)
    error_404_60s: int = Field(default=0)
    error_4xx_5xx_60s: int = Field(default=0)
    auth_event: str = Field(default="none")
    username_attempted: str = Field(default="")


class RetrainRequest(BaseModel):
    csv_url: Optional[str] = Field(default="")


class AIGuardBrain:
    """คลาสจัดการโมเดล AI ในหน่วยความจำเพื่อการตัดสินใจระดับมิลลิวินาที (Fast NumPy In-Memory Inference)."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.lock = threading.Lock()
        self.bundle: Dict[str, Any] = {}
        self.num_mean: Optional[np.ndarray] = None
        self.num_scale: Optional[np.ndarray] = None
        self.cat_categories: list = []
        self.total_predictions = 0
        self.blocked_count = 0
        self.recent_latencies_ms: Deque[float] = deque(maxlen=200)
        self.load_or_initialize()

    def load_or_initialize(self) -> None:
        if not self.model_path.exists():
            print(f"[AI-GUARD] ยังไม่พบไฟล์โมเดล '{self.model_path}' -> กำลังสั่งเทรนโมเดลตั้งต้นอัตโนมัติ...")
            train_and_save_models(csv_path=CSV_PATH, csv_url=WEB_CSV_URL, model_output_path=self.model_path)

        with self.lock:
            self.bundle = joblib.load(self.model_path)
            if "rf_classifier" in self.bundle:
                self.bundle["rf_classifier"].n_jobs = 1
            if "isolation_forest" in self.bundle:
                self.bundle["isolation_forest"].n_jobs = 1

            preprocessor = self.bundle["preprocessor"]
            scaler = preprocessor.named_transformers_["num"]
            encoder = preprocessor.named_transformers_["cat"]
            self.num_mean = np.asarray(scaler.mean_, dtype=np.float64)
            self.num_scale = np.asarray(scaler.scale_, dtype=np.float64)
            self.cat_categories = list(encoder.categories_)

            print(
                f"[AI-GUARD] โหลดโมเดล '{self.model_path}' สำเร็จ! "
                f"(Trained at: {self.bundle.get('trained_at')}, "
                f"RF Accuracy: {self.bundle.get('metrics', {}).get('rf_accuracy', 0) * 100:.2f}%)"
            )

    def _build_fast_feature_vector(self, payload: TelemetryPayload) -> Tuple[np.ndarray, Dict[str, Any]]:
        """สกัด Feature จาก URL/Path + Rate ด้วย NumPy โดยตรง (ลดเวลาประมวลผลเหลือ < 2 ms)."""
        path_str = payload.path or "/"
        ua_str = payload.user_agent or "-"

        path_length = len(path_str)
        path_depth = path_str.count("/")
        has_suspicious_keyword = 1 if SUSPICIOUS_PATH_REGEX.search(path_str) else 0
        special_char_count = len(SPECIAL_CHARS_REGEX.findall(path_str))
        is_login_path = 1 if "login" in path_str.lower() else 0
        is_bot_ua = 1 if BOT_UA_REGEX.search(ua_str) else 0

        rpm_safe = max(1.0, float(payload.rpm_60s))
        fails_safe = max(1.0, float(payload.failed_auth_60s))

        auth_fail_ratio = float(payload.failed_auth_60s) / rpm_safe
        user_stuffing_ratio = (
            float(payload.unique_users_60s) / fails_safe if payload.failed_auth_60s > 0 else 0.0
        )
        burst_intensity = float(payload.rps_1s) / max(1.0, rpm_safe / 60.0)
        error_404_ratio = float(payload.error_404_60s) / rpm_safe

        num_vals = np.array(
            [
                float(payload.status_code),
                float(payload.payload_size),
                float(payload.rps_1s),
                float(payload.rpm_10s),
                float(payload.rpm_60s),
                float(payload.failed_auth_60s),
                float(payload.unique_users_60s),
                float(payload.unique_paths_60s),
                float(payload.error_404_60s),
                float(payload.error_4xx_5xx_60s),
                float(path_length),
                float(path_depth),
                float(has_suspicious_keyword),
                float(special_char_count),
                float(is_login_path),
                float(is_bot_ua),
                auth_fail_ratio,
                user_stuffing_ratio,
                burst_intensity,
                error_404_ratio,
            ],
            dtype=np.float64,
        )

        scaled_num = (num_vals - self.num_mean) / self.num_scale

        # Fast One-Hot Encoding for ["method", "auth_event"]
        cat_inputs = [payload.method or "GET", payload.auth_event or "none"]
        cat_parts = []
        for val, cats in zip(cat_inputs, self.cat_categories):
            one_hot = np.zeros(len(cats), dtype=np.float64)
            matches = np.where(cats == val)[0]
            if len(matches) > 0:
                one_hot[matches[0]] = 1.0
            cat_parts.append(one_hot)

        full_vec = np.concatenate([scaled_num, *cat_parts]).reshape(1, -1)
        eng_summary = {
            "path_length": path_length,
            "has_suspicious_keyword": has_suspicious_keyword,
            "is_bot_ua": is_bot_ua,
            "auth_fail_ratio": round(auth_fail_ratio, 3),
            "user_stuffing_ratio": round(user_stuffing_ratio, 3),
        }
        return full_vec, eng_summary

    def predict_realtime(self, payload: TelemetryPayload) -> Dict[str, Any]:
        start_ns = time.perf_counter_ns()

        with self.lock:
            X_trans, eng_features = self._build_fast_feature_vector(payload)
            rf_model = self.bundle["rf_classifier"]
            iso_model = self.bundle["isolation_forest"]
            classes = self.bundle["classes"]

            # 1. โมเดลตัวที่ 1: RandomForestClassifier
            rf_probs = rf_model.predict_proba(X_trans)[0]
            best_idx = int(np.argmax(rf_probs))
            predicted_class = str(classes[best_idx])
            confidence = round(float(rf_probs[best_idx]), 4)

            # 2. โมเดลตัวที่ 2: IsolationForest (-1 = Anomaly, 1 = Normal)
            iso_pred = int(iso_model.predict(X_trans)[0])
            iso_score = round(float(iso_model.decision_function(X_trans)[0]), 4)
            is_anomaly = bool(iso_pred == -1 or predicted_class != "normal")

        # ตรรกะการตัดสินใจ (Decision Policy) จากผลลัพธ์ของโมเดลทั้ง 2 ตัว
        if predicted_class == "ddos_or_flood":
            risk_level = "CRITICAL"
            action = "RATE_LIMIT_OR_BLOCK"
        elif predicted_class in ("brute_force_login", "credential_stuffing"):
            risk_level = "HIGH" if confidence >= 0.6 else "MEDIUM"
            action = "BLOCK_IP" if payload.failed_auth_60s >= 5 else "CHALLENGE_CAPTCHA"
        elif predicted_class == "port_or_path_scan":
            risk_level = "HIGH"
            action = "BLOCK_IP"
        elif iso_pred == -1:
            risk_level = "MEDIUM"
            action = "CHALLENGE_CAPTCHA"
        else:
            risk_level = "LOW"
            action = "ALLOW"

        inference_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)

        with self.lock:
            self.total_predictions += 1
            if action in ("BLOCK_IP", "RATE_LIMIT_OR_BLOCK"):
                self.blocked_count += 1
            self.recent_latencies_ms.append(inference_ms)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_anomaly": is_anomaly,
            "isolation_forest_outlier": (iso_pred == -1),
            "anomaly_score": iso_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "inference_time_ms": inference_ms,
            "engineered_features": eng_features,
        }


brain = AIGuardBrain(MODEL_PATH)

app = FastAPI(
    title="AI Guard Real-Time Cybersecurity Engine",
    description="สมองกล AI สำหรับตรวจจับ Brute-Force, Credential Stuffing, DDoS และ Path Scan ระดับมิลลิวินาที",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    avg_ms = (
        round(sum(brain.recent_latencies_ms) / len(brain.recent_latencies_ms), 3)
        if brain.recent_latencies_ms
        else 0.0
    )
    return {
        "status": "online",
        "model_loaded": bool(brain.bundle),
        "model_path": str(brain.model_path),
        "trained_at": brain.bundle.get("trained_at"),
        "total_samples_trained": brain.bundle.get("total_samples", 0),
        "metrics": brain.bundle.get("metrics", {}),
        "classes": brain.bundle.get("classes", []),
        "total_predictions": brain.total_predictions,
        "blocked_or_mitigated": brain.blocked_count,
        "avg_inference_ms": avg_ms,
        "api_key_required": REQUIRE_API_KEY,
    }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(payload: TelemetryPayload):
    return brain.predict_realtime(payload)


@app.post("/retrain", dependencies=[Depends(verify_api_key)])
async def retrain(req: Optional[RetrainRequest] = None):
    try:
        csv_url = req.csv_url if req and req.csv_url else WEB_CSV_URL
        metrics = train_and_save_models(
            csv_path=CSV_PATH,
            csv_url=csv_url,
            model_output_path=MODEL_PATH,
        )
        brain.load_or_initialize()
        return {
            "status": "retrained",
            "trained_at": brain.bundle.get("trained_at"),
            "total_samples": brain.bundle.get("total_samples"),
            "metrics": metrics,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_guard_service:app", host="0.0.0.0", port=8001, reload=False)
