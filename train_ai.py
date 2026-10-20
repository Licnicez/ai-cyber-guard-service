"""
train_ai.py
สคริปต์สำหรับอ่านไฟล์ CSV จากเว็บ (หรือโฟลเดอร์ logs/app/traffic_features.csv),
สกัด Feature เพิ่มเติมจาก Path/URL + User-Agent + พฤติกรรมความถี่ (RPS/Auth),
และเทรนโมเดล AI ทั้ง 2 ตัว:
  1. RandomForestClassifier (Supervised Multi-Class Attack Classification)
  2. IsolationForest (Unsupervised Zero-Day / Outlier Anomaly Detection)
จากนั้นบันทึกโมเดลและ Pipeline ทั้งหมดลงในไฟล์ cyber_ai_model.pkl
"""

import argparse
import io
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = Path(os.getenv("CSV_PATH", BASE_DIR / "logs" / "app" / "traffic_features.csv"))
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "cyber_ai_model.pkl"))

SUSPICIOUS_PATH_REGEX = re.compile(
    r"(\.env|\.git|wp-login|wp-admin|phpmyadmin|actuator|config|backup|\.sql|\.php|passwd|shell|cgi-bin|server-status)",
    re.IGNORECASE,
)
BOT_UA_REGEX = re.compile(
    r"(hydra|bot|nmap|curl|python-requests|go-http-client|sqlmap|nikto|scanner| Masscan|zgrab)",
    re.IGNORECASE,
)
SPECIAL_CHARS_REGEX = re.compile(r"(\.\.|%|\?|=|&|'|\"|;|<|>)")

NUMERICAL_FEATURES = [
    "status_code",
    "payload_size",
    "rps_1s",
    "rpm_10s",
    "rpm_60s",
    "failed_auth_60s",
    "unique_users_60s",
    "unique_paths_60s",
    "error_404_60s",
    "error_4xx_5xx_60s",
    # Engineered URL / Path / Behavioral features
    "path_length",
    "path_depth",
    "has_suspicious_keyword",
    "special_char_count",
    "is_login_path",
    "is_bot_ua",
    "auth_fail_ratio",
    "user_stuffing_ratio",
    "burst_intensity",
    "error_404_ratio",
]

CATEGORICAL_FEATURES = [
    "method",
    "auth_event",
]


def generate_bootstrap_dataset(num_samples_per_class: int = 80) -> pd.DataFrame:
    """
    สร้างข้อมูลจำลองตั้งต้น (Bootstrap Synthetic Data) ครบทั้ง 5 คลาส
    ในกรณีที่ไฟล์ traffic_features.csv ยังไม่มีข้อมูลหรือมีข้อมูลไม่ครบทุกประเภทการโจมตี
    เพื่อให้สามารถเทรนโมเดล AI ได้ทันทีโดยไม่เกิด Error
    """
    rng = np.random.default_rng(42)
    rows: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    normal_paths = ["/", "/api/products", "/api/products?q=AI", "/api/products?q=Firewall", "/api/login"]
    scan_paths = [
        "/.env", "/wp-login.php", "/phpmyadmin", "/.git/config",
        "/admin/config.php", "/backup.sql", "/actuator/health", "/etc/passwd", "/shell.php",
    ]

    # 1. Normal traffic
    for i in range(num_samples_per_class):
        is_login = (i % 6 == 0)
        path = "/api/login" if is_login else rng.choice(normal_paths[:-1])
        method = "POST" if is_login else "GET"
        rpm_60 = int(rng.integers(1, 10))
        rows.append({
            "timestamp": now_iso,
            "client_ip": f"10.0.1.{rng.integers(2, 50)}",
            "method": method,
            "path": path,
            "status_code": 200,
            "latency_ms": float(rng.uniform(2.0, 25.0)),
            "payload_size": int(rng.integers(30, 80)) if is_login else 0,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0 Safari/537.36",
            "rps_1s": int(rng.integers(1, 3)),
            "rpm_10s": int(rng.integers(1, 5)),
            "rpm_60s": rpm_60,
            "failed_auth_60s": 0,
            "unique_users_60s": 1 if is_login else 0,
            "unique_paths_60s": int(rng.integers(1, 4)),
            "error_404_60s": 0,
            "error_4xx_5xx_60s": 0,
            "auth_event": "accepted_password" if is_login else "none",
            "username_attempted": "admin" if is_login else "",
            "suspected_pattern": "normal",
        })

    # 2. Brute-Force Login (Single account tried repeatedly)
    for i in range(num_samples_per_class):
        fails = int(rng.integers(5, 35))
        rows.append({
            "timestamp": now_iso,
            "client_ip": f"192.168.99.{rng.integers(10, 99)}",
            "method": "POST",
            "path": "/api/login",
            "status_code": 401,
            "latency_ms": float(rng.uniform(3.0, 18.0)),
            "payload_size": int(rng.integers(35, 65)),
            "user_agent": rng.choice(["Hydra/9.5 (BruteForce-Sim)", "python-requests/2.31.0", "Mozilla/5.0"]),
            "rps_1s": int(rng.integers(2, 10)),
            "rpm_10s": int(rng.integers(5, 25)),
            "rpm_60s": fails,
            "failed_auth_60s": fails,
            "unique_users_60s": int(rng.integers(1, 3)),
            "unique_paths_60s": 1,
            "error_404_60s": 0,
            "error_4xx_5xx_60s": fails,
            "auth_event": "failed_bad_password",
            "username_attempted": "admin",
            "suspected_pattern": "brute_force_login",
        })

    # 3. Credential Stuffing (Many unique usernames tried)
    for i in range(num_samples_per_class):
        fails = int(rng.integers(6, 40))
        unique_u = int(rng.integers(max(4, fails - 3), fails + 1))
        rows.append({
            "timestamp": now_iso,
            "client_ip": f"203.0.113.{rng.integers(10, 99)}",
            "method": "POST",
            "path": "/api/login",
            "status_code": 401,
            "latency_ms": float(rng.uniform(3.0, 20.0)),
            "payload_size": int(rng.integers(40, 75)),
            "user_agent": rng.choice(["CredentialStuffingBot/1.2", "python-requests/2.31.0"]),
            "rps_1s": int(rng.integers(3, 12)),
            "rpm_10s": int(rng.integers(6, 30)),
            "rpm_60s": fails,
            "failed_auth_60s": fails,
            "unique_users_60s": unique_u,
            "unique_paths_60s": 1,
            "error_404_60s": 0,
            "error_4xx_5xx_60s": fails,
            "auth_event": "failed_invalid_user",
            "username_attempted": f"leaked_user_{i}",
            "suspected_pattern": "credential_stuffing",
        })

    # 4. DDoS / High-RPS Flood
    for i in range(num_samples_per_class):
        rps_1 = int(rng.integers(15, 80))
        rpm_10 = int(rng.integers(max(30, rps_1), 250))
        rpm_60 = int(rng.integers(rpm_10, 600))
        rows.append({
            "timestamp": now_iso,
            "client_ip": f"198.51.100.{rng.integers(10, 250)}",
            "method": "GET",
            "path": f"/api/products?q=flood_{i}",
            "status_code": 200,
            "latency_ms": float(rng.uniform(1.0, 50.0)),
            "payload_size": 0,
            "user_agent": rng.choice(["Go-http-client/1.1", "python-requests/2.31.0", "curl/8.5.0"]),
            "rps_1s": rps_1,
            "rpm_10s": rpm_10,
            "rpm_60s": rpm_60,
            "failed_auth_60s": 0,
            "unique_users_60s": 0,
            "unique_paths_60s": int(rng.integers(1, 5)),
            "error_404_60s": 0,
            "error_4xx_5xx_60s": 0,
            "auth_event": "none",
            "username_attempted": "",
            "suspected_pattern": "ddos_or_flood",
        })

    # 5. Port / Path / Vulnerability Scanning
    for i in range(num_samples_per_class):
        u_paths = int(rng.integers(6, 25))
        err_404 = int(rng.integers(4, u_paths + 1))
        rows.append({
            "timestamp": now_iso,
            "client_ip": f"45.33.32.{rng.integers(10, 200)}",
            "method": "GET",
            "path": str(rng.choice(scan_paths)),
            "status_code": int(rng.choice([403, 404])),
            "latency_ms": float(rng.uniform(1.0, 12.0)),
            "payload_size": 0,
            "user_agent": rng.choice([
                "Nmap Scripting Engine; https://nmap.org/book/nse.html",
                "Nikto/2.1.6",
                "sqlmap/1.8",
            ]),
            "rps_1s": int(rng.integers(3, 14)),
            "rpm_10s": u_paths,
            "rpm_60s": u_paths,
            "failed_auth_60s": 0,
            "unique_users_60s": 0,
            "unique_paths_60s": u_paths,
            "error_404_60s": err_404,
            "error_4xx_5xx_60s": u_paths,
            "auth_event": "none",
            "username_attempted": "",
            "suspected_pattern": "port_or_path_scan",
        })

    return pd.DataFrame(rows)


def clean_and_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    ตรวจสอบและจัดการค่าว่าง (Missing / NULL values) ตามหลัก ML Best Practices:
    - คอลัมน์ข้อความ (path, user_agent, method, auth_event, username_attempted) แทนค่าว่างด้วยค่าเริ่มต้นที่มีความหมาย
    - คอลัมน์ตัวเลข แทนค่า NaN ด้วย 0
    - ลบแถวที่ไม่มี Target Label (suspected_pattern)
    """
    df = df.copy()
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        print(f"[INFO] ตรวจพบค่าว่าง (NULL/NaN) ทั้งหมด {total_nulls} จุด -> กำลังทำ Imputation...")

    # Drop rows missing target label if any exist in training set
    if "suspected_pattern" in df.columns:
        df["suspected_pattern"] = df["suspected_pattern"].replace("", np.nan)
        df = df.dropna(subset=["suspected_pattern"])

    text_defaults = {
        "method": "GET",
        "path": "/",
        "user_agent": "-",
        "auth_event": "none",
        "username_attempted": "",
    }
    for col, default_val in text_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val).astype(str)

    base_num_cols = [
        "status_code", "latency_ms", "payload_size",
        "rps_1s", "rpm_10s", "rpm_60s",
        "failed_auth_60s", "unique_users_60s", "unique_paths_60s",
        "error_404_60s", "error_4xx_5xx_60s",
    ]
    for col in base_num_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ปรับแก้ Noisy Labels ช่วงเริ่มต้นของการจำลอง (เช่น Request ที่ 1-2 ก่อนที่ RPS หรือจำนวนครั้งที่รหัสผิดจะพุ่งสูง)
    if "suspected_pattern" in df.columns:
        low_rate_flood_mask = (
            (df["suspected_pattern"] == "ddos_or_flood")
            & (df["rps_1s"] < 10)
            & (df["rpm_10s"] < 20)
        )
        single_fail_auth_mask = (
            df["suspected_pattern"].isin(["brute_force_login", "credential_stuffing"])
            & (df["failed_auth_60s"] <= 1)
        )
        df.loc[low_rate_flood_mask | single_fail_auth_mask, "suspected_pattern"] = "normal"

        # หากแถวที่เป็น normal บังเอิญเกิดจากการกดปุ่มจำลองซ้ำๆ บน IP เดิมจนความถี่สูงเกินจริง ให้ปรับค่าความถี่ให้อยู่ในเกณฑ์ผู้ใช้ปกติ
        normal_mask = df["suspected_pattern"] == "normal"
        df.loc[normal_mask & (df["rps_1s"] > 5), "rps_1s"] = 3
        df.loc[normal_mask & (df["rpm_10s"] > 12), "rpm_10s"] = 8
        df.loc[normal_mask & (df["rpm_60s"] > 25), "rpm_60s"] = 15

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    สกัด Feature เพิ่มเติมจาก Path/URL, User-Agent และสัดส่วนพฤติกรรม (Behavioral Ratios)
    ใช้ร่วมกันทั้งตอนเทรน (train_ai.py) และตอนตรวจจับจริงระดับมิลลิวินาที (ai_guard_service.py)
    """
    df = clean_and_handle_missing_values(df)

    paths = df["path"].astype(str)
    uas = df["user_agent"].astype(str)

    # 1. URL / Path Features
    df["path_length"] = paths.str.len()
    df["path_depth"] = paths.str.count("/")
    df["has_suspicious_keyword"] = paths.apply(lambda p: 1 if SUSPICIOUS_PATH_REGEX.search(p) else 0)
    df["special_char_count"] = paths.apply(lambda p: len(SPECIAL_CHARS_REGEX.findall(p)))
    df["is_login_path"] = paths.apply(lambda p: 1 if "login" in p.lower() else 0)

    # 2. User-Agent Bot Signature Feature
    df["is_bot_ua"] = uas.apply(lambda u: 1 if BOT_UA_REGEX.search(u) else 0)

    # 3. Behavioral Rate & Ratio Features
    rpm_safe = np.maximum(1.0, df["rpm_60s"].astype(float))
    fails_safe = np.maximum(1.0, df["failed_auth_60s"].astype(float))

    df["auth_fail_ratio"] = df["failed_auth_60s"].astype(float) / rpm_safe
    df["user_stuffing_ratio"] = np.where(
        df["failed_auth_60s"] > 0,
        df["unique_users_60s"].astype(float) / fails_safe,
        0.0,
    )
    df["burst_intensity"] = df["rps_1s"].astype(float) / np.maximum(1.0, rpm_safe / 60.0)
    df["error_404_ratio"] = df["error_404_60s"].astype(float) / rpm_safe

    return df


def load_csv_dataset(csv_path: Path, csv_url: str = "") -> pd.DataFrame:
    """
    โหลดไฟล์ CSV จาก URL ของเว็บแอป (เช่น http://localhost:8080/api/logs/download/traffic_features.csv)
    หรืออ่านจากไฟล์ในเครื่อง (logs/app/traffic_features.csv)
    พร้อมเสริมข้อมูล Bootstrap หากข้อมูลจริงยังมีไม่ครบทุกคลาส
    """
    df_loaded = pd.DataFrame()

    if csv_url:
        try:
            print(f"[INFO] กำลังดึงไฟล์ CSV จากเว็บ: {csv_url} ...")
            with urllib.request.urlopen(csv_url, timeout=5) as resp:
                raw_csv = resp.read().decode("utf-8")
            df_loaded = pd.read_csv(io.StringIO(raw_csv))
            print(f"[INFO] ดึงข้อมูลจากเว็บสำเร็จ: {len(df_loaded)} แถว")
        except Exception as exc:
            print(f"[WARN] ไม่สามารถดึง CSV จาก URL ({exc}) -> สลับไปอ่านไฟล์ในเครื่องแทน")

    if df_loaded.empty and csv_path.exists():
        try:
            df_loaded = pd.read_csv(csv_path)
            print(f"[INFO] อ่านข้อมูลจากไฟล์ {csv_path}: {len(df_loaded)} แถว")
        except Exception as exc:
            print(f"[WARN] อ่านไฟล์ CSV ไม่สำเร็จ ({exc})")

    required_classes = {
        "normal",
        "brute_force_login",
        "credential_stuffing",
        "ddos_or_flood",
        "port_or_path_scan",
    }

    existing_classes = (
        set(df_loaded["suspected_pattern"].dropna().unique())
        if not df_loaded.empty and "suspected_pattern" in df_loaded.columns
        else set()
    )

    if len(df_loaded) < 100 or not required_classes.issubset(existing_classes):
        print(
            f"[INFO] ข้อมูลปัจจุบันมี {len(df_loaded)} แถว (คลาสที่พบ: {existing_classes or 'ไม่มี'}) "
            "-> กำลังสร้างข้อมูลจำลองตั้งต้น (Bootstrap Dataset) ร่วมกับข้อมูลจริงเพื่อให้เทรนได้ครบทั้ง 5 คลาส..."
        )
        df_boot = generate_bootstrap_dataset(num_samples_per_class=80)
        df_combined = pd.concat([df_loaded, df_boot], ignore_index=True) if not df_loaded.empty else df_boot
        # Save back to CSV if CSV didn't exist or was empty
        if len(df_loaded) == 0:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_combined.to_csv(csv_path, index=False)
            print(f"[INFO] บันทึกข้อมูลตั้งต้นลงใน {csv_path} เรียบร้อยแล้ว ({len(df_combined)} แถว)")
        return df_combined

    return df_loaded


def train_and_save_models(
    csv_path: Path = DEFAULT_CSV_PATH,
    csv_url: str = "",
    model_output_path: Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    กระบวนการเทรนโมเดล AI ทั้ง 2 ตัวตามหลัก Strict Featurization Ordering:
      1. แบ่ง Train / Test Split ก่อนทำ Scaling / One-Hot Encoding เพื่อป้องกัน Data Leakage
      2. โมเดลตัวที่ 1: RandomForestClassifier (จำแนกประเภทการโจมตี 5 คลาส)
      3. โมเดลตัวที่ 2: IsolationForest (ตรวจจับพฤติกรรมผิดปกติแบบ Unsupervised Anomaly Detection)
      4. บันทึกโมเดลทั้งคู่และ Preprocessor ลงไฟล์ cyber_ai_model.pkl
    """
    df_raw = load_csv_dataset(csv_path=csv_path, csv_url=csv_url)
    df_feat = engineer_features(df_raw)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_feat[feature_cols]
    y = df_feat["suspected_pattern"].astype(str)

    # Strict Featurization Ordering: Split into train & test BEFORE fitting preprocessor
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    # Fit preprocessor ONLY on training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # =========================================================================
    # โมเดลตัวที่ 1: Supervised RandomForestClassifier (จำแนกชนิดการโจมตี)
    # =========================================================================
    print("\n[1/2] กำลังเทรนโมเดลตัวที่ 1: RandomForestClassifier (Multi-Class Attack Detector)...")
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_classifier.fit(X_train_transformed, y_train)

    y_pred = rf_classifier.predict(X_test_transformed)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    cls_report = classification_report(y_test, y_pred, digits=4)
    conf_mat = confusion_matrix(y_test, y_pred, labels=rf_classifier.classes_)

    print(f"  -> Accuracy (Test Set): {acc * 100:.2f}%")
    print(f"  -> Macro F1-Score     : {f1_macro * 100:.2f}%")
    print("\nClassification Report:\n" + cls_report)

    # =========================================================================
    # โมเดลตัวที่ 2: Unsupervised IsolationForest (ตรวจจับ Anomaly / Zero-Day)
    # =========================================================================
    print("[2/2] กำลังเทรนโมเดลตัวที่ 2: IsolationForest (Unsupervised Anomaly Detector)...")
    # Train IsolationForest primarily on normal traffic from X_train to model normal baseline
    normal_mask_train = (y_train == "normal").values
    X_train_normal = X_train_transformed[normal_mask_train]
    if len(X_train_normal) < 10:
        X_train_normal = X_train_transformed

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_normal)

    # Evaluate IsolationForest on test set (-1 = anomaly, 1 = normal)
    iso_preds_test = iso_forest.predict(X_test_transformed)
    actual_anomalies_test = np.where(y_test.values == "normal", 1, -1)
    iso_detection_acc = float(accuracy_score(actual_anomalies_test, iso_preds_test))
    print(f"  -> IsolationForest Anomaly Agreement Rate: {iso_detection_acc * 100:.2f}%")

    # ปิดการทำงานแบบ Multi-thread ตอนทำ Single-Row Real-time Inference เพื่อให้ตอบสนองระดับ 1-3 มิลลิวินาที
    rf_classifier.n_jobs = 1
    iso_forest.n_jobs = 1

    # =========================================================================
    # บันทึกโมเดลทั้ง 2 ตัวและ Pipeline ลงไฟล์ cyber_ai_model.pkl
    # =========================================================================
    artifact_bundle = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(df_feat),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": list(rf_classifier.classes_),
        "preprocessor": preprocessor,
        "rf_classifier": rf_classifier,
        "isolation_forest": iso_forest,
        "metrics": {
            "rf_accuracy": round(acc, 4),
            "rf_f1_macro": round(f1_macro, 4),
            "isolation_forest_agreement": round(iso_detection_acc, 4),
            "confusion_matrix": conf_mat.tolist(),
        },
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_bundle, model_output_path)
    print(f"\n[SUCCESS] บันทึกโมเดลทั้ง 2 ตัวลงในไฟล์ '{model_output_path}' เรียบร้อยแล้ว!")

    return artifact_bundle["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dual Cybersecurity AI Models from CSV.")
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="Path to local traffic_features.csv",
    )
    parser.add_argument(
        "--csv-url",
        default="",
        help="Optional URL to download traffic_features.csv from the web app (e.g., http://localhost:8080/api/logs/download/traffic_features.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for cyber_ai_model.pkl",
    )
    args = parser.parse_args()

    train_and_save_models(
        csv_path=Path(args.csv_path),
        csv_url=args.csv_url,
        model_output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
