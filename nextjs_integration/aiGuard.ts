/**
 * lib/aiGuard.ts
 * โมดูลเชื่อมต่อสมองกล AI Guard สำหรับโปรเจกต์ Next.js (รองรับทั้ง App Router และ Pages Router)
 * ทำหน้าที่นับความถี่ Request ต่อวินาที (RPS), จำนวนครั้งที่ Login ผิด (Brute-Force / Credential Stuffing)
 * และส่งไปให้ AI Guard Service ตัดสินใจแบบ Real-time
 */

export interface AIGuardVerdict {
  predicted_class: string;
  confidence: number;
  is_anomaly: boolean;
  isolation_forest_outlier: boolean;
  anomaly_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  recommended_action: "ALLOW" | "CHALLENGE_CAPTCHA" | "RATE_LIMIT_OR_BLOCK" | "BLOCK_IP";
  inference_time_ms: number;
}

interface IPState {
  reqTimestamps: number[];
  failedAuthTimestamps: number[];
  userAttempts: Array<{ ts: number; username: string }>;
  pathAttempts: Array<{ ts: number; path: string }>;
  status404Timestamps: number[];
  statusErrorTimestamps: number[];
}

// เก็บสถิติ Sliding Window 60 วินาทีต่อ IP ไว้ใน Memory
const globalForAI = globalThis as unknown as { __aiGuardStore?: Map<string, IPState> };
const ipStore: Map<string, IPState> = globalForAI.__aiGuardStore || new Map<string, IPState>();
globalForAI.__aiGuardStore = ipStore;

const AI_GUARD_URL = (
  process.env.AI_GUARD_URL || "https://spencer-controversy-longer-worm.trycloudflare.com"
).replace(/\/$/, "");
const AI_GUARD_API_KEY = process.env.AI_GUARD_API_KEY || "ai-cyber-secret-key-2026";

function getOrCreateIPState(ip: string, now: number): IPState {
  let state = ipStore.get(ip);
  if (!state) {
    state = {
      reqTimestamps: [],
      failedAuthTimestamps: [],
      userAttempts: [],
      pathAttempts: [],
      status404Timestamps: [],
      statusErrorTimestamps: [],
    };
    ipStore.set(ip, state);
  }

  // ลบข้อมูลที่เก่ากว่า 60 วินาทีออก (Sliding Window 60s)
  const cutoff60 = now - 60_000;
  state.reqTimestamps = state.reqTimestamps.filter((t) => t > cutoff60);
  state.failedAuthTimestamps = state.failedAuthTimestamps.filter((t) => t > cutoff60);
  state.userAttempts = state.userAttempts.filter((item) => item.ts > cutoff60);
  state.pathAttempts = state.pathAttempts.filter((item) => item.ts > cutoff60);
  state.status404Timestamps = state.status404Timestamps.filter((t) => t > cutoff60);
  state.statusErrorTimestamps = state.statusErrorTimestamps.filter((t) => t > cutoff60);

  return state;
}

/**
 * ดึง Client IP จริงจาก Headers ของ Next.js / Cloudflare / Nginx / Vercel
 */
export function extractClientIP(headers: Headers): string {
  const cfIP = headers.get("cf-connecting-ip");
  if (cfIP) return cfIP.trim();

  const xff = headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();

  const realIP = headers.get("x-real-ip");
  if (realIP) return realIP.trim();

  return "127.0.0.1";
}

/**
 * บันทึกผลการพยายาม Login (เรียกใช้ใน API Route /api/login หรือ Server Action ของ Next.js)
 */
export function recordLoginAttempt(ip: string, username: string, success: boolean): void {
  const now = Date.now();
  const state = getOrCreateIPState(ip, now);
  if (username) {
    state.userAttempts.push({ ts: now, username });
  }
  if (!success) {
    state.failedAuthTimestamps.push(now);
    state.statusErrorTimestamps.push(now);
  }
}

/**
 * ส่งข้อมูล Telemetry ของ Request ปัจจุบันไปให้สมองกล AI Guard ตรวจสอบ
 */
export async function analyzeRequestWithAI(options: {
  ip: string;
  method: string;
  path: string;
  userAgent?: string;
  payloadSize?: number;
  statusCode?: number;
  authEvent?: "none" | "accepted_password" | "failed_bad_password" | "failed_invalid_user";
  usernameAttempted?: string;
}): Promise<AIGuardVerdict> {
  const now = Date.now();
  const state = getOrCreateIPState(options.ip, now);

  state.reqTimestamps.push(now);
  state.pathAttempts.push({ ts: now, path: options.path });

  const statusCode = options.statusCode ?? 200;
  if (statusCode === 404) state.status404Timestamps.push(now);
  if (statusCode >= 400) state.statusErrorTimestamps.push(now);

  const rps_1s = state.reqTimestamps.filter((t) => now - t <= 1_000).length;
  const rpm_10s = state.reqTimestamps.filter((t) => now - t <= 10_000).length;
  const rpm_60s = state.reqTimestamps.length;
  const failed_auth_60s = state.failedAuthTimestamps.length;
  const unique_users_60s = new Set(state.userAttempts.map((u) => u.username)).size;
  const unique_paths_60s = new Set(state.pathAttempts.map((p) => p.path)).size;
  const error_404_60s = state.status404Timestamps.length;
  const error_4xx_5xx_60s = state.statusErrorTimestamps.length;

  const payload = {
    client_ip: options.ip,
    method: options.method,
    path: options.path,
    status_code: statusCode,
    latency_ms: 1.0,
    payload_size: options.payloadSize ?? 0,
    user_agent: options.userAgent || "-",
    rps_1s,
    rpm_10s,
    rpm_60s,
    failed_auth_60s,
    unique_users_60s,
    unique_paths_60s,
    error_404_60s,
    error_4xx_5xx_60s,
    auth_event: options.authEvent ?? "none",
    username_attempted: options.usernameAttempted ?? "",
  };

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500); // Timeout 1.5s ไม่ให้เว็บหลักค้าง

    const res = await fetch(`${AI_GUARD_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": AI_GUARD_API_KEY,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timer);

    if (res.ok) {
      return (await res.json()) as AIGuardVerdict;
    }
  } catch (err) {
    // Fail-Open: หากเซิร์ฟเวอร์ AI ปิดอยู่ ให้ปล่อยผ่านเพื่อไม่ให้เว็บ Next.js ล่ม
    console.warn("[AI-Guard] Unreachable, failing open:", (err as Error).message);
  }

  return {
    predicted_class: "normal",
    confidence: 1.0,
    is_anomaly: false,
    isolation_forest_outlier: false,
    anomaly_score: 0.0,
    risk_level: "LOW",
    recommended_action: "ALLOW",
    inference_time_ms: 0,
  };
}
