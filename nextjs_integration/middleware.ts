/**
 * middleware.ts (วางไว้ที่โฟลเดอร์หลัก หรือ src/middleware.ts ของโปรเจกต์ Next.js)
 * ดักจับทุก Request ที่วิ่งเข้าเว็บ Next.js เพื่อตรวจจับ DDoS / High-RPS Flood และ Path/Port Scan
 */

import { NextRequest, NextResponse } from "next/server";
import { analyzeRequestWithAI, extractClientIP } from "./aiGuard";

export async function middleware(req: NextRequest) {
  const path = req.nextUrl.pathname + (req.nextUrl.search || "");

  // ข้ามไฟล์ Static ของ Next.js เพื่อความเร็วสูงสุด
  if (
    path.startsWith("/_next") ||
    path.startsWith("/favicon.ico") ||
    path.match(/\.(png|jpg|jpeg|gif|svg|css|js|woff2?)$/i)
  ) {
    return NextResponse.next();
  }

  const clientIP = extractClientIP(req.headers);
  const userAgent = req.headers.get("user-agent") || "-";
  const contentLength = Number(req.headers.get("content-length") || 0);

  // ส่งข้อมูลให้สมองกล AI Guard วิเคราะห์
  const verdict = await analyzeRequestWithAI({
    ip: clientIP,
    method: req.method,
    path,
    userAgent,
    payloadSize: contentLength,
  });

  // หาก AI สั่ง BLOCK_IP หรือ RATE_LIMIT_OR_BLOCK ให้ตัดการเชื่อมต่อด้วย HTTP 403 ทันที
  if (
    verdict.recommended_action === "BLOCK_IP" ||
    verdict.recommended_action === "RATE_LIMIT_OR_BLOCK"
  ) {
    return NextResponse.json(
      {
        status: "blocked_by_ai_guard",
        message: `Access denied by AI Security Engine (${verdict.predicted_class})`,
        risk_level: verdict.risk_level,
        confidence: verdict.confidence,
      },
      { status: 403 }
    );
  }

  // แนบ Header ผลลัพธ์ของ AI ไว้ตรวจสอบใน Browser DevTools ได้
  const response = NextResponse.next();
  response.headers.set("X-AI-Guard-Class", verdict.predicted_class);
  response.headers.set("X-AI-Guard-Action", verdict.recommended_action);
  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
