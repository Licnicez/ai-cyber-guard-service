/**
 * app/api/login/route.ts (ตัวอย่างการนำไปใส่ใน API Login ของ Next.js App Router)
 * สำหรับตรวจจับ Brute-Force Login และ Credential Stuffing แบบ Real-time
 */

import { NextRequest, NextResponse } from "next/server";
import { analyzeRequestWithAI, extractClientIP, recordLoginAttempt } from "@/lib/aiGuard";

export async function POST(req: NextRequest) {
  const ip = extractClientIP(req.headers);
  const userAgent = req.headers.get("user-agent") || "-";
  const body = await req.json();
  const { username, password } = body;

  // --- ตัวอย่างการเช็กฐานข้อมูลผู้ใช้ในเว็บ Next.js ของคุณ ---
  // const user = await db.user.findUnique({ where: { username } });
  const userExists = username === "admin";
  const passwordValid = userExists && password === "Admin@2026!";

  if (!userExists || !passwordValid) {
    // 1. บันทึกสถิติว่า IP นี้ล็อกอินพลาดกับชื่อบัญชีนี้
    recordLoginAttempt(ip, username || "unknown", false);

    // 2. ส่งให้สมองกล AI วิเคราะห์ทันทีว่าเป็น Brute-Force หรือ Credential Stuffing หรือไม่
    const authEvent = !userExists ? "failed_invalid_user" : "failed_bad_password";
    const verdict = await analyzeRequestWithAI({
      ip,
      method: "POST",
      path: "/api/login",
      userAgent,
      statusCode: 401,
      authEvent,
      usernameAttempted: username,
    });

    if (verdict.recommended_action === "BLOCK_IP") {
      return NextResponse.json(
        {
          error: `ถูกระงับชั่วคราวโดย AI เนื่องจากตรวจพบพฤติกรรม ${verdict.predicted_class}`,
          ai_verdict: verdict,
        },
        { status: 403 }
      );
    }

    return NextResponse.json({ error: "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง" }, { status: 401 });
  }

  // กรณีล็อกอินสำเร็จ
  recordLoginAttempt(ip, username, true);
  return NextResponse.json({ status: "ok", user: username });
}
