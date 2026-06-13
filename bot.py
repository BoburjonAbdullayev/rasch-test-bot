import asyncio
import csv
import io
import json
import logging
import math
import os
import sqlite3
import time
from contextlib import closing
import numpy as np
import pandas as pd

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
try:
    from aiogram.fsm.storage.redis import RedisStorage
    _redis_url = os.environ.get("REDIS_URL", "")
    if _redis_url:
        _storage = RedisStorage.from_url(_redis_url)
    else:
        _storage = MemoryStorage()
except Exception:
    _storage = MemoryStorage()
from aiogram.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    WebAppInfo, FSInputFile, BufferedInputFile
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOKEN    = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable o'rnatilmagan!")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))
WEB_APP_URL = "https://boburjonabdullayev.github.io/test-platform2/"

TOTAL_QUESTIONS = 55
CLOSED_COUNT    = 35
DB = "rasch_bot.db"

# ── States ────────────────────────────────────────────────────────────────
class Admin(StatesGroup):
    test_name   = State()
    test_code   = State()
    wait_keys   = State()
    excel_code  = State()

class Student(StatesGroup):
    enter_code  = State()
    enter_name  = State()
    wait_answers = State()

# ── Math normalizer ───────────────────────────────────────────────────────
def clean_math_expression(expr):
    if not expr: return ""
    s = str(expr).strip().lower()
    s = s.replace(" ", "")
    s = s.replace(r"\cdot", "").replace("*", "").replace(r"\times", "")
    s = s.replace("{", "").replace("}", "").replace("(", "").replace(")", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace(r"\frac", "")
    return s

# ── DB ────────────────────────────────────────────────────────────────────
def db_init():
    with closing(sqlite3.connect(DB, timeout=30)) as con:
        try:
            con.execute("PRAGMA journal_mode=WAL;")
        except Exception as e:
            log.warning(f"WAL rejimi yoqilmadi, standart rejimda davom etiladi: {e}")
        con.execute("PRAGMA busy_timeout=30000;")
        con.executescript("""
        CREATE TABLE IF NOT EXISTS tests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            code        TEXT NOT NULL UNIQUE,
            name        TEXT NOT NULL,
            n_questions INTEGER NOT NULL DEFAULT 55,
            answer_key  TEXT NOT NULL,
            is_active   INTEGER DEFAULT 1,
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS responses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id     INTEGER NOT NULL,
            tg_id       INTEGER NOT NULL,
            full_name   TEXT NOT NULL,
            raw_answers TEXT NOT NULL,
            binary_str  TEXT NOT NULL,
            submitted_at TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(test_id, tg_id)
        );
        CREATE TABLE IF NOT EXISTS admins (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id   INTEGER NOT NULL UNIQUE,
            added_at TEXT DEFAULT (datetime('now','localtime'))
        );
        """)
        try:
            con.execute("INSERT OR IGNORE INTO admins (tg_id) VALUES (?)", (ADMIN_ID,))
            con.commit()
        except Exception:
            pass

def db_get(q, p=()):
    with closing(sqlite3.connect(DB, timeout=30)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(q, p).fetchall()
    return rows

def db_run(q, p=()):
    with closing(sqlite3.connect(DB, timeout=30)) as con:
        try:
            lid = con.execute(q, p).lastrowid
            con.commit()
        except Exception:
            con.rollback()
            raise
    return lid

def is_admin(tg_id: int) -> bool:
    return bool(db_get("SELECT 1 FROM admins WHERE tg_id=?", (tg_id,)))

# ── IRT 2PL (o'zgartirilmagan) ────────────────────────────────────────────
def prob_2pl(theta, alpha, beta):
    return 1.0 / (1.0 + np.exp(-alpha * (theta - beta)))

def estimate_2pl_parameters(matrix):
    n_s, n_q = matrix.shape
    alpha = np.ones(n_q); beta = np.zeros(n_q)
    total_scores = matrix.sum(axis=1)
    for q in range(n_q):
        correct = np.sum(matrix[:, q])
        p = max(0.01, min(0.99, correct / n_s))
        beta[q] = -math.log(p / (1.0 - p))
        c_mask = (matrix[:, q] == 1); i_mask = (matrix[:, q] == 0)
        if np.any(c_mask) and np.any(i_mask):
            diff = np.mean(total_scores[c_mask]) - np.mean(total_scores[i_mask])
            alpha[q] = max(0.5, min(2.5, 1.0 + diff * 0.1))
    return alpha, beta

def eap_theta_2pl(responses, alpha, beta):
    n_nodes = 151; theta_nodes = np.linspace(-6, 6, n_nodes)
    prior = np.exp(-0.5 * theta_nodes**2); log_lik = np.zeros(n_nodes)
    for q, ans in enumerate(responses):
        p = np.clip(prob_2pl(theta_nodes, alpha[q], beta[q]), 1e-12, 1.0 - 1e-12)
        log_lik += np.log(p) if ans == 1 else np.log(1.0 - p)
    w = np.exp(log_lik) * prior; den = np.sum(w)
    return np.sum(theta_nodes * w) / den if den > 1e-250 else 0.0

def irt_2pl_calc(matrix_list):
    if not matrix_list or not matrix_list[0]: return []
    matrix = np.array(matrix_list, dtype=int)
    n_s, n_q = matrix.shape
    alpha, beta = estimate_2pl_parameters(matrix)
    thetas = [eap_theta_2pl(matrix[s], alpha, beta) for s in range(n_s)]
    thetas = np.array(thetas)
    mu = np.mean(thetas)
    sigma = np.std(thetas, ddof=0)
    if sigma == 0: sigma = 1.0
    z_scores = (thetas - mu) / sigma
    t_scores = np.clip(50 + 10 * z_scores, 0.0, 100.0)
    return [{"xom": int(np.sum(matrix[s])), "overall": round(t_scores[s], 2)} for s in range(n_s)]

def daraja(v):
    if v is None: return "—"
    if v >= 70: return "A+"
    if v >= 65: return "A"
    if v >= 60: return "B+"
    if v >= 55: return "B"
    if v >= 50: return "C+"
    if v >= 46: return "C"
    return "Failed"

# ── Bot & Dispatcher ──────────────────────────────────────────────────────
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp  = Dispatcher(storage=_storage)

# ── Admin filter ──────────────────────────────────────────────────────────
def AdminFilter(msg: Message) -> bool:
    return is_admin(msg.from_user.id)

# ══════════════════════════════════════════════════════════════════════════
# ADMIN HANDLERS
# ══════════════════════════════════════════════════════════════════════════

@dp.message(Command("start"), F.func(AdminFilter))
async def admin_start(msg: Message, state: FSMContext):
    await state.clear()
    await msg.answer(
        "👋 <b>Admin panelga xush kelibsiz!</b>\n\n"
        "📋 <b>Buyruqlar:</b>\n"
        "/newtest — Yangi test yaratish\n"
        "/tests — Barcha testlar ro'yxati\n"
        "/results KOD — Natijalar + CSV\n"
        "/endtest KOD — Testni yopish\n"
        "/deltest KOD — Testni o'chirish\n"
        "/addadmin ID — Yangi admin qo'shish\n"
        "/admins — Adminlar ro'yxati"
    )

@dp.message(Command("newtest"), F.func(AdminFilter))
async def admin_newtest(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Admin.test_name)
    await msg.answer("📝 <b>Yangi test nomini kiriting:</b>")

@dp.message(Admin.test_name, F.func(AdminFilter))
async def admin_got_name(msg: Message, state: FSMContext):
    await state.update_data(test_name=msg.text.strip())
    await state.set_state(Admin.test_code)
    await msg.answer("🔑 <b>Test kodini kiriting (Masalan: MAT55):</b>")

@dp.message(Admin.test_code, F.func(AdminFilter))
async def admin_got_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    if db_get("SELECT id FROM tests WHERE code=?", (code,)):
        await msg.answer("❌ Bu kod band. Boshqasini kiriting:")
        return
    await state.update_data(test_code=code)
    await state.set_state(Admin.wait_keys)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🧮 Kalitlarni kiritish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer("👇 Pastdagi tugmani bosib, to'g'ri kalitlarni kiritib yuboring:", reply_markup=kb)

@dp.message(Admin.wait_keys, F.web_app_data)
async def admin_save_keys(msg: Message, state: FSMContext):
    data = await state.get_data()
    web_data = json.loads(msg.web_app_data.data)
    final_keys = web_data["closed_answers"] + web_data["open_answers"]
    final_keys = [str(k).strip() for k in final_keys]
    try:
        db_run(
            "INSERT INTO tests (code, name, n_questions, answer_key) VALUES (?, ?, ?, ?)",
            (data["test_code"], data["test_name"], TOTAL_QUESTIONS, json.dumps(final_keys))
        )
    except Exception as e:
        log.error(f"Test saqlashda xato: {e}")
        await msg.answer(
            "❌ Testni saqlashda xatolik yuz berdi. Qaytadan /newtest buyrug'ini bering.",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.clear()
        return

    await state.clear()
    await msg.answer(
        f"✅ <b>Test ishlanishga tayyor</b>\n"
        f"🗒 Test nomi: {data['test_name']}\n"
        f"🔢 Savollar: 55 ta (1–35 ABCD, 36–45 a/b ochiq)\n"
        f"‼️ Test kodi: <code>{data['test_code']}</code>\n\n"
        f"📌 Qatnashish uchun @mirt_2pl_calc_bot ga kirib test kodini yuboring.",
        reply_markup=ReplyKeyboardRemove()
    )

# ── /tests ─────────────────────────────────────────────────────────────────
@dp.message(Command("tests"), F.func(AdminFilter))
async def admin_tests(msg: Message):
    rows = db_get("SELECT code, name, is_active, created_at FROM tests ORDER BY id DESC")
    if not rows:
        await msg.answer("❌ Hech qanday test yo'q.")
        return
    lines = ["📋 <b>Barcha testlar:</b>\n"]
    for r in rows:
        status = "✅ Faol" if r["is_active"] else "🔴 Yopiq"
        count = db_get("SELECT COUNT(*) as c FROM responses WHERE test_id=(SELECT id FROM tests WHERE code=?)", (r["code"],))
        n = count[0]["c"] if count else 0
        lines.append(f"• <code>{r['code']}</code> — {r['name']}\n  {status} | 👥 {n} ta javob | 📅 {r['created_at'][:10]}")
    await msg.answer("\n\n".join(lines))

# ── /results KOD ──────────────────────────────────────────────────────────
@dp.message(Command("results"), F.func(AdminFilter))
async def admin_results(msg: Message):
    parts = msg.text.strip().split()
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /results KOD\nMasalan: /results MAT55")
        return
    code = parts[1].upper()
    tests = db_get("SELECT * FROM tests WHERE code=?", (code,))
    if not tests:
        await msg.answer(f"❌ <code>{code}</code> kodli test topilmadi.")
        return
    test_id   = tests[0]["id"]
    test_name = tests[0]["name"]
    resp = db_get("SELECT full_name, binary_str, submitted_at FROM responses WHERE test_id=? ORDER BY rowid", (test_id,))
    if not resp:
        await msg.answer("❌ Hali hech kim javob topshirmagan.")
        return

    matrix  = [json.loads(r["binary_str"]) for r in resp]
    results = irt_2pl_calc(matrix)

    lines = [f"📊 <b>{test_name}</b> (<code>{code}</code>) — {len(resp)} ta ishtirokchi\n"]
    for i, r in enumerate(resp):
        d = daraja(results[i]["overall"])
        lines.append(f"{i+1}. {r['full_name']} — {results[i]['xom']}/{tests[0]['n_questions']} | {results[i]['overall']} bal | {d}")
    chunk = []
    chunk_len = 0
    for line in lines:
        if chunk_len + len(line) > 3800:
            await msg.answer("\n".join(chunk))
            chunk = []
            chunk_len = 0
        chunk.append(line)
        chunk_len += len(line)
    if chunk:
        await msg.answer("\n".join(chunk))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "Ism Familiya", "To'g'ri javoblar", "Rasch bali", "Daraja", "Sana"])
    for i, r in enumerate(resp):
        writer.writerow([i+1, r["full_name"], results[i]["xom"], results[i]["overall"], daraja(results[i]["overall"]), r["submitted_at"]])
    csv_bytes = buf.getvalue().encode("utf-8-sig")
    await msg.answer_document(
        BufferedInputFile(csv_bytes, filename=f"Natijalar_{code}.csv"),
        caption=f"📎 {test_name} — CSV hisoboti"
    )

# ── /endtest KOD ──────────────────────────────────────────────────────────
@dp.message(Command("endtest"), F.func(AdminFilter))
async def admin_endtest(msg: Message):
    parts = msg.text.strip().split()
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /endtest KOD")
        return
    code = parts[1].upper()
    rows = db_get("SELECT id FROM tests WHERE code=?", (code,))
    if not rows:
        await msg.answer(f"❌ <code>{code}</code> kodli test topilmadi.")
        return
    db_run("UPDATE tests SET is_active=0 WHERE code=?", (code,))
    await msg.answer(f"🔴 <code>{code}</code> testi yopildi. Endi talabalar javob yubora olmaydi.")

# ── /deltest KOD ──────────────────────────────────────────────────────────
@dp.message(Command("deltest"), F.func(AdminFilter))
async def admin_deltest(msg: Message):
    parts = msg.text.strip().split()
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /deltest KOD")
        return
    code = parts[1].upper()
    rows = db_get("SELECT id FROM tests WHERE code=?", (code,))
    if not rows:
        await msg.answer(f"❌ <code>{code}</code> kodli test topilmadi.")
        return
    test_id = rows[0]["id"]
    db_run("DELETE FROM responses WHERE test_id=?", (test_id,))
    db_run("DELETE FROM tests WHERE id=?", (test_id,))
    await msg.answer(f"🗑 <code>{code}</code> testi va barcha javoblari o'chirildi.")

# ── /addadmin ID ──────────────────────────────────────────────────────────
@dp.message(Command("addadmin"), F.func(AdminFilter))
async def admin_addadmin(msg: Message):
    parts = msg.text.strip().split()
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /addadmin TELEGRAM_ID\nMasalan: /addadmin 123456789")
        return
    try:
        new_id = int(parts[1])
    except ValueError:
        await msg.answer("❌ ID raqam bo'lishi kerak.")
        return
    try:
        db_run("INSERT OR IGNORE INTO admins (tg_id) VALUES (?)", (new_id,))
        await msg.answer(f"✅ <code>{new_id}</code> ID li foydalanuvchi admin qilindi.")
    except Exception as e:
        await msg.answer(f"❌ Xatolik: {e}")

# ── /admins ────────────────────────────────────────────────────────────────
@dp.message(Command("admins"), F.func(AdminFilter))
async def admin_admins(msg: Message):
    rows = db_get("SELECT tg_id, added_at FROM admins ORDER BY id")
    if not rows:
        await msg.answer("❌ Adminlar yo'q.")
        return
    lines = ["👥 <b>Adminlar ro'yxati:</b>\n"]
    for i, r in enumerate(rows):
        lines.append(f"{i+1}. <code>{r['tg_id']}</code> — {r['added_at'][:10]}")
    await msg.answer("\n".join(lines))

# ── /excel KOD (eski funksiya saqlanadi) ──────────────────────────────────
@dp.message(Command("excel"), F.func(AdminFilter))
async def export_excel_start(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Admin.excel_code)
    await msg.answer("📊 Qaysi test natijalarini yuklamoqchisiz?\nTest kodini yuboring (Masalan: MAT55):")

@dp.message(Admin.excel_code, F.func(AdminFilter))
async def export_to_excel_process(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    tests = db_get("SELECT * FROM tests WHERE code=?", (code,))
    if not tests:
        await msg.answer("❌ Bunday kodli test topilmadi. Qayta urinib ko'ring:")
        return
    test_id   = tests[0]["id"]
    test_name = tests[0]["name"]
    resp = db_get("SELECT full_name, binary_str FROM responses WHERE test_id=? ORDER BY rowid", (test_id,))
    if not resp:
        await msg.answer("❌ Ushbu testga hali hech kim javob topshirmagan.")
        await state.clear()
        return
    matrix  = [json.loads(r["binary_str"]) for r in resp]
    results = irt_2pl_calc(matrix)
    excel_data = []
    for i, r in enumerate(resp):
        excel_data.append({
            "Nomer": i + 1,
            "Ism familiyasi": r["full_name"],
            "Nechta topgani": results[i]["xom"],
            "Rasch bali": results[i]["overall"],
            "Darajasi": daraja(results[i]["overall"])
        })
    df = pd.DataFrame(excel_data)

    file_path = f"Rasch_Natijalar_{code}_{msg.from_user.id}_{int(time.time()*1_000_000)}.xlsx"

    try:
        df.to_excel(file_path, index=False)
        await msg.answer_document(
            FSInputFile(file_path),
            caption=f"📊 <b>{test_name}</b> ({code}) — Rasch IRT hisoboti."
        )
    except Exception as e:
        log.error(f"Excel yuborishda xato: {e}")
        await msg.answer("❌ Excel faylini tayyorlashda xatolik yuz berdi. Qaytadan urinib ko'ring.")
    finally:
        await state.clear()
        if os.path.exists(file_path):
            os.remove(file_path)

# ══════════════════════════════════════════════════════════════════════════
# STUDENT HANDLERS
# ══════════════════════════════════════════════════════════════════════════

@dp.message(Command("start"))
async def student_start(msg: Message, state: FSMContext):
    if is_admin(msg.from_user.id):
        await admin_start(msg, state)
        return
    await state.clear()
    await state.set_state(Student.enter_code)
    await msg.answer("👋 Xush kelibsiz!\nIltimos, faol <b>Test kodini</b> kiriting:")

@dp.message(Student.enter_code)
async def student_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    rows = db_get("SELECT * FROM tests WHERE code=? AND is_active=1", (code,))
    if not rows:
        await msg.answer("❌ Kod noto'g'ri yoki test yopilgan. Qaytadan urinib ko'ring:")
        return
    t = rows[0]
    if db_get("SELECT id FROM responses WHERE test_id=? AND tg_id=?", (t["id"], msg.from_user.id)):
        await msg.answer("⚠️ Siz bu testni allaqachon topshirib bo'lgansiz.")
        await state.clear()
        return
    await state.update_data(test_id=t["id"], test_name=t["name"], answer_key=t["answer_key"], n_questions=t["n_questions"])
    await state.set_state(Student.enter_name)
    await msg.answer("📝 To'liq ism-familiyangizni kiriting:")

@dp.message(Student.enter_name)
async def student_name(msg: Message, state: FSMContext):
    name = msg.text.strip()
    if len(name) < 4:
        await msg.answer("❌ Iltimos, ism va familiyangizni to'liq kiriting:")
        return
    await state.update_data(full_name=name)
    await state.set_state(Student.wait_answers)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📝 Imtihon Oynasini Ochish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer("✅ Rahmat! 👇 Pastdagi tugmani bosing va imtihonni topshiring:", reply_markup=kb)

@dp.message(Student.wait_answers, F.web_app_data)
async def student_get_answers(msg: Message, state: FSMContext):
    data     = await state.get_data()
    web_data = json.loads(msg.web_app_data.data)

    stud_answers = web_data["closed_answers"] + web_data["open_answers"]
    test_keys    = json.loads(data["answer_key"])
    n_questions  = data["n_questions"]

    binary = []
    correct_count = 0

    for i in range(n_questions):
        s_ans = stud_answers[i].strip() if i < len(stud_answers) else ""
        k_ans = test_keys[i].strip()    if i < len(test_keys)    else ""
        is_correct = False
        if i < CLOSED_COUNT:
            if s_ans.upper() == k_ans.upper() and s_ans not in ["—", ""]:
                is_correct = True
        else:
            if clean_math_expression(s_ans) == clean_math_expression(k_ans) and s_ans != "":
                is_correct = True
        if is_correct:
            binary.append(1); correct_count += 1
        else:
            binary.append(0)

    try:
        db_run(
            "INSERT INTO responses (test_id, tg_id, full_name, raw_answers, binary_str) VALUES (?, ?, ?, ?, ?)",
            (data["test_id"], msg.from_user.id, data["full_name"], json.dumps(stud_answers), json.dumps(binary))
        )
    except sqlite3.IntegrityError:
        await state.clear()
        await msg.answer(
            "⚠️ Siz bu testni allaqachon topshirib bo'lgansiz. Javobingiz oldin saqlangan.",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    except Exception as e:
        log.error(f"Javobni saqlashda xato: {e}")
        await msg.answer(
            "❌ Javobingizni saqlashda xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring "
            "yoki admin bilan bog'laning.",
            reply_markup=ReplyKeyboardRemove()
        )
        return

    await state.clear()
    await msg.answer(
        f"🎉 <b>Tabriklaymiz, {data['full_name']}!</b>\n\n"
        f"✅ To'g'ri javoblar: <b>{correct_count}/{n_questions}</b>\n\n"
        f"📊 Rasch modeli bo'yicha yakuniy balingiz imtihon yakunlangach admin tomonidan e'lon qilinadi.",
        reply_markup=ReplyKeyboardRemove()
    )

# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    db_init()
    log.info("Bot ishga tushdi...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
