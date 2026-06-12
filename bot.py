import asyncio
import json
import logging
import math
import os
import sqlite3
import re
import numpy as np
import pandas as pd

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage  # FIX 1: MemoryStorage o'rniga Redis
from aiogram.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    WebAppInfo, FSInputFile
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOKEN    = os.environ.get("BOT_TOKEN")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "880108541"))  # FIX 2: env dan olish
WEB_APP_URL      = "https://boburjonabdullayev.github.io/test-platform2/"
TOTAL_QUESTIONS  = 55
CLOSED_COUNT     = 35
DB               = "rasch_bot.db"

class Admin(StatesGroup):
    test_name  = State()
    test_code  = State()
    wait_keys  = State()
    excel_code = State()

class Student(StatesGroup):
    enter_code  = State()
    enter_name  = State()
    wait_answers = State()

# --- MATEMATIK BILDIRGICH / NORMALIZATOR ---
def clean_math_expression(expr):
    if not expr: return ""
    s = str(expr).strip().lower()
    s = s.replace(" ", "")
    s = s.replace(r"\cdot", "").replace("*", "").replace(r"\times", "")
    s = s.replace("{", "").replace("}", "").replace("(", "").replace(")", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace(r"\frac", "")
    return s

# --- BAZA INTEGRATSIYASI ---
def db_init():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS tests (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        code         TEXT NOT NULL UNIQUE,
        name         TEXT NOT NULL,
        n_questions  INTEGER NOT NULL DEFAULT 55,
        answer_key   TEXT NOT NULL,
        is_active    INTEGER DEFAULT 1,
        created_at   TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE TABLE IF NOT EXISTS responses (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id      INTEGER NOT NULL,
        tg_id        INTEGER NOT NULL,
        full_name    TEXT NOT NULL,
        raw_answers  TEXT NOT NULL,
        binary_str   TEXT NOT NULL,
        submitted_at TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(test_id, tg_id)
    );
    """)
    con.commit(); con.close()

def db_get(q, p=()):
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = con.execute(q, p).fetchall(); con.close(); return rows

def db_run(q, p=()):
    con = sqlite3.connect(DB)
    try: lid = con.execute(q, p).lastrowid; con.commit()
    except Exception as e: con.rollback(); con.close(); raise e
    con.close(); return lid

# --- IRT 2PL ALGORITMI (O'ZGARTIRILMAGAN) ---
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

# FIX 1: RedisStorage
storage = RedisStorage.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp  = Dispatcher(storage=storage)

# --- ADMIN PROCESS ---
@dp.message(Command("start"), F.from_user.id == ADMIN_ID)
async def admin_start(msg: Message, state: FSMContext):
    await state.clear()
    await msg.answer("👋 <b>Admin panelga xush kelibsiz!</b>\n\n/newtest — Yangi elektron test yaratish\n/excel — Excel formatda hisobot yuklab olish")

@dp.message(Command("newtest"), F.from_user.id == ADMIN_ID)
async def admin_newtest(msg: Message, state: FSMContext):
    await state.clear(); await state.set_state(Admin.test_name)
    await msg.answer("📝 <b>Yangi test nomini kiriting:</b>")

@dp.message(Admin.test_name, F.from_user.id == ADMIN_ID)
async def admin_got_name(msg: Message, state: FSMContext):
    await state.update_data(test_name=msg.text.strip())
    await state.set_state(Admin.test_code)
    await msg.answer("🔑 <b>Test kodini kiriting (Masalan: MAT55):</b>")

@dp.message(Admin.test_code, F.from_user.id == ADMIN_ID)
async def admin_got_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    if db_get("SELECT id FROM tests WHERE code=?", (code,)):
        await msg.answer("❌ Bu kod band. Boshqasini kiriting:"); return
    await state.update_data(test_code=code)
    await state.set_state(Admin.wait_keys)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🧮 Kalitlarni Saytda kiritish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer("👇 Pastdagi tugmani bosib, ushbu test uchun <b>to'g'ri kalitlarni</b> kiritib yuboring:", reply_markup=kb)

@dp.message(Admin.wait_keys, F.web_app_data, F.from_user.id == ADMIN_ID)
async def admin_save_keys(msg: Message, state: FSMContext):
    # FIX 3: try/except
    try:
        data     = await state.get_data()
        web_data = json.loads(msg.web_app_data.data)
        final_keys = web_data["closed_answers"] + web_data["open_answers"]
        final_keys = [str(k).strip() for k in final_keys]
        db_run(
            "INSERT INTO tests (code, name, n_questions, answer_key) VALUES (?, ?, 55, ?)",
            (data["test_code"], data["test_name"], json.dumps(final_keys))
        )
        await state.clear()
        promo_text = (
            f"✅ <b>Test ishlanishga tayyor</b>\n"
            f"🗒 Test nomi: {data['test_name']}\n"
            f"🔢 Testlar soni: 55 ta (1-35 ABCD, 36-45 a/b ochiq)\n"
            f"‼️ Test kodi: <code>{data['test_code']}</code>\n"
            f"👤 Test yaratuvchisi: Abdullayev Boburjon\n\n"
            f"Test javoblaringizni quyidagi botga jo'nating:\n"
            f"👉 @mirt_2pl_calc_bot\n\n"
            f"📌 Testda qatnashish uchun botga kirib test kodini yuboring."
        )
        await msg.answer(promo_text, reply_markup=ReplyKeyboardRemove())
    except Exception as e:
        log.error(f"admin_save_keys xatosi: {e}")
        await msg.answer("❌ Kalitlarni saqlashda xatolik yuz berdi. Qayta urinib ko'ring.")
        await state.clear()

# --- EXCEL EXPORT ---
@dp.message(Command("excel"), F.from_user.id == ADMIN_ID)
async def export_excel_start(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Admin.excel_code)
    await msg.answer("📊 <b>Qaysi test natijalarini yuklamoqchisiz?</b>\nIltimos, test kodini yuboring (Masalan: MAT55):")

@dp.message(Admin.excel_code, F.from_user.id == ADMIN_ID)
async def export_to_excel_process(msg: Message, state: FSMContext):
    # FIX 3: try/except
    try:
        code  = msg.text.strip().upper().replace(" ", "")
        tests = db_get("SELECT * FROM tests WHERE code=?", (code,))
        if not tests:
            await msg.answer("❌ Bunday kodli test topilmadi. Qayta urinib ko'ring:"); return

        test_id   = tests[0]["id"]
        test_name = tests[0]["name"]
        resp      = db_get("SELECT full_name, binary_str FROM responses WHERE test_id=? ORDER BY rowid", (test_id,))
        if not resp:
            await msg.answer("❌ Ushbu testga hali hech kim javob topshirmagan.")
            await state.clear(); return

        matrix  = [json.loads(r["binary_str"]) for r in resp]
        results = irt_2pl_calc(matrix)

        excel_data = []
        for i, r in enumerate(resp):
            excel_data.append({
                "Nomer":           i + 1,
                "Ism familiyasi":  r["full_name"],
                "Nechta topgani":  results[i]["xom"],
                "Rasch bali":      results[i]["overall"],
                "Darajasi":        daraja(results[i]["overall"])
            })

        df = pd.DataFrame(excel_data)
        # FIX 4: /tmp/ papkasiga yozish (Railway uchun)
        file_path = f"/tmp/Rasch_Natijalar_{code}.xlsx"
        df.to_excel(file_path, index=False)
        await msg.answer_document(
            FSInputFile(file_path),
            caption=f"📊 <b>{test_name}</b> ({code}) testi bo'yicha IRT 2PL hisoboti."
        )
        await state.clear()
        if os.path.exists(file_path): os.remove(file_path)

    except Exception as e:
        log.error(f"export_to_excel xatosi: {e}")
        await msg.answer("❌ Excel yaratishda xatolik yuz berdi.")
        await state.clear()

# --- STUDENT PROCESS ---
@dp.message(Command("start"), F.from_user.id != ADMIN_ID)
async def student_start(msg: Message, state: FSMContext):
    await state.clear(); await state.set_state(Student.enter_code)
    await msg.answer("👋 Xush kelibsiz! Iltimos, faol <b>Test kodini</b> kiriting:")

@dp.message(Student.enter_code)
async def student_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    rows = db_get("SELECT * FROM tests WHERE code=? AND is_active=1", (code,))
    if not rows:
        await msg.answer("❌ Kod noto'g'ri yoki faol emas. Qaytadan urinib ko'ring:"); return
    t = rows[0]
    if db_get("SELECT id FROM responses WHERE test_id=? AND tg_id=?", (t["id"], msg.from_user.id)):
        await msg.answer("⚠️ Siz bu testni topshirib bo'lgansiz."); await state.clear(); return
    await state.update_data(test_id=t["id"], test_name=t["name"], answer_key=t["answer_key"])
    await state.set_state(Student.enter_name)
    await msg.answer("📝 To'liq ism-familiyangizni kiriting:")

@dp.message(Student.enter_name)
async def student_name(msg: Message, state: FSMContext):
    name = msg.text.strip()
    if len(name) < 4:
        await msg.answer("❌ Iltimos, ism va familiyangizni to'liq kiriting:"); return
    await state.update_data(full_name=name)
    await state.set_state(Student.wait_answers)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📝 Imtihon Oynasini Ochish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer("✅ Rahmat! 👇 Pastdagi tugmani bosing va imtihonni topshiring:", reply_markup=kb)

@dp.message(Student.wait_answers, F.web_app_data)
async def student_get_answers(msg: Message, state: FSMContext):
    # FIX 3: try/except
    try:
        data      = await state.get_data()
        web_data  = json.loads(msg.web_app_data.data)

        stud_answers = web_data["closed_answers"] + web_data["open_answers"]
        test_keys    = json.loads(data["answer_key"])

        binary        = []
        correct_count = 0

        for i in range(55):
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

        db_run(
            "INSERT OR IGNORE INTO responses (test_id, tg_id, full_name, raw_answers, binary_str) VALUES (?, ?, ?, ?, ?)",
            (data["test_id"], msg.from_user.id, data["full_name"], json.dumps(stud_answers), json.dumps(binary))
        )
        await state.clear()
        await msg.answer(
            f"🎉 <b>Tabriklaymiz siz {correct_count}/55 ta to'g'ri topdingiz.</b>\n\n"
            f"Rasch modeli bo'yicha yakuniy balingiz va darajangiz imtihon yakunlangach admin tomonidan e'lon qilinadi.",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        log.error(f"student_get_answers xatosi: {e}")
        await msg.answer("❌ Javoblarni qabul qilishda xatolik yuz berdi. Qayta urinib ko'ring.")
        await state.clear()

async def main():
    db_init()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
