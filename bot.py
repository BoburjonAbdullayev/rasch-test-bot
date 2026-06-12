import asyncio
import json
import logging
import math
import os
import sqlite3
import numpy as np
import pandas as pd

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    WebAppInfo, FSInputFile
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOKEN = os.environ.get("BOT_TOKEN")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "880108541"))
WEB_APP_URL = "https://boburjonabdullayev.github.io/test-platform2/"
TOTAL_QUESTIONS = 55
CLOSED_COUNT = 35
DB = "/data/rasch_bot.db"

class S1(StatesGroup):
    s1 = State()
    s2 = State()
    s3 = State()
    s4 = State()

class S2(StatesGroup):
    s1 = State()
    s2 = State()
    s3 = State()

def clean_math_expression(expr):
    if not expr: return ""
    s = str(expr).strip().lower()
    s = s.replace(" ", "")
    s = s.replace(r"\cdot", "").replace("*", "").replace(r"\times", "")
    s = s.replace("{", "").replace("}", "").replace("(", "").replace(")", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace(r"\frac", "")
    return s

def db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
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

    row_sums = matrix.sum(axis=1)
    normal_mask = (row_sums > 0) & (row_sums < n_q)
    normal_idx  = np.where(normal_mask)[0]

    results = [None] * n_s

    for s in range(n_s):
        if row_sums[s] == 0:
            results[s] = {"xom": 0, "overall": 0.0}
        elif row_sums[s] == n_q:
            results[s] = {"xom": int(n_q), "overall": 100.0}

    if len(normal_idx) == 0:
        for s in range(n_s):
            if results[s] is None:
                results[s] = {"xom": int(row_sums[s]), "overall": 50.0}
        return results

    if len(normal_idx) == 1:
        s = normal_idx[0]
        t = round(row_sums[s] / n_q * 100, 2)
        results[s] = {"xom": int(row_sums[s]), "overall": float(np.clip(t, 0, 100))}
        return results

    normal_matrix = matrix[normal_idx]
    alpha, beta   = estimate_2pl_parameters(normal_matrix)
    thetas = np.array([eap_theta_2pl(normal_matrix[i], alpha, beta) for i in range(len(normal_idx))])

    mu    = np.mean(thetas)
    sigma = np.std(thetas, ddof=0)
    if sigma < 1e-9: sigma = 1.0

    z_scores = (thetas - mu) / sigma
    t_scores = np.clip(50 + 10 * z_scores, 0.0, 100.0)

    for i, s in enumerate(normal_idx):
        results[s] = {"xom": int(row_sums[s]), "overall": round(float(t_scores[i]), 2)}

    return results

def get_grade(v):
    if v is None: return "—"
    if v >= 70: return "A+"
    if v >= 65: return "A"
    if v >= 60: return "B+"
    if v >= 55: return "B"
    if v >= 50: return "C+"
    if v >= 46: return "C"
    return "F"

storage = RedisStorage.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp  = Dispatcher(storage=storage)

@dp.message(Command("start"), F.from_user.id == ADMIN_ID)
async def admin_start(msg: Message, state: FSMContext):
    await state.clear()
    await msg.answer(
        "👋 <b>Muloqot boshlandi</b>\n\n"
        "/newtest — Yangi element yaratish\n"
        "/excel — Faylni yuklab olish"
    )

@dp.message(Command("newtest"), F.from_user.id == ADMIN_ID)
async def admin_newtest(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(S1.s1)
    await msg.answer("📝 <b>Nom kiriting:</b>")

@dp.message(S1.s1, F.from_user.id == ADMIN_ID)
async def admin_got_name(msg: Message, state: FSMContext):
    await state.update_data(test_name=msg.text.strip())
    await state.set_state(S1.s2)
    await msg.answer("🔑 <b>Kod kiriting:</b>")

@dp.message(S1.s2, F.from_user.id == ADMIN_ID)
async def admin_got_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    if db_get("SELECT id FROM tests WHERE code=?", (code,)):
        await msg.answer("❌ Ushbu kod band. Boshqasini kiriting:"); return
    await state.update_data(test_code=code)
    await state.set_state(S1.s3)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="➡️ Ma'lumotlarni kiritish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer(
        "👇 Quyidagi tugma orqali ma'lumotlarni yuboring:",
        reply_markup=kb
    )

@dp.message(S1.s3, F.web_app_data, F.from_user.id == ADMIN_ID)
async def admin_save_keys(msg: Message, state: FSMContext):
    try:
        data       = await state.get_data()
        web_data   = json.loads(msg.web_app_data.data)
        final_keys = web_data["closed_answers"] + web_data["open_answers"]
        final_keys = [str(k).strip() for k in final_keys]
        db_run(
            "INSERT INTO tests (code, name, n_questions, answer_key) VALUES (?, ?, 55, ?)",
            (data["test_code"], data["test_name"], json.dumps(final_keys))
        )
        await state.clear()
        await msg.answer(
            f"✅ <b>Jarayon yakunlandi</b>\n"
            f"🗒 Nomi: {data['test_name']}\n"
            f"🔢 Miqdori: 55 ta\n"
            f"‼️ Kodi: <code>{data['test_code']}</code>\n\n"
            f"Tizimdan foydalanish uchun kodni yuborish kifoya.",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        log.error(f"Error: {e}")
        await msg.answer("❌ Xatolik yuz berdi. Qayta urinib ko'ring.")
        await state.clear()

@dp.message(Command("excel"), F.from_user.id == ADMIN_ID)
async def export_excel_start(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(S1.s4)
    await msg.answer("📊 <b>Kod kiriting:</b>")

@dp.message(S1.s4, F.from_user.id == ADMIN_ID)
async def export_to_excel_process(msg: Message, state: FSMContext):
    try:
        code  = msg.text.strip().upper().replace(" ", "")
        tests = db_get("SELECT * FROM tests WHERE code=?", (code,))
        if not tests:
            await msg.answer("❌ Kod topilmadi. Qayta urinib ko'ring:"); return

        test_id   = tests[0]["id"]
        test_name = tests[0]["name"]
        resp      = db_get(
            "SELECT full_name, binary_str FROM responses WHERE test_id=? ORDER BY rowid",
            (test_id,)
        )
        if not resp:
            await msg.answer("❌ Ma'lumot topilmadi.")
            await state.clear(); return

        matrix  = [json.loads(r["binary_str"]) for r in resp]
        results = irt_2pl_calc(matrix)

        excel_data = []
        for i, r in enumerate(resp):
            excel_data.append({
                "ID":             i + 1,
                "F.I.O":          r["full_name"],
                "Natija (Xom)":   results[i]["xom"],
                "Natija (Baho)":  results[i]["overall"],
                "Daraja":         get_grade(results[i]["overall"])
            })

        df        = pd.DataFrame(excel_data)
        file_path = f"/tmp/Report_{code}.xlsx"
        df.to_excel(file_path, index=False)
        await msg.answer_document(
            FSInputFile(file_path),
            caption=f"📊 <b>{test_name}</b> ({code}) hisoboti."
        )
        await state.clear()
        if os.path.exists(file_path): os.remove(file_path)

    except Exception as e:
        log.error(f"Error: {e}")
        await msg.answer("❌ Amaliyot bajarilmadi.")
        await state.clear()

@dp.message(Command("start"), F.from_user.id != ADMIN_ID)
async def student_start(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(S2.s1)
    await msg.answer("👋 Xush kelibsiz! Iltimos, faol <b>Kodni</b> kiriting:")

@dp.message(S2.s1)
async def student_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper().replace(" ", "")
    rows = db_get("SELECT * FROM tests WHERE code=? AND is_active=1", (code,))
    if (!rows):
        await msg.answer("❌ Noto'g'ri kod. Qayta urinib ko'ring:"); return
    t = rows[0]
    if db_get("SELECT id FROM responses WHERE test_id=? AND tg_id=?", (t["id"], msg.from_user.id)):
        await msg.answer("⚠️ Ushbu bo'limdan avval foydalanilgan.")
        await state.clear(); return
    await state.update_data(test_id=t["id"], test_name=t["name"], answer_key=t["answer_key"])
    await state.set_state(S2.s2)
    await msg.answer("📝 Ism va familiyangizni kiriting:")

@dp.message(S2.s2)
async def student_name(msg: Message, state: FSMContext):
    name = msg.text.strip()
    if len(name) < 4:
        await msg.answer("❌ Ma'lumotni to'liq kiriting:"); return
    await state.update_data(full_name=name)
    await state.set_state(S2.s3)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📝 Oynani ochish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer("✅ Qabul qilindi. 👇 Pastdagi tugma orqali davom eting:", reply_markup=kb)

@dp.message(S2.s3, F.web_app_data)
async def student_get_answers(msg: Message, state: FSMContext):
    try:
        data         = await state.get_data()
        web_data     = json.loads(msg.web_app_data.data)
        stud_answers = web_data["closed_answers"] + web_data["open_answers"]
        test_keys    = json.loads(data["answer_key"])

        binary = []; correct_count = 0

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
            "INSERT OR IGNORE INTO responses "
            "(test_id, tg_id, full_name, raw_answers, binary_str) VALUES (?, ?, ?, ?, ?)",
            (data["test_id"], msg.from_user.id, data["full_name"],
             json.dumps(stud_answers), json.dumps(binary))
        )
        await state.clear()
        await msg.answer(
            f"🎉 <b>Ma'lumotlar muvaffaqiyatli qabul qilindi.</b>\n\n"
            f"Natijangiz: {correct_count}/55\n"
            f"Yakuniy hisobotlar keyinchalik e'lon qilinadi.",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        log.error(f"Error: {e}")
        await msg.answer("❌ Xatolik yuz berdi. Qayta urinib ko'ring.")
        await state.clear()

async def main():
    db_init()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
