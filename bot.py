import asyncio, csv, io, json, logging, math, os, sqlite3
from datetime import datetime
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    WebAppInfo, InlineKeyboardMarkup, InlineKeyboardButton, BufferedInputFile
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOKEN    = "8940355412:AAGgFXKlMt0MSfA0RVYH2gMq0d8C0MX37FU"
ADMIN_ID = 880108541

WEB_APP_URL = "https://boburjonabdullayev.github.io/test-platform2/" 

TOTAL_QUESTIONS = 55
CLOSED_COUNT    = 30

class Admin(StatesGroup):
    test_name   = State()
    test_code   = State()
    wait_keys   = State()

class Student(StatesGroup):
    enter_code    = State()
    enter_name    = State()
    wait_answers  = State()

DB = "rasch_bot.db"

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
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(q, p).fetchall()
    con.close(); return rows

def db_run(q, p=()):
    con = sqlite3.connect(DB)
    try:
        lid = con.execute(q, p).lastrowid
        con.commit()
    except Exception as e:
        con.rollback(); con.close(); raise e
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
    alpha, beta = estimate_2pl_parameters(matrix)
    thetas = [eap_theta_2pl(matrix[s], alpha, beta) for s in range(n_s)]
    thetas = np.array(thetas); mu = np.mean(thetas); sigma = np.std(thetas, ddof=1) if n_s > 1 else 1.0
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

async def auto_rasch(bot: Bot, test_id: int, new_tg_id: int):
    resp = db_get("SELECT tg_id, full_name, binary_str FROM responses WHERE test_id=? ORDER BY rowid", (test_id,))
    if not resp: return
    names = [r["full_name"] for r in resp]; matrix = [json.loads(r["binary_str"]) for r in resp]
    tg_ids = [r["tg_id"] for r in resp]; total = len(matrix)
    idx = next((i for i, tid in enumerate(tg_ids) if tid == new_tg_id), None)
    
    new_name = names[idx] if idx is not None else "Noma'lum"
    new_xom = sum(matrix[idx]) if idx is not None else 0
    results = irt_2pl_calc(matrix)
    new_ov = results[idx]["overall"] if idx is not None else 0

    lines = [
        f"🔔 <b>Yangi ishtirokchi qo'shildi!</b>",
        f"👤 <b>{new_name}</b> | ✅ {new_xom}/55 | T-ball: <b>{new_ov:.2f}</b> {daraja(new_ov)}",
        f"👥 Jami ishtirokchilar: <b>{total}</b> kishi",
        f"📊 <i>Barcha natijalarni va Excel formatni olish uchun:</i> <code>/results {test_id}</code>"
    ]
    await bot.send_message(ADMIN_ID, "\n".join(lines))

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp  = Dispatcher(storage=MemoryStorage())

@dp.message(Command("start"), F.from_user.id == ADMIN_ID)
async def admin_start(msg: Message, state: FSMContext):
    await state.clear()
    await msg.answer("👋 <b>Admin panelga xush kelibsiz!</b>\n\n/newtest — Yangi elektron test yaratish\n/tests — Barcha testlar ro'yxati\n/results [KOD/ID] — Natijalarni Excelda yuklash")

@dp.message(Command("tests"), F.from_user.id == ADMIN_ID)
async def admin_tests(msg: Message):
    rows = db_get("SELECT * FROM tests ORDER BY id DESC")
    if not rows:
        await msg.answer("📭 Hozircha hech qanday test yaratilmagan.")
        return
    text = "📋 <b>Mavjud testlar ro'yxati:</b>\n\n"
    for r in rows:
        status = "🟢 Faol" if r["is_active"] else "🔴 Yakunlangan"
        text += f"🔹 <b>{r['name']}</b> (Kod: <code>{r['code']}</code>) | ID: {r['id']} | {status}\n"
    await msg.answer(text)

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
    code = msg.text.strip().upper().replace(" ","")
    if db_get("SELECT id FROM tests WHERE code=?", (code,)):
        await msg.answer("❌ Bu kod band. Boshqasini kiriting:"); return
    await state.update_data(test_code=code)
    await state.set_state(Admin.wait_keys)
    
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="🧮 Kalitlarni Saytda kiritish", web_app=WebAppInfo(url=WEB_APP_URL))]], resize_keyboard=True)
    await msg.answer("👇 Pastdagi tugmani bosib, ushbu test uchun **to'g'ri kalitlarni** kiritib yuboring:", reply_markup=kb)

@dp.message(Admin.wait_keys, F.web_app_data)
async def admin_save_keys(msg: Message, state: FSMContext):
    data = await state.get_data()
    web_data = json.loads(msg.web_app_data.data)
    
    bot_info = await bot.get_me()
    bot_username = bot_info.username
    
    final_keys = web_data["closed_answers"] + web_data["open_answers"]
    db_run("INSERT INTO tests (code, name, n_questions, answer_key) VALUES (?, ?, 55, ?)", (data["test_code"], data["test_name"], json.dumps(final_keys)))
    await state.clear()
    
    reklama_matni = (
        f"✅ <b>Test ishlanishga tayyor</b>\n"
        f"🗒 <b>Test nomi:</b> {data['test_name']}\n"
        f"🔢 <b>Testlar soni:</b> 55 ta\n"
        f"‼️ <b>Test kodi:</b> <code>{data['test_code']}</code>\n"
        f"👤 <b>Test yaratuvchisi:</b> Abdullayev Boburjon\n\n"
        f"Test javoblaringizni quyidagi botga jo'nating:\n"
        f"👉 @{bot_username}\n\n"
        f"📌 Testda qatnashish uchun @{bot_username} ga kirib test kodini yuboring.\n\n"
        f"♻️ <b>Test ishlanishga tayyor!!!</b>"
    )
    await msg.answer(reklama_matni, reply_markup=ReplyKeyboardRemove())

@dp.message(Command("results"), F.from_user.id == ADMIN_ID)
async def admin_results(msg: Message, command: CommandObject):
    if not command.args:
        await msg.answer("❌ Iltimos test kodini yoki ID sini yozing. Masalan: <code>/results MAT55</code>")
        return
    
    target = command.args.strip().upper()
    test_row = db_get("SELECT * FROM tests WHERE code=? OR id=?", (target, target))
    if not test_row:
        await msg.answer("❌ Bunday test topilmadi.")
        return
    
    t = test_row[0]
    resp = db_get("SELECT full_name, binary_str, submitted_at FROM responses WHERE test_id=? ORDER BY rowid", (t["id"],))
    if not resp:
        await msg.answer("📭 Ushbu testga hali hech kim javob yo'llamadi.")
        return
        
    matrix = [json.loads(r["binary_str"]) for r in resp]
    calc_res = irt_2pl_calc(matrix)
    
    output = io.StringIO()
    output.write('\ufeff') 
    writer = csv.writer(output, delimiter=';')
    writer.writerow(["Nomer", "Ism familiyasi", "Nechta topgani", "Rasch bali", "Darajasi", "Topshirilgan vaqt"])
    
    for i, r in enumerate(resp):
        nomer = i + 1
        name = r["full_name"]
        xom = calc_res[i]["xom"]
        rasch_ball = calc_res[i]["overall"]
        drj = daraja(rasch_ball)
        s_time = r["submitted_at"]
        writer.writerow([nomer, name, f"{xom}/55", rasch_ball, drj, s_time])
        
    file_data = output.getvalue().encode('utf-8')
    output.close()
    
    input_file = BufferedInputFile(file_data, filename=f"Natijalar_{t['code']}.csv")
    await msg.answer_document(input_file, caption=f"📊 <b>{t['name']}</b> testi bo'yicha yakuniy hisobot (Rasch 2PL).")

@dp.message(Command("start"), F.from_user.id != ADMIN_ID)
async def student_start(msg: Message, state: FSMContext):
    await state.clear(); await state.set_state(Student.enter_code)
    await msg.answer("👋 Xush kelibsiz! Iltimos, faol **Test kodini** kiriting:")

@dp.message(Student.enter_code)
async def student_code(msg: Message, state: FSMContext):
    code = msg.text.strip().upper()
    rows = db_get("SELECT * FROM tests WHERE code=? AND is_active=1", (code,))
    if not rows: await msg.answer("❌ Kod noto'g'ri yoki test yakunlangan."); return
    t = rows[0]
    if db_get("SELECT id FROM responses WHERE test_id=? AND tg_id=?", (t["id"], msg.from_user.id)):
        await msg.answer("⚠️ Siz bu testni topshirib bo'lgansiz."); return
    await state.update_data(test_id=t["id"], test_name=t["name"], answer_key=t["answer_key"])
    await state.set_state(Student.enter_name)
    await msg.answer("📝 To'liq ism-familiyangizni kiriting:")

@dp.message(Student.enter_name)
async def student_name(msg: Message, state: FSMContext):
    name = msg.text.strip()
    if len(name) < 4: 
        await msg.answer("❌ Iltimos ism familiyangizni to'liq kiriting:")
        return
    await state.update_data(full_name=name)
    await state.set_state(Student.wait_answers)
    
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="📝 Imtihon Oynasini Ochish", web_app=WebAppInfo(url=WEB_APP_URL))]], resize_keyboard=True)
    await msg.answer(f"✅ Rahmat, {name}! Hamma narsa tayyor.\n\n👇 Pastdagi tugmani bosing va imtihonni topshiring:", reply_markup=kb)

@dp.message(Student.wait_answers, F.web_app_data)
async def student_get_answers(msg: Message, state: FSMContext):
    data = await state.get_data()
    web_data = json.loads(msg.web_app_data.data)
    
    stud_answers = web_data["closed_answers"] + web_data["open_answers"]
    test_keys = json.loads(data["answer_key"])
    
    binary = []
    xom_ball = 0
    for i in range(55):
        s_ans = stud_answers[i].strip()
        k_ans = test_keys[i].strip()
        if i < CLOSED_COUNT:
            is_correct = 1 if s_ans.upper() == k_ans.upper() else 0
        else:
            is_correct = 1 if s_ans.lower() == k_ans.lower() else 0
        binary.append(is_correct)
        if is_correct == 1:
            xom_ball += 1
            
    db_run("INSERT OR IGNORE INTO responses (test_id, tg_id, full_name, raw_answers, binary_str) VALUES (?, ?, ?, ?, ?)",
           (data["test_id"], msg.from_user.id, data["full_name"], json.dumps(stud_answers), json.dumps(binary)))
    
    await state.clear()
    
    success_msg = (
        f"🎉 <b>Tabriklaymiz, siz {xom_ball}/55 ta savolga to'g'ri javob topdingiz!</b>\n\n"
        f"📊 Rasch model (2PL IRT) bo'yicha yakuniy aniqlashtirilgan balingiz va darajangiz "
        f"imtihon to'liq yakunlangandan so'ng Admin tomonidan e'lon qilinadi."
    )
    await msg.answer(success_msg, reply_markup=ReplyKeyboardRemove())
    
    await auto_rasch(bot, data["test_id"], msg.from_user.id)

async def main():
    db_init()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
