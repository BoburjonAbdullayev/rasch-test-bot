

import asyncio, csv, io, json, logging, math, os, sqlite3
from datetime import datetime
from typing import Any
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    WebAppInfo, InlineKeyboardMarkup, InlineKeyboardButton,
    BufferedInputFile
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  1. MUHIT O'ZGARUVCHILARI
# ═══════════════════════════════════════════════════
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN environment variable topilmadi!")

_admin_ids_raw = os.getenv("ADMIN_IDS", os.getenv("ADMIN_ID", ""))
ADMIN_IDS: set[int] = set()
for _aid in _admin_ids_raw.split(","):
    _aid = _aid.strip()
    if _aid.isdigit():
        ADMIN_IDS.add(int(_aid))

if not ADMIN_IDS:
    raise ValueError("ADMIN_IDS (yoki ADMIN_ID) environment variable topilmadi!")

log.info(f"Admin IDlar: {ADMIN_IDS}")

WEB_APP_URL     = os.getenv("WEB_APP_URL", "https://boburjonabdullayev.github.io/test-platform2/")
TOTAL_QUESTIONS = 55
CLOSED_COUNT    = 30

# ═══════════════════════════════════════════════════
#  2. DATABASE — /data/ papkasida (Railway Volume)
# ═══════════════════════════════════════════════════
DATA_DIR = os.getenv("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)
DB = os.path.join(DATA_DIR, "rasch_bot.db")
log.info(f"Database yo'li: {DB}")


def db_get(q: str, p: tuple = ()) -> list:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(q, p).fetchall()
        return rows
    finally:
        con.close()


def db_run(q: str, p: tuple = ()) -> int:
    con = sqlite3.connect(DB)
    try:
        lid = con.execute(q, p).lastrowid
        con.commit()
        return lid
    except Exception as e:
        con.rollback()
        raise e
    finally:
        con.close()


def db_init():
    con = sqlite3.connect(DB)
    try:
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
            t_score      REAL    DEFAULT NULL,
            xom_ball     INTEGER DEFAULT NULL,
            submitted_at TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(test_id, tg_id)
        );
        CREATE TABLE IF NOT EXISTS admins (
            tg_id    INTEGER PRIMARY KEY,
            added_at TEXT DEFAULT (datetime('now','localtime'))
        );
        """)
        for aid in ADMIN_IDS:
            con.execute("INSERT OR IGNORE INTO admins (tg_id) VALUES (?)", (aid,))
        con.commit()
    finally:
        con.close()


def is_admin(tg_id: int) -> bool:
    if tg_id in ADMIN_IDS:
        return True
    return len(db_get("SELECT tg_id FROM admins WHERE tg_id=?", (tg_id,))) > 0


# ═══════════════════════════════════════════════════
#  FSM STATES
# ═══════════════════════════════════════════════════
class Admin(StatesGroup):
    test_name = State()
    test_code = State()
    wait_keys = State()

class Student(StatesGroup):
    enter_code   = State()
    enter_name   = State()
    wait_answers = State()


# ═══════════════════════════════════════════════════
#  3. 2PL IRT MODELI
# ═══════════════════════════════════════════════════
def prob_2pl(theta, alpha, beta):
    return 1.0 / (1.0 + np.exp(-alpha * (theta - beta)))


def estimate_2pl_parameters(matrix: np.ndarray):
    n_s, n_q     = matrix.shape
    alpha        = np.ones(n_q)
    beta         = np.zeros(n_q)
    total_scores = matrix.sum(axis=1)
    for q in range(n_q):
        correct = np.sum(matrix[:, q])
        p       = max(0.01, min(0.99, correct / n_s))
        beta[q] = -math.log(p / (1.0 - p))
        c_mask  = (matrix[:, q] == 1)
        i_mask  = (matrix[:, q] == 0)
        if np.any(c_mask) and np.any(i_mask):
            diff     = np.mean(total_scores[c_mask]) - np.mean(total_scores[i_mask])
            alpha[q] = max(0.5, min(2.5, 1.0 + diff * 0.1))
    return alpha, beta


def eap_theta_2pl(responses, alpha, beta):
    n_nodes     = 151
    theta_nodes = np.linspace(-6, 6, n_nodes)
    prior       = np.exp(-0.5 * theta_nodes**2)
    log_lik     = np.zeros(n_nodes)
    for q, ans in enumerate(responses):
        p = np.clip(prob_2pl(theta_nodes, alpha[q], beta[q]), 1e-12, 1.0 - 1e-12)
        log_lik += np.log(p) if ans == 1 else np.log(1.0 - p)
    w   = np.exp(log_lik) * prior
    den = np.sum(w)
    return float(np.sum(theta_nodes * w) / den) if den > 1e-250 else 0.0


def irt_2pl_calc(matrix_list: list) -> list:
    if not matrix_list or not matrix_list[0]:
        return []
    matrix      = np.array(matrix_list, dtype=int)
    n_s, _      = matrix.shape
    alpha, beta = estimate_2pl_parameters(matrix)
    thetas      = np.array([eap_theta_2pl(matrix[s], alpha, beta) for s in range(n_s)])
    mu          = float(np.mean(thetas))
    sigma       = float(np.std(thetas, ddof=1)) if n_s > 1 else 1.0
    if sigma < 1e-9:
        sigma = 1.0
    z_scores = (thetas - mu) / sigma
    t_scores = np.clip(50 + 10 * z_scores, 0.0, 100.0)
    return [{"xom": int(np.sum(matrix[s])), "overall": round(float(t_scores[s]), 2)} for s in range(n_s)]


def daraja(v) -> str:
    if v is None: return "—"
    if v >= 70:   return "A+"
    if v >= 65:   return "A"
    if v >= 60:   return "B+"
    if v >= 55:   return "B"
    if v >= 50:   return "C+"
    if v >= 46:   return "C"
    return "Failed"


def daraja_emoji(v) -> str:
    return {"A+": "🏆", "A": "🥇", "B+": "🥈", "B": "🥉",
            "C+": "✅", "C": "📘", "Failed": "❌"}.get(daraja(v), "")


# ═══════════════════════════════════════════════════
#  4. AVTOMATIK RASCH + NATIJANI O'QUVCHIGA YUBORISH
# ═══════════════════════════════════════════════════
async def auto_rasch(bot: Bot, test_id: int, new_tg_id: int, new_name: str):
    resp = db_get(
        "SELECT tg_id, full_name, binary_str FROM responses WHERE test_id=? ORDER BY rowid",
        (test_id,)
    )
    if not resp:
        return

    names   = [r["full_name"]              for r in resp]
    matrix  = [json.loads(r["binary_str"]) for r in resp]
    tg_ids  = [r["tg_id"]                  for r in resp]
    total   = len(matrix)

    idx     = next((i for i, tid in enumerate(tg_ids) if tid == new_tg_id), None)
    results = irt_2pl_calc(matrix)

    if idx is None or not results:
        return

    new_xom   = results[idx]["xom"]
    new_ov    = results[idx]["overall"]
    new_d     = daraja(new_ov)
    new_emoji = daraja_emoji(new_ov)

    # Barcha natijalarni DB ga yangilash
    for i, r in enumerate(results):
        db_run(
            "UPDATE responses SET t_score=?, xom_ball=? WHERE test_id=? AND tg_id=?",
            (r["overall"], r["xom"], test_id, tg_ids[i])
        )

    # ✅ FIX: Reyting — bir xil ball bo'lsa ham to'g'ri ishlaydi
    # enumerate bilan indeks asosida hisoblash
    scores_with_idx = [(r["overall"], i) for i, r in enumerate(results)]
    scores_with_idx.sort(key=lambda x: x[0], reverse=True)
    rank = next((pos + 1 for pos, (sc, i) in enumerate(scores_with_idx) if i == idx), total)

    # O'quvchiga natija
    student_msg = (
        f"🎉 <b>Natijangiz tayyor!</b>\n\n"
        f"👤 <b>{new_name}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"✅ Xom ball:   <b>{new_xom}/{TOTAL_QUESTIONS}</b>\n"
        f"📊 T-ball:     <b>{new_ov:.2f}</b>\n"
        f"🎯 Daraja:     <b>{new_emoji} {new_d}</b>\n"
        f"🏅 Reyting:    <b>{rank}-o'rin</b> ({total} ishtirokchi ichida)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
    )
    motivations = {
        "Failed": "💪 Keyingi safar ko'proq tayyorlanib keling!",
        "C":      "📚 Yaxshi natija! Yana biroz mashq qiling.",
        "C+":     "📚 Yaxshi natija! Yana biroz mashq qiling.",
        "B":      "👏 Zo'r natija! Davom eting!",
        "B+":     "👏 Zo'r natija! Davom eting!",
        "A":      "🌟 A'lo natija! Tabriklaymiz!",
        "A+":     "🏆 Mukammal natija! Siz eng yaxshilardansiz!",
    }
    student_msg += motivations.get(new_d, "")

    try:
        await bot.send_message(new_tg_id, student_msg)
    except Exception as e:
        log.warning(f"O'quvchiga xabar yuborib bo'lmadi ({new_tg_id}): {e}")

    # Adminga xabar
    top3     = sorted(zip(names, [r["overall"] for r in results]), key=lambda x: x[1], reverse=True)[:3]
    medals   = ["🥇", "🥈", "🥉"]
    top3_txt = "\n".join(f"  {medals[i]} {nm} — {sc:.2f} ({daraja(sc)})" for i, (nm, sc) in enumerate(top3))

    admin_msg = (
        f"🔔 <b>Yangi natija!</b>\n\n"
        f"👤 {new_name}\n"
        f"✅ {new_xom}/{TOTAL_QUESTIONS} | T-ball: <b>{new_ov:.2f}</b> {new_emoji} {new_d}\n"
        f"🏅 {rank}-o'rin | 👥 Jami: <b>{total}</b> kishi\n\n"
        f"🏆 <b>Top 3:</b>\n{top3_txt}"
    )
    for aid in ADMIN_IDS:
        try:
            await bot.send_message(aid, admin_msg)
        except Exception as e:
            log.warning(f"Adminga xabar yuborib bo'lmadi ({aid}): {e}")


# ═══════════════════════════════════════════════════
#  BOT VA DISPATCHER
# ═══════════════════════════════════════════════════
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp  = Dispatcher(storage=MemoryStorage())


# ═══════════════════════════════════════════════════
#  ADMIN BUYRUQLARI
# ═══════════════════════════════════════════════════
@dp.message(Command("start"))
async def cmd_start(msg: Message, state: FSMContext):
    await state.clear()
    if is_admin(msg.from_user.id):
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
    else:
        await state.set_state(Student.enter_code)
        await msg.answer("👋 <b>Xush kelibsiz!</b>\n\nIltimos, faol <b>Test kodini</b> kiriting:")


@dp.message(Command("newtest"))
async def admin_newtest(msg: Message, state: FSMContext):
    if not is_admin(msg.from_user.id):
        return
    await state.clear()
    await state.set_state(Admin.test_name)
    await msg.answer("📝 <b>Yangi test nomini kiriting:</b>")


@dp.message(Admin.test_name)
async def admin_got_name(msg: Message, state: FSMContext):
    if not is_admin(msg.from_user.id):
        return
    await state.update_data(test_name=msg.text.strip())
    await state.set_state(Admin.test_code)
    await msg.answer("🔑 <b>Test kodini kiriting</b> (masalan: MAT55, FIZ2026):")


@dp.message(Admin.test_code)
async def admin_got_code(msg: Message, state: FSMContext):
    if not is_admin(msg.from_user.id):
        return
    code = msg.text.strip().upper().replace(" ", "")
    if db_get("SELECT id FROM tests WHERE code=?", (code,)):
        await msg.answer("❌ Bu kod band. Boshqa kod kiriting:")
        return
    await state.update_data(test_code=code)
    await state.set_state(Admin.wait_keys)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🧮 Kalitlarni Saytda kiritish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer("👇 Pastdagi tugmani bosib, to'g'ri kalitlarni kiriting:", reply_markup=kb)


@dp.message(Admin.wait_keys, F.web_app_data)
async def admin_save_keys(msg: Message, state: FSMContext):
    if not is_admin(msg.from_user.id):
        return
    data       = await state.get_data()
    web_data   = json.loads(msg.web_app_data.data)
    final_keys = web_data["closed_answers"] + web_data["open_answers"]
    db_run(
        "INSERT INTO tests (code, name, n_questions, answer_key) VALUES (?, ?, 55, ?)",
        (data["test_code"], data["test_name"], json.dumps(final_keys))
    )
    await state.clear()
    await msg.answer(
        f"✅ <b>Test muvaffaqiyatli yaratildi!</b>\n\n"
        f"📋 Test nomi: <b>{data['test_name']}</b>\n"
        f"🔑 O'quvchilar uchun kod: <code>{data['test_code']}</code>",
        reply_markup=ReplyKeyboardRemove()
    )


@dp.message(Command("tests"))
async def admin_tests(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    rows = db_get("SELECT code, name, is_active, created_at FROM tests ORDER BY id DESC")
    if not rows:
        await msg.answer("📭 Hozircha test yo'q. /newtest bilan yarating.")
        return
    lines = ["📋 <b>Barcha testlar:</b>\n"]
    for r in rows:
        status = "🟢 Faol" if r["is_active"] else "🔴 Yopiq"
        cnt    = db_get(
            "SELECT COUNT(*) as c FROM responses WHERE test_id=(SELECT id FROM tests WHERE code=?)",
            (r["code"],)
        )
        count = cnt[0]["c"] if cnt else 0
        lines.append(
            f"{status} <code>{r['code']}</code> — <b>{r['name']}</b>\n"
            f"   👥 {count} ishtirokchi | 📅 {r['created_at'][:10]}"
        )
    await msg.answer("\n".join(lines))


@dp.message(Command("results"))
async def admin_results(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /results KOD\nMasalan: /results MAT55")
        return
    code = parts[1].strip().upper()
    test = db_get("SELECT * FROM tests WHERE code=?", (code,))
    if not test:
        await msg.answer(f"❌ <code>{code}</code> kodli test topilmadi.")
        return
    test = test[0]

    # ✅ FIX: NULLS LAST o'rniga CASE WHEN ishlatildi (barcha SQLite versiyasida ishlaydi)
    resp = db_get(
        "SELECT full_name, xom_ball, t_score, submitted_at FROM responses "
        "WHERE test_id=? ORDER BY CASE WHEN t_score IS NULL THEN 0 ELSE 1 END DESC, t_score DESC",
        (test["id"],)
    )
    if not resp:
        await msg.answer(f"📭 <code>{code}</code> testida hali natija yo'q.")
        return

    lines  = [f"📊 <b>{test['name']}</b> (<code>{code}</code>) natijalari:\n"]
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, r in enumerate(resp, 1):
        medal = medals.get(i, f"{i}.")
        t_sc  = f"{r['t_score']:.2f}" if r["t_score"] is not None else "—"
        xom   = r["xom_ball"] if r["xom_ball"] is not None else "—"
        d     = daraja(r["t_score"])
        lines.append(f"{medal} {r['full_name']} | {xom}/55 | {t_sc} | {d}")

    await msg.answer("\n".join(lines[:50]))

    # CSV fayl
    buf    = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "To'liq ism", "Xom ball", "T-ball", "Daraja", "Sana"])
    for i, r in enumerate(resp, 1):
        t_sc = f"{r['t_score']:.2f}" if r["t_score"] is not None else ""
        writer.writerow([i, r["full_name"], r["xom_ball"], t_sc, daraja(r["t_score"]), r["submitted_at"]])

    file = BufferedInputFile(buf.getvalue().encode("utf-8-sig"), filename=f"{code}_natijalar.csv")
    await msg.answer_document(file, caption=f"📥 <b>{code}</b> — to'liq natijalar CSV")


@dp.message(Command("endtest"))
async def admin_endtest(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /endtest KOD\nMasalan: /endtest MAT55")
        return
    code = parts[1].strip().upper()
    test = db_get("SELECT * FROM tests WHERE code=?", (code,))
    if not test:
        await msg.answer(f"❌ <code>{code}</code> topilmadi.")
        return
    if not test[0]["is_active"]:
        await msg.answer(f"⚠️ <code>{code}</code> allaqachon yopiq.")
        return
    db_run("UPDATE tests SET is_active=0 WHERE code=?", (code,))
    await msg.answer(
        f"🔴 <code>{code}</code> — <b>{test[0]['name']}</b> yopildi.\n"
        f"Endi o'quvchilar bu testni topshira olmaydi."
    )


@dp.message(Command("deltest"))
async def admin_deltest(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) < 2:
        await msg.answer("❗ Ishlatish: /deltest KOD\nMasalan: /deltest MAT55")
        return
    code = parts[1].strip().upper()
    test = db_get("SELECT * FROM tests WHERE code=?", (code,))
    if not test:
        await msg.answer(f"❌ <code>{code}</code> topilmadi.")
        return
    cnt = db_get("SELECT COUNT(*) as c FROM responses WHERE test_id=?", (test[0]["id"],))
    n   = cnt[0]["c"] if cnt else 0
    kb  = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Ha, o'chir", callback_data=f"del_{code}"),
        InlineKeyboardButton(text="❌ Bekor",      callback_data="del_cancel")
    ]])
    await msg.answer(
        f"⚠️ <b>Diqqat!</b>\n"
        f"<code>{code}</code> — <b>{test[0]['name']}</b>\n"
        f"👥 {n} ta natija ham o'chib ketadi!\n\n"
        f"Tasdiqlaysizmi?",
        reply_markup=kb
    )


# ✅ FIX: CallbackQuery type annotation to'g'rilandi
@dp.callback_query(F.data.startswith("del_"))
async def confirm_delete(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("❌ Ruxsat yo'q!", show_alert=True)
        return
    if call.data == "del_cancel":
        await call.message.edit_text("❌ O'chirish bekor qilindi.")
        return
    code = call.data[4:]  # "del_" ni olib tashlaymiz
    test = db_get("SELECT id FROM tests WHERE code=?", (code,))
    if test:
        db_run("DELETE FROM responses WHERE test_id=?", (test[0]["id"],))
        db_run("DELETE FROM tests WHERE id=?",          (test[0]["id"],))
        await call.message.edit_text(f"🗑 <code>{code}</code> va barcha natijalari o'chirildi.")
    else:
        await call.message.edit_text("❌ Test topilmadi.")
    await call.answer()


@dp.message(Command("addadmin"))
async def add_admin(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip().isdigit():
        await msg.answer("❗ Ishlatish: /addadmin TELEGRAM_ID\nMasalan: /addadmin 123456789")
        return
    new_id = int(parts[1].strip())
    db_run("INSERT OR IGNORE INTO admins (tg_id) VALUES (?)", (new_id,))
    ADMIN_IDS.add(new_id)
    await msg.answer(f"✅ <code>{new_id}</code> admin sifatida qo'shildi.")


@dp.message(Command("admins"))
async def list_admins(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    rows  = db_get("SELECT tg_id, added_at FROM admins ORDER BY added_at")
    lines = ["👥 <b>Adminlar ro'yxati:</b>\n"]
    for r in rows:
        lines.append(f"• <code>{r['tg_id']}</code> — {r['added_at'][:10]}")
    await msg.answer("\n".join(lines))


# ═══════════════════════════════════════════════════
#  O'QUVCHILAR BILAN ISHLASH
# ═══════════════════════════════════════════════════
@dp.message(Student.enter_code)
async def student_code(msg: Message, state: FSMContext):
    if is_admin(msg.from_user.id):
        return
    code = msg.text.strip().upper()
    rows = db_get("SELECT * FROM tests WHERE code=? AND is_active=1", (code,))
    if not rows:
        await msg.answer("❌ Kod noto'g'ri yoki test yakunlangan.\nQaytadan kiriting:")
        return
    t = rows[0]
    if db_get("SELECT id FROM responses WHERE test_id=? AND tg_id=?", (t["id"], msg.from_user.id)):
        await msg.answer("⚠️ Siz bu testni allaqachon topshirib bo'lgansiz.")
        await state.clear()
        return
    await state.update_data(test_id=t["id"], test_name=t["name"], answer_key=t["answer_key"])
    await state.set_state(Student.enter_name)
    await msg.answer(f"✅ Test topildi: <b>{t['name']}</b>\n\n📝 To'liq ism-familiyangizni kiriting:")


@dp.message(Student.enter_name)
async def student_name(msg: Message, state: FSMContext):
    if is_admin(msg.from_user.id):
        return
    name = msg.text.strip()
    if len(name) < 4:
        await msg.answer("❗ Ism-familiyangizni to'liq kiriting (kamida 4 harf):")
        return
    await state.update_data(full_name=name)
    await state.set_state(Student.wait_answers)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📝 Imtihon Oynasini Ochish", web_app=WebAppInfo(url=WEB_APP_URL))]],
        resize_keyboard=True
    )
    await msg.answer(
        f"✅ Rahmat, <b>{name}</b>!\n\n"
        f"👇 Pastdagi tugmani bosing va imtihonni boshlang:",
        reply_markup=kb
    )


@dp.message(Student.wait_answers, F.web_app_data)
async def student_get_answers(msg: Message, state: FSMContext):
    if is_admin(msg.from_user.id):
        return

    # ✅ FIX: Bot restart bo'lsa FSM data bo'sh — qayta yo'naltirish
    fsm_data = await state.get_data()
    if not fsm_data.get("test_id"):
        await state.clear()
        await state.set_state(Student.enter_code)
        await msg.answer(
            "⚠️ Bot yangilandi, ma'lumotlar tozalandi.\n"
            "Iltimos, test kodini qaytadan kiriting:",
            reply_markup=ReplyKeyboardRemove()
        )
        return

    try:
        web_data     = json.loads(msg.web_app_data.data)
        stud_answers = web_data["closed_answers"] + web_data["open_answers"]
        test_keys    = json.loads(fsm_data["answer_key"])
    except (json.JSONDecodeError, KeyError) as e:
        log.error(f"Web app data xatosi: {e}")
        await msg.answer("❌ Ma'lumotlarni o'qishda xatolik. Qaytadan urinib ko'ring.")
        return

    binary = []
    for i in range(TOTAL_QUESTIONS):
        s_ans = stud_answers[i].strip() if i < len(stud_answers) else ""
        k_ans = test_keys[i].strip()    if i < len(test_keys)    else ""
        if i < CLOSED_COUNT:
            binary.append(1 if s_ans.upper() == k_ans.upper() else 0)
        else:
            binary.append(1 if s_ans.lower() == k_ans.lower() else 0)

    try:
        db_run(
            "INSERT OR IGNORE INTO responses (test_id, tg_id, full_name, raw_answers, binary_str) "
            "VALUES (?, ?, ?, ?, ?)",
            (fsm_data["test_id"], msg.from_user.id, fsm_data["full_name"],
             json.dumps(stud_answers), json.dumps(binary))
        )
    except Exception as e:
        log.error(f"Javoblarni saqlashda xato: {e}")
        await msg.answer("❌ Saqlashda xatolik yuz berdi. Qaytadan urinib ko'ring.")
        return

    await state.clear()
    await msg.answer(
        "⏳ <b>Javoblaringiz qabul qilindi!</b>\n\nNatijangiz hisoblanmoqda, biroz kuting...",
        reply_markup=ReplyKeyboardRemove()
    )
    await auto_rasch(bot, fsm_data["test_id"], msg.from_user.id, fsm_data["full_name"])


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════
async def main():
    db_init()
    log.info("Bot ishga tushmoqda... v6.1")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
