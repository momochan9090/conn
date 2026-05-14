from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib, pandas as pd, numpy as np, sqlite3, os
from datetime import datetime, timedelta

MODELS_PATH = "/app"
DB_PATH     = "/app/transactions.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

try:
    rf        = joblib.load(os.path.join(MODELS_PATH, "rf_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_PATH, "xgb_model.pkl"))
    os.environ["LIGHTGBM_EXEC_PATH"] = ""
    lgb_model = joblib.load(os.path.join(MODELS_PATH, "lgb_model.pkl"))
except Exception as e:
    raise RuntimeError(f"فشل تحميل الموديلات: {e}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        user_type TEXT DEFAULT 'شاري',
        email TEXT DEFAULT NULL,
        phone TEXT DEFAULT NULL,
        transaction_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        Transaction_Amount REAL NOT NULL,
        Account_Balance REAL NOT NULL,
        Device_Type TEXT NOT NULL,
        Merchant_Category TEXT NOT NULL,
        Card_Type TEXT NOT NULL,
        Card_Age INTEGER NOT NULL,
        Is_Weekend INTEGER NOT NULL,
        Previous_Fraudulent_Activity INTEGER NOT NULL,
        Daily_Transaction_Count INTEGER NOT NULL,
        Avg_Transaction_Amount_7d REAL NOT NULL,
        Failed_Transaction_Count_7d INTEGER NOT NULL,
        Device_Risk INTEGER NOT NULL,
        Merchant_Risk INTEGER NOT NULL,
        Card_Risk INTEGER NOT NULL,
        Device_Type_Laptop INTEGER NOT NULL,
        Device_Type_Mobile INTEGER NOT NULL,
        Device_Type_Tablet INTEGER NOT NULL,
        Merchant_Category_Clothing INTEGER NOT NULL,
        Merchant_Category_Electronics INTEGER NOT NULL,
        Merchant_Category_Groceries INTEGER NOT NULL,
        Merchant_Category_Restaurants INTEGER NOT NULL,
        Merchant_Category_Travel INTEGER NOT NULL,
        Card_Type_Amex INTEGER NOT NULL,
        Card_Type_Discover INTEGER NOT NULL,
        Card_Type_Mastercard INTEGER NOT NULL,
        Card_Type_Visa INTEGER NOT NULL,
        rf_verdict TEXT NOT NULL,
        rf_probability REAL NOT NULL,
        xgb_verdict TEXT NOT NULL,
        xgb_probability REAL NOT NULL,
        lgb_verdict TEXT NOT NULL,
        lgb_probability REAL NOT NULL,
        final_verdict TEXT NOT NULL,
        fraud_votes INTEGER NOT NULL,
        avg_probability REAL NOT NULL,
        risk_score REAL DEFAULT 0,
        fraud_reason TEXT DEFAULT NULL,
        is_blocked INTEGER DEFAULT 0
    )""")
    conn.commit()
    conn.close()

def init_cards_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        card_number TEXT NOT NULL,
        card_holder TEXT NOT NULL,
        exp_month TEXT NOT NULL,
        exp_year TEXT NOT NULL,
        card_type TEXT NOT NULL,
        cvv TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(card_number)
    )""")
    conn.commit()
    conn.close()
init_cards_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        user_type TEXT DEFAULT 'شاري',
        email TEXT DEFAULT NULL,
        phone TEXT DEFAULT NULL,
        transaction_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        Transaction_Amount REAL NOT NULL,
        Account_Balance REAL NOT NULL,
        Device_Type TEXT NOT NULL,
        Merchant_Category TEXT NOT NULL,
        Card_Type TEXT NOT NULL,
        Card_Age INTEGER NOT NULL,
        Is_Weekend INTEGER NOT NULL,
        Previous_Fraudulent_Activity INTEGER NOT NULL,
        Daily_Transaction_Count INTEGER NOT NULL,
        Avg_Transaction_Amount_7d REAL NOT NULL,
        Failed_Transaction_Count_7d INTEGER NOT NULL,
        Device_Risk INTEGER NOT NULL,
        Merchant_Risk INTEGER NOT NULL,
        Card_Risk INTEGER NOT NULL,
        Device_Type_Laptop INTEGER NOT NULL,
        Device_Type_Mobile INTEGER NOT NULL,
        Device_Type_Tablet INTEGER NOT NULL,
        Merchant_Category_Clothing INTEGER NOT NULL,
        Merchant_Category_Electronics INTEGER NOT NULL,
        Merchant_Category_Groceries INTEGER NOT NULL,
        Merchant_Category_Restaurants INTEGER NOT NULL,
        Merchant_Category_Travel INTEGER NOT NULL,
        Card_Type_Amex INTEGER NOT NULL,
        Card_Type_Discover INTEGER NOT NULL,
        Card_Type_Mastercard INTEGER NOT NULL,
        Card_Type_Visa INTEGER NOT NULL,
        rf_verdict TEXT NOT NULL,
        rf_probability REAL NOT NULL,
        xgb_verdict TEXT NOT NULL,
        xgb_probability REAL NOT NULL,
        lgb_verdict TEXT NOT NULL,
        lgb_probability REAL NOT NULL,
        final_verdict TEXT NOT NULL,
        fraud_votes INTEGER NOT NULL,
        avg_probability REAL NOT NULL,
        risk_score REAL DEFAULT 0,
        fraud_reason TEXT DEFAULT NULL,
        is_blocked INTEGER DEFAULT 0
    )""")
    conn.commit()
    conn.close()
init_db()

def init_cards_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        card_number TEXT NOT NULL,
        card_holder TEXT NOT NULL,
        exp_month TEXT NOT NULL,
        exp_year TEXT NOT NULL,
        card_type TEXT NOT NULL,
        cvv TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(card_number)
    )""")
    conn.commit()
    conn.close()
init_cards_db()

DEVICE_RISK   = {"Mobile":2,"Laptop":1,"Tablet":0}
MERCHANT_RISK = {"Electronics":4,"Travel":3,"Clothing":2,"Restaurants":1,"Groceries":0}
CARD_RISK     = {"Visa":3,"Mastercard":2,"Amex":1,"Discover":0}
FEATURE_ORDER = ["Transaction_Amount","Account_Balance","Previous_Fraudulent_Activity",
    "Daily_Transaction_Count","Avg_Transaction_Amount_7d","Failed_Transaction_Count_7d",
    "Card_Age","Is_Weekend","Device_Risk","Merchant_Risk","Card_Risk",
    "Device_Type_Laptop","Device_Type_Mobile","Device_Type_Tablet",
    "Merchant_Category_Clothing","Merchant_Category_Electronics","Merchant_Category_Groceries",
    "Merchant_Category_Restaurants","Merchant_Category_Travel",
    "Card_Type_Amex","Card_Type_Discover","Card_Type_Mastercard","Card_Type_Visa"]

def calc_cumulative(username, current_amount, now):
    conn = sqlite3.connect(DB_PATH)
    today = now.strftime("%Y-%m-%d")
    daily = conn.execute("SELECT COUNT(*) FROM transactions WHERE username=? AND timestamp LIKE ?",
        (username, f"{today}%")).fetchone()[0] + 1
    seven_days_ago = (now - timedelta(days=7)).isoformat()
    amounts = [r[0] for r in conn.execute(
        "SELECT Transaction_Amount FROM transactions WHERE username=? AND timestamp>=?",
        (username, seven_days_ago)).fetchall()] + [current_amount]
    avg_7d = round(float(np.mean(amounts)), 2)
    failed_7d = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=? AND timestamp>=? AND final_verdict=?",
        (username, seven_days_ago, "Fraud")).fetchone()[0]
    prev_fraud = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=? AND final_verdict=?",
        (username, "Fraud")).fetchone()[0]
    conn.close()
    return {"Daily_Transaction_Count":daily,"Avg_Transaction_Amount_7d":avg_7d,
            "Failed_Transaction_Count_7d":failed_7d,"Previous_Fraudulent_Activity":prev_fraud}

def get_user_tier(avg_amount):
    if avg_amount is None or avg_amount < 5000: return "low"
    elif avg_amount < 10000: return "medium"
    else: return "high"

TIER_ORDER = {"low":1,"medium":2,"high":3}

def build_features(data):
    row = {
        "Transaction_Amount":data["Transaction_Amount"],
        "Account_Balance":data["Account_Balance"],
        "Previous_Fraudulent_Activity":data["Previous_Fraudulent_Activity"],
        "Daily_Transaction_Count":data["Daily_Transaction_Count"],
        "Avg_Transaction_Amount_7d":data["Avg_Transaction_Amount_7d"],
        "Failed_Transaction_Count_7d":data["Failed_Transaction_Count_7d"],
        "Card_Age":data["Card_Age"],"Is_Weekend":data["Is_Weekend"],
        "Device_Risk":DEVICE_RISK[data["Device_Type"]],
        "Merchant_Risk":MERCHANT_RISK[data["Merchant_Category"]],
        "Card_Risk":CARD_RISK[data["Card_Type"]],
        "Device_Type_Laptop":1 if data["Device_Type"]=="Laptop" else 0,
        "Device_Type_Mobile":1 if data["Device_Type"]=="Mobile" else 0,
        "Device_Type_Tablet":1 if data["Device_Type"]=="Tablet" else 0,
        "Merchant_Category_Clothing":1 if data["Merchant_Category"]=="Clothing" else 0,
        "Merchant_Category_Electronics":1 if data["Merchant_Category"]=="Electronics" else 0,
        "Merchant_Category_Groceries":1 if data["Merchant_Category"]=="Groceries" else 0,
        "Merchant_Category_Restaurants":1 if data["Merchant_Category"]=="Restaurants" else 0,
        "Merchant_Category_Travel":1 if data["Merchant_Category"]=="Travel" else 0,
        "Card_Type_Amex":1 if data["Card_Type"]=="Amex" else 0,
        "Card_Type_Discover":1 if data["Card_Type"]=="Discover" else 0,
        "Card_Type_Mastercard":1 if data["Card_Type"]=="Mastercard" else 0,
        "Card_Type_Visa":1 if data["Card_Type"]=="Visa" else 0}
    return pd.DataFrame([row])[FEATURE_ORDER]

def run_voting(df):
    results = {}
    for name, model in [("rf",rf),("xgb",xgb_model),("lgb",lgb_model)]:
        prob = float(model.predict_proba(df)[:,1][0])
        pred = int(model.predict(df)[0])
        results[name] = {"prob":round(prob,4),"pred":pred,"verdict":"Fraud" if pred==1 else "Safe"}
    fraud_votes = sum(r["pred"] for r in results.values())
    avg_prob    = round(float(np.mean([r["prob"] for r in results.values()])),4)
    final       = "Fraud" if fraud_votes >= 2 else "Safe"
    return {**results,"fraud_votes":fraud_votes,"avg_probability":avg_prob,"final_verdict":final}

class TransactionInput(BaseModel):
    username:                     str
    user_type:                    Optional[str] = "شاري"
    email:                        Optional[str] = None
    phone:                        Optional[str] = None
    card_number:                  Optional[str] = None
    card_holder:                  Optional[str] = None
    exp_month:                    Optional[str] = None
    exp_year:                     Optional[str] = None
    cvv:                          Optional[str] = None
    Transaction_Amount:           float
    Account_Balance:              float
    Device_Type:                  str
    Merchant_Category:            str
    Card_Type:                    str
    Card_Age:                     int

class TransactionResponse(BaseModel):
    username:         str
    transaction_id:   str
    timestamp:        str
    final_verdict:    str
    fraud_votes:      int
    avg_probability:  float
    risk_score:       float
    fraud_reason:     Optional[str]
    is_blocked:       bool
    rf_verdict:       str
    rf_probability:   float
    xgb_verdict:      str
    xgb_probability:  float
    lgb_verdict:      str
    lgb_probability:  float
    cumulative:       dict

app = FastAPI(title="Fraud Detection API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"message":"Fraud Detection API v4","ui":"/ui","docs":"/docs"}

@app.get("/health")
def health(): return {"status":"ok","version":"4.0.0"}

@app.post("/transaction", response_model=TransactionResponse)
def register_transaction(t: TransactionInput):
    if t.Device_Type not in DEVICE_RISK: raise HTTPException(400, f"Device_Type: {list(DEVICE_RISK)}")
    if t.Merchant_Category not in MERCHANT_RISK: raise HTTPException(400, f"Merchant_Category: {list(MERCHANT_RISK)}")
    if t.Card_Type not in CARD_RISK: raise HTTPException(400, f"Card_Type: {list(CARD_RISK)}")

    now            = datetime.now()
    transaction_id = "TXN-" + t.username + "-" + str(int(now.timestamp()))
    is_weekend     = 1 if now.weekday() >= 5 else 0
    risk_score     = 0.0
    fraud_reasons  = []
    is_blocked     = False

    # ============================================================
    # بيانات المستخدم من DB
    # ============================================================
    conn_check = sqlite3.connect(DB_PATH)

    # هل الحساب متجمد؟
    today = now.strftime("%Y-%m-%d")
    blocked = conn_check.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=? AND is_blocked=1 AND timestamp LIKE ?",
        (t.username, f"{today}%")).fetchone()[0]
    if blocked > 0:
        conn_check.close()
        cum = calc_cumulative(t.username, t.Transaction_Amount, now)
        return TransactionResponse(
            username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
            final_verdict="Fraud", fraud_votes=3, avg_probability=1.0,
            risk_score=1.0, fraud_reason="الحساب متجمد لتجاوز حد المعاملات المسموح به اليوم",
            is_blocked=True,
            rf_verdict="Fraud", rf_probability=1.0,
            xgb_verdict="Fraud", xgb_probability=1.0,
            lgb_verdict="Fraud", lgb_probability=1.0,
            cumulative=cum)

    # عدد المعاملات في آخر ساعة
    one_hour_ago = (now - timedelta(hours=1)).isoformat()
    txns_last_hour = conn_check.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=? AND timestamp>=?",
        (t.username, one_hour_ago)).fetchone()[0]

    # إجمالي المعاملات
    total_txns = conn_check.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=?",
        (t.username,)).fetchone()[0]

    # متوسط المبالغ التاريخية
    avg_historical = conn_check.execute(
        "SELECT AVG(Transaction_Amount) FROM transactions WHERE username=?",
        (t.username,)).fetchone()[0]

    # متوسط عدد المعاملات الأسبوعي
    oldest_txn = conn_check.execute(
        "SELECT MIN(timestamp) FROM transactions WHERE username=?",
        (t.username,)).fetchone()[0]

    conn_check.close()

    # ============================================================
    # القاعدة 1 — رصيد غير كافي
    # ============================================================
    if t.Transaction_Amount > t.Account_Balance:
        cum = calc_cumulative(t.username, t.Transaction_Amount, now)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.username, t.user_type, t.email, t.phone, transaction_id, now.isoformat(),
             t.Transaction_Amount, t.Account_Balance, t.Device_Type, t.Merchant_Category,
             t.Card_Type, t.Card_Age, is_weekend,
             cum["Previous_Fraudulent_Activity"], cum["Daily_Transaction_Count"],
             cum["Avg_Transaction_Amount_7d"], cum["Failed_Transaction_Count_7d"],
             DEVICE_RISK[t.Device_Type], MERCHANT_RISK[t.Merchant_Category], CARD_RISK[t.Card_Type],
             1 if t.Device_Type=="Laptop" else 0, 1 if t.Device_Type=="Mobile" else 0, 1 if t.Device_Type=="Tablet" else 0,
             1 if t.Merchant_Category=="Clothing" else 0, 1 if t.Merchant_Category=="Electronics" else 0,
             1 if t.Merchant_Category=="Groceries" else 0, 1 if t.Merchant_Category=="Restaurants" else 0,
             1 if t.Merchant_Category=="Travel" else 0,
             1 if t.Card_Type=="Amex" else 0, 1 if t.Card_Type=="Discover" else 0,
             1 if t.Card_Type=="Mastercard" else 0, 1 if t.Card_Type=="Visa" else 0,
             "Fraud", 1.0, "Fraud", 1.0, "Fraud", 1.0, "Fraud", 3, 1.0, 1.0,
             "رصيد غير كافي", 0))
        conn.commit(); conn.close()
        return TransactionResponse(
            username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
            final_verdict="Fraud", fraud_votes=3, avg_probability=1.0,
            risk_score=1.0, fraud_reason="لا يمكن اتمام المعاملة — لا يوجد رصيد كافي في الحساب المرفق",
            is_blocked=False,
            rf_verdict="Fraud", rf_probability=1.0,
            xgb_verdict="Fraud", xgb_probability=1.0,
            lgb_verdict="Fraud", lgb_probability=1.0,
            cumulative=cum)

    # ============================================================
    # القاعدة 2 — تجاوز 10 معاملات في ساعة
    # ============================================================
    if txns_last_hour >= 10:
        is_blocked = True
        risk_score += 0.40
        fraud_reasons.append("تجاوز الحد المسموح به من المعاملات في ساعة واحدة")

    # ============================================================
    # القاعدة 3 — كارت جديد (0-3 شهور)
    # ============================================================
    if t.Card_Age < 3:
        risk_score += 0.10
        fraud_reasons.append("كارت حديث الإصدار")
        if t.Transaction_Amount >= 20000:
            user_tier = get_user_tier(avg_historical)
            if user_tier != "high":
                risk_score += 0.20
                fraud_reasons.append("مبلغ عالي جداً لكارت جديد")

    # ============================================================
    # القاعدة 4 — تجاوز التصنيف السعري
    # ============================================================
    if total_txns > 0 and avg_historical is not None:
        user_tier = get_user_tier(avg_historical)
        txn_tier  = get_user_tier(t.Transaction_Amount)
        if TIER_ORDER[txn_tier] > TIER_ORDER[user_tier]:
            risk_score += 0.20
            fraud_reasons.append(f"المعاملة تتجاوز التصنيف السعري المعتاد للمستخدم")

    # ============================================================
    # القاعدة 5 — زيادة مفاجئة في عدد المعاملات
    # ============================================================
    if oldest_txn and total_txns >= 5:
        oldest_date = datetime.fromisoformat(oldest_txn)
        days_active = max((now - oldest_date).days, 1)
        avg_daily   = total_txns / days_active
        today_count = calc_cumulative(t.username, t.Transaction_Amount, now)["Daily_Transaction_Count"]
        if today_count > avg_daily * 3:
            risk_score += 0.15
            fraud_reasons.append("زيادة مفاجئة في عدد المعاملات عن المعدل المعتاد")

    # ============================================================
    # القاعدة 6 — موثوقية الحساب
    # ============================================================
    trust_bonus = 0.0
    if t.email: trust_bonus += 0.05
    if t.phone: trust_bonus += 0.05
    risk_score = max(0.0, risk_score - trust_bonus)

    # ============================================================
    # تشغيل الموديلات
    # ============================================================
    cum = calc_cumulative(t.username, t.Transaction_Amount, now)
    full_data = {**t.__dict__, "Is_Weekend": is_weekend, **cum}
    df = build_features(full_data)
    v  = run_voting(df)

    # دمج نسبة الموديل مع القواعد
    final_risk   = round(min(v["avg_probability"] + risk_score, 1.0), 4)
    final_verdict = "Fraud" if final_risk >= 0.5 else "Safe"
    fraud_reason  = " | ".join(fraud_reasons) if fraud_reasons and final_verdict == "Fraud" else None

    # ============================================================
    # حفظ في DB
    # ============================================================
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (t.username, t.user_type, t.email, t.phone, transaction_id, now.isoformat(),
         t.Transaction_Amount, t.Account_Balance, t.Device_Type, t.Merchant_Category,
         t.Card_Type, t.Card_Age, is_weekend,
         cum["Previous_Fraudulent_Activity"], cum["Daily_Transaction_Count"],
         cum["Avg_Transaction_Amount_7d"], cum["Failed_Transaction_Count_7d"],
         DEVICE_RISK[t.Device_Type], MERCHANT_RISK[t.Merchant_Category], CARD_RISK[t.Card_Type],
         1 if t.Device_Type=="Laptop" else 0, 1 if t.Device_Type=="Mobile" else 0, 1 if t.Device_Type=="Tablet" else 0,
         1 if t.Merchant_Category=="Clothing" else 0, 1 if t.Merchant_Category=="Electronics" else 0,
         1 if t.Merchant_Category=="Groceries" else 0, 1 if t.Merchant_Category=="Restaurants" else 0,
         1 if t.Merchant_Category=="Travel" else 0,
         1 if t.Card_Type=="Amex" else 0, 1 if t.Card_Type=="Discover" else 0,
         1 if t.Card_Type=="Mastercard" else 0, 1 if t.Card_Type=="Visa" else 0,
         v["rf"]["verdict"], v["rf"]["prob"],
         v["xgb"]["verdict"], v["xgb"]["prob"],
         v["lgb"]["verdict"], v["lgb"]["prob"],
         final_verdict, v["fraud_votes"], final_risk,
         final_risk, fraud_reason, 1 if is_blocked else 0))
    conn.commit(); conn.close()

    # حفظ الكارت لو اتبعت بياناته
    if hasattr(t, 'card_number') and t.card_number:
        card_num = t.card_number.replace(" ","")
        conn_c = sqlite3.connect(DB_PATH)
        existing = conn_c.execute(
            "SELECT id FROM cards WHERE card_number=?", (card_num,)).fetchone()
        if not existing and hasattr(t, 'card_holder') and t.card_holder:
            conn_c.execute(
                "INSERT OR IGNORE INTO cards VALUES (NULL,?,?,?,?,?,?,?,?)",
                (t.username, card_num, t.card_holder,
                 getattr(t,'exp_month',''), getattr(t,'exp_year',''),
                 t.Card_Type, getattr(t,'cvv',''), now.isoformat()))
            conn_c.commit()
        conn_c.close()

    return TransactionResponse(
        username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
        final_verdict=final_verdict, fraud_votes=v["fraud_votes"],
        avg_probability=final_risk, risk_score=final_risk,
        fraud_reason=fraud_reason, is_blocked=is_blocked,
        rf_verdict=v["rf"]["verdict"], rf_probability=v["rf"]["prob"],
        xgb_verdict=v["xgb"]["verdict"], xgb_probability=v["xgb"]["prob"],
        lgb_verdict=v["lgb"]["verdict"], lgb_probability=v["lgb"]["prob"],
        cumulative=cum)

@app.get("/user/{username}/history")
def get_user_history(username: str):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM transactions WHERE username=? ORDER BY id DESC",
        (username,)).fetchall()
    cols = [d[0] for d in conn.execute("PRAGMA table_info(transactions)").fetchall()]
    conn.close()
    if not rows: raise HTTPException(404, f"المستخدم غير موجود")
    return {"username": username, "total_transactions": len(rows),
            "transactions": [dict(zip(cols, r)) for r in rows]}

@app.get("/user/{username}/status")
def get_user_status(username: str):
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM transactions WHERE username=?", (username,)).fetchone()[0]
    fraud = conn.execute("SELECT COUNT(*) FROM transactions WHERE username=? AND final_verdict='Fraud'", (username,)).fetchone()[0]
    avg   = conn.execute("SELECT AVG(Transaction_Amount) FROM transactions WHERE username=?", (username,)).fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    blocked = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=? AND is_blocked=1 AND timestamp LIKE ?",
        (username, f"{today}%")).fetchone()[0]
    conn.close()
    if total == 0: raise HTTPException(404, "المستخدم غير موجود")
    tier = get_user_tier(avg)
    return {
        "username": username,
        "total_transactions": total,
        "total_fraud": fraud,
        "avg_transaction_amount": round(avg, 2) if avg else 0,
        "user_tier": tier,
        "is_blocked_today": blocked > 0,
    }


@app.post("/card/register")
def register_card(data: dict):
    username   = data.get("username")
    card_number= data.get("card_number","").replace(" ","")
    card_holder= data.get("card_holder")
    exp_month  = data.get("exp_month")
    exp_year   = data.get("exp_year")
    card_type  = data.get("card_type")
    cvv        = data.get("cvv")

    if not all([username, card_number, card_holder, exp_month, exp_year, card_type, cvv]):
        raise HTTPException(400, "كل الحقول مطلوبة")
    if len(card_number) != 16:
        raise HTTPException(400, "رقم الكارت لازم يكون 16 رقم")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cards VALUES (NULL,?,?,?,?,?,?,?,?)",
            (username, card_number, card_holder, exp_month, exp_year, card_type, cvv, datetime.now().isoformat()))
        conn.commit()
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        conn.close()
    return {"message": "تم تسجيل الكارت بنجاح", "card_number": card_number[-4:]}

@app.get("/card/{card_number}")
def get_card(card_number: str):
    card_number = card_number.replace(" ","")
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT * FROM cards WHERE card_number=?", (card_number,)).fetchone()
    cols = [d[0] for d in conn.execute("PRAGMA table_info(cards)").fetchall()]
    conn.close()
    if not row:
        raise HTTPException(404, "الكارت مش متسجل")
    card = dict(zip(cols, row))
    # حساب عمر الكارت
    exp  = datetime(int(card["exp_year"]), int(card["exp_month"]), 1)
    now  = datetime.now()
    issue= datetime(exp.year - 5, exp.month, 1)
    age  = (now.year - issue.year)*12 + (now.month - issue.month)
    card["card_age"] = max(0, age)
    card["cvv"]      = "***"  # مش هنرجع الـ CVV
    return card

@app.get("/stats")
def get_stats():
    conn  = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud = conn.execute("SELECT COUNT(*) FROM transactions WHERE final_verdict='Fraud'").fetchone()[0]
    users = conn.execute("SELECT COUNT(DISTINCT username) FROM transactions").fetchone()[0]
    conn.close()
    rate = str(round(fraud/total*100,1))+"%" if total else "0%"
    return {"total_transactions":total,"total_fraud":fraud,
            "total_safe":total-fraud,"fraud_rate":rate,"total_users":users}
