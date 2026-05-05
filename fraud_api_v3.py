from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib, pandas as pd, numpy as np, sqlite3, os
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import os
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
        username TEXT NOT NULL, transaction_id TEXT NOT NULL, timestamp TEXT NOT NULL,
        Transaction_Amount REAL NOT NULL, Account_Balance REAL NOT NULL,
        Device_Type TEXT NOT NULL, Merchant_Category TEXT NOT NULL,
        Card_Type TEXT NOT NULL, Card_Age INTEGER NOT NULL, Is_Weekend INTEGER NOT NULL,
        Previous_Fraudulent_Activity INTEGER NOT NULL, Daily_Transaction_Count INTEGER NOT NULL,
        Avg_Transaction_Amount_7d REAL NOT NULL, Failed_Transaction_Count_7d INTEGER NOT NULL,
        Device_Risk INTEGER NOT NULL, Merchant_Risk INTEGER NOT NULL, Card_Risk INTEGER NOT NULL,
        Device_Type_Laptop INTEGER NOT NULL, Device_Type_Mobile INTEGER NOT NULL, Device_Type_Tablet INTEGER NOT NULL,
        Merchant_Category_Clothing INTEGER NOT NULL, Merchant_Category_Electronics INTEGER NOT NULL,
        Merchant_Category_Groceries INTEGER NOT NULL, Merchant_Category_Restaurants INTEGER NOT NULL,
        Merchant_Category_Travel INTEGER NOT NULL,
        Card_Type_Amex INTEGER NOT NULL, Card_Type_Discover INTEGER NOT NULL,
        Card_Type_Mastercard INTEGER NOT NULL, Card_Type_Visa INTEGER NOT NULL,
        rf_verdict TEXT NOT NULL, rf_probability REAL NOT NULL,
        xgb_verdict TEXT NOT NULL, xgb_probability REAL NOT NULL,
        lgb_verdict TEXT NOT NULL, lgb_probability REAL NOT NULL,
        final_verdict TEXT NOT NULL, fraud_votes INTEGER NOT NULL, avg_probability REAL NOT NULL)""")
    conn.commit(); conn.close()
init_db()

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
        "SELECT COUNT(*) FROM transactions WHERE username=? AND timestamp>=? AND final_verdict='Fraud'",
        (username, seven_days_ago)).fetchone()[0]
    prev_fraud = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=? AND final_verdict='Fraud'",
        (username,)).fetchone()[0]
    conn.close()
    return {"Daily_Transaction_Count":daily,"Avg_Transaction_Amount_7d":avg_7d,
            "Failed_Transaction_Count_7d":failed_7d,"Previous_Fraudulent_Activity":prev_fraud}

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
    
    # حساب Transaction Risk بدون التاريخ
    df_no_history = df.copy()
    df_no_history["Previous_Fraudulent_Activity"]  = 0
    df_no_history["Failed_Transaction_Count_7d"]   = 0
    
    txn_probs = []
    for name, model in [("rf",rf),("xgb",xgb_model),("lgb",lgb_model)]:
        txn_probs.append(float(model.predict_proba(df_no_history)[:,1][0]))
    txn_risk = round(float(np.mean(txn_probs)),4)
    
    # تطبيق المنطق
    if txn_risk < 0.20:
        # المعاملة نفسها آمنة جداً — Safe بغض النظر عن التاريخ
        final_verdict = "Safe"
        avg_prob = txn_risk
        for name in results:
            results[name]["verdict"] = "Safe"
            results[name]["prob"] = txn_risk
        fraud_votes = 0
    elif txn_risk > 0.50:
        # المعاملة نفسها خطيرة — زود الاحتمالية
        boosted_prob = round(min(avg_prob * 1.3, 1.0), 4)
        avg_prob = boosted_prob
        final_verdict = "Fraud" if fraud_votes >= 2 else "Safe"
    else:
        # المنطقة الرمادية — الموديل يقرر بالتاريخ عادي
        final_verdict = "Fraud" if fraud_votes >= 2 else "Safe"
    
    return {**results, "fraud_votes":fraud_votes,
            "avg_probability":avg_prob,
            "final_verdict":final_verdict,
            "txn_risk":txn_risk}
class TransactionInput(BaseModel):
    username: str
    Transaction_Amount: float
    Account_Balance: float
    Device_Type: str
    Merchant_Category: str
    Card_Type: str
    Card_Age: int

class TransactionResponse(BaseModel):
    username: str
    transaction_id: str
    timestamp: str
    final_verdict: str
    fraud_votes: int
    avg_probability: float
    rf_verdict: str
    rf_probability: float
    xgb_verdict: str
    xgb_probability: float
    lgb_verdict: str
    lgb_probability: float
    cumulative: dict

app = FastAPI(title="Fraud Detection API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root(): return {"message":"Fraud Detection API v3","ui":"/ui","docs":"/docs"}

@app.get("/health")
def health(): return {"status":"ok","version":"3.0.0"}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse("""<!DOCTYPE html><html lang="ar" dir="rtl">
<head><meta charset="UTF-8"/><title>Fraud Detection</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:Segoe UI,sans-serif;background:#f5f6fa;direction:rtl}
.header{background:#1a1a2e;color:white;padding:1.25rem 2rem;display:flex;align-items:center;gap:12px}
.header h1{font-size:20px}.container{max-width:750px;margin:2rem auto;padding:0 1rem}
.card{background:white;border-radius:12px;padding:1.5rem;margin-bottom:1.25rem;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.stitle{font-size:11px;font-weight:600;color:#888;text-transform:uppercase;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid #f0f0f0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.field{display:flex;flex-direction:column;gap:6px}.field label{font-size:13px;color:#555;font-weight:500}
.field input,.field select{border:1.5px solid #e8e8e8;border-radius:8px;padding:9px 12px;font-size:14px;font-family:inherit;background:#fafafa}
.field input:focus,.field select:focus{outline:none;border-color:#4f6ef7}
.badge{display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:8px;border:1.5px solid #e8e8e8;background:#fafafa;font-size:13px;color:#777;min-height:42px}
.dot{width:8px;height:8px;border-radius:50%}.dot.idle{background:#ccc}.dot.new{background:#22c55e}.dot.ex{background:#3b82f6}
.cum-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:.75rem}
.cum-card{background:#f8f9ff;border:1px solid #e8eaf6;border-radius:8px;padding:.75rem;text-align:center}
.cum-card .val{font-size:22px;font-weight:700;color:#3730a3}.cum-card .lbl{font-size:11px;color:#888;margin-top:3px}
.btn{width:100%;padding:13px;background:#1a1a2e;color:white;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;font-family:inherit}
.btn:hover{background:#2d2d5e}.btn:disabled{background:#ccc;cursor:not-allowed}
.result{border-radius:12px;overflow:hidden;border:1px solid #e8e8e8;margin-top:1.25rem}
.rh{display:flex;align-items:center;gap:12px;padding:1.25rem 1.5rem}
.rh.fraud{background:#fef2f2;border-bottom:1px solid #fecaca}.rh.safe{background:#f0fdf4;border-bottom:1px solid #bbf7d0}
.rh .verdict{font-size:20px;font-weight:700}.rh.fraud .verdict{color:#dc2626}.rh.safe .verdict{color:#16a34a}
.rh .sub{font-size:13px;margin-top:2px}.rh.fraud .sub{color:#b91c1c}.rh.safe .sub{color:#15803d}
.mrow{display:grid;grid-template-columns:repeat(3,1fr)}
.mc{padding:.9rem 1rem;text-align:center;border-right:1px solid #f0f0f0}.mc:last-child{border-right:none}
.mn{font-size:11px;color:#999;margin-bottom:4px}.mv{font-size:15px;font-weight:700}
.mv.fraud{color:#dc2626}.mv.safe{color:#16a34a}.mp{font-size:12px;color:#888}
.info{background:#f8f9ff;border-radius:8px;padding:.75rem 1rem;margin-top:.75rem;font-size:12px;color:#666}
.info span{color:#3730a3;font-weight:600}.error{background:#fef2f2;border:1px solid #fecaca;color:#dc2626;padding:.75rem 1rem;border-radius:8px;margin-top:1rem;font-size:14px}
</style></head><body>
<div class="header"><span style="font-size:24px">&#128269;</span><h1>Fraud Detection System</h1></div>
<div class="container">
<div class="card"><div class="stitle">بيانات المستخدم</div>
<div class="grid2">
<div class="field"><label>اسم المستخدم</label><input type="text" id="username" placeholder="ahmed_ali"/></div>
<div class="field"><label>حالة المستخدم</label><div class="badge"><div class="dot idle" id="sdot"></div><span id="stext">في انتظار الاسم...</span></div></div>
</div></div>
<div class="card"><div class="stitle">بيانات المعاملة</div>
<div class="grid2" style="margin-bottom:14px">
<div class="field"><label>مبلغ المعاملة</label><input type="number" id="amount" placeholder="500" min="0"/></div>
<div class="field"><label>رصيد الحساب</label><input type="number" id="balance" placeholder="12000" min="0"/></div>
</div>
<div class="grid3" style="margin-bottom:14px">
<div class="field"><label>نوع الجهاز</label><select id="device"><option value="">اختر...</option><option>Mobile</option><option>Laptop</option><option>Tablet</option></select></div>
<div class="field"><label>فئة التاجر</label><select id="merchant"><option value="">اختر...</option><option>Electronics</option><option>Travel</option><option>Clothing</option><option>Restaurants</option><option>Groceries</option></select></div>
<div class="field"><label>نوع الكارت</label><select id="card_type"><option value="">اختر...</option><option>Visa</option><option>Mastercard</option><option>Amex</option><option>Discover</option></select></div>
</div>
<div class="grid2"><div class="field"><label>عمر الكارت (شهر)</label><input type="number" id="card_age" placeholder="24" min="0"/></div></div>
</div>
<div class="card"><div class="stitle">القيم التراكمية (محسوبة تلقائياً)</div>
<div style="font-size:12px;color:#999;margin-bottom:.5rem">بتتجيب من تاريخ المستخدم في قاعدة البيانات</div>
<div class="cum-grid">
<div class="cum-card"><div class="val" id="c1">-</div><div class="lbl">معاملات اليوم</div></div>
<div class="cum-card"><div class="val" id="c2">-</div><div class="lbl">متوسط 7 أيام</div></div>
<div class="cum-card"><div class="val" id="c3">-</div><div class="lbl">Fraud 7 أيام</div></div>
<div class="cum-card"><div class="val" id="c4">-</div><div class="lbl">إجمالي Fraud</div></div>
</div></div>
<button class="btn" id="btn" disabled>&#128269; فحص المعاملة</button>
<div id="err"></div><div id="res"></div>
</div>
<script>
document.addEventListener("DOMContentLoaded",function(){
const API=window.location.origin;
let timer;
async function loadUser(u){
  try{
    const r=await fetch(API+"/user/"+u+"/history");
    const dot=document.getElementById("sdot"),txt=document.getElementById("stext");
    if(r.ok){const d=await r.json();const fc=d.transactions.filter(t=>t.final_verdict==="Fraud").length;
      dot.className="dot ex";txt.textContent="موجود - "+d.total_transactions+" معاملة | "+fc+" fraud";
      const today=new Date().toISOString().slice(0,10);
      document.getElementById("c1").textContent=d.transactions.filter(t=>t.timestamp.slice(0,10)===today).length+1;
      const s7=new Date(Date.now()-7*864e5).toISOString(),l7=d.transactions.filter(t=>t.timestamp>=s7);
      const amt=parseFloat(document.getElementById("amount").value)||0;
      const amts=l7.map(t=>t.Transaction_Amount).concat([amt]);
      document.getElementById("c2").textContent=(amts.reduce((a,b)=>a+b,0)/amts.length).toFixed(2);
      document.getElementById("c3").textContent=l7.filter(t=>t.final_verdict==="Fraud").length;
      document.getElementById("c4").textContent=fc;
    }else{dot.className="dot new";txt.textContent="مستخدم جديد";
      ["c1","c2","c3","c4"].forEach((id,i)=>document.getElementById(id).textContent=i===0?1:0);}
  }catch{document.getElementById("stext").textContent="تعذر الاتصال";}
}
document.getElementById("username").addEventListener("input",function(){
  clearTimeout(timer);const u=this.value.trim();
  if(!u){document.getElementById("sdot").className="dot idle";document.getElementById("stext").textContent="في انتظار الاسم...";return;}
  timer=setTimeout(()=>loadUser(u),600);checkBtn();
});
document.getElementById("amount").addEventListener("input",()=>{const u=document.getElementById("username").value.trim();if(u)loadUser(u);checkBtn();});
["balance","device","merchant","card_type","card_age"].forEach(id=>{
  document.getElementById(id).addEventListener("input",checkBtn);
  document.getElementById(id).addEventListener("change",checkBtn);});
function checkBtn(){document.getElementById("btn").disabled=false;}
document.getElementById("btn").addEventListener("click",async()=>{
  document.getElementById("err").innerHTML="";document.getElementById("res").innerHTML="";
  document.getElementById("btn").disabled=true;document.getElementById("btn").textContent="جاري الفحص...";
  try{
    const r=await fetch(API+"/transaction",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({username:document.getElementById("username").value.trim(),
        Transaction_Amount:parseFloat(document.getElementById("amount").value),
        Account_Balance:parseFloat(document.getElementById("balance").value),
        Device_Type:document.getElementById("device").value,
        Merchant_Category:document.getElementById("merchant").value,
        Card_Type:document.getElementById("card_type").value,
        Card_Age:parseInt(document.getElementById("card_age").value)})});
    const d=await r.json();
    if(!r.ok){document.getElementById("err").innerHTML="<div class=\"error\">"+d.detail+"</div>";return;}
    const f=d.final_verdict==="Fraud",cls=f?"fraud":"safe";
    document.getElementById("res").innerHTML="<div class=\"result\"><div class=\"rh "+cls+"\"><span style=\"font-size:24px\">"+(f?"&#9888;":"&#9989;")+"</span><div><div class=\"verdict\">"+(f?"احتيال محتمل":"معاملة آمنة")+"</div><div class=\"sub\">احتمالية: "+(d.avg_probability*100).toFixed(1)+"% - "+d.fraud_votes+"/3 قالوا Fraud</div></div></div><div class=\"mrow\"><div class=\"mc\"><div class=\"mn\">Random Forest</div><div class=\"mv "+(d.rf_verdict==="Fraud"?"fraud":"safe")+"\">"+d.rf_verdict+"</div><div class=\"mp\">"+(d.rf_probability*100).toFixed(1)+"%</div></div><div class=\"mc\"><div class=\"mn\">XGBoost</div><div class=\"mv "+(d.xgb_verdict==="Fraud"?"fraud":"safe")+"\">"+d.xgb_verdict+"</div><div class=\"mp\">"+(d.xgb_probability*100).toFixed(1)+"%</div></div><div class=\"mc\"><div class=\"mn\">LightGBM</div><div class=\"mv "+(d.lgb_verdict==="Fraud"?"fraud":"safe")+"\">"+d.lgb_verdict+"</div><div class=\"mp\">"+(d.lgb_probability*100).toFixed(1)+"%</div></div></div></div><div class=\"info\">رقم المعاملة: <span>"+d.transaction_id+"</span></div>";
    loadUser(document.getElementById("username").value.trim());
  }catch{document.getElementById("err").innerHTML="<div class=\"error\">تعذر الاتصال بالسيرفر</div>";}
  finally{document.getElementById("btn").disabled=false;document.getElementById("btn").textContent="&#128269; فحص المعاملة";}
});
});
</script></body></html>""")

@app.post("/transaction", response_model=TransactionResponse)
def register_transaction(t: TransactionInput):
    # حساب تصنيف المستخدم من التاريخ
    conn_check = sqlite3.connect(DB_PATH)
    avg_historical = conn_check.execute(
        "SELECT AVG(Transaction_Amount) FROM transactions WHERE username=?",
        (t.username,)).fetchone()[0]
    total_txns = conn_check.execute(
        "SELECT COUNT(*) FROM transactions WHERE username=?",
        (t.username,)).fetchone()[0]
    conn_check.close()

    # تحديد التصنيف السعري
    def get_tier(amount):
        if amount is None or amount < 5000: return "low"
        elif amount < 10000: return "medium"
        else: return "high"

    user_tier = get_tier(avg_historical)
    txn_tier  = get_tier(t.Transaction_Amount)
    tier_order = {"low": 1, "medium": 2, "high": 3}

    # القاعدة 1 — كارت جديد + مبلغ عالي
    if t.Card_Age < 3 and t.Transaction_Amount >= 10000:
        if user_tier != "high":
            now = datetime.now()
            transaction_id = "TXN-" + t.username + "-" + str(int(now.timestamp()))
            is_weekend = 1 if now.weekday() >= 5 else 0
            cum = calc_cumulative(t.username, t.Transaction_Amount, now)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.username, transaction_id, now.isoformat(), t.Transaction_Amount, t.Account_Balance,
                 t.Device_Type, t.Merchant_Category, t.Card_Type, t.Card_Age, is_weekend,
                 cum["Previous_Fraudulent_Activity"], cum["Daily_Transaction_Count"],
                 cum["Avg_Transaction_Amount_7d"], cum["Failed_Transaction_Count_7d"],
                 DEVICE_RISK.get(t.Device_Type,0), MERCHANT_RISK.get(t.Merchant_Category,0), CARD_RISK.get(t.Card_Type,0),
                 1 if t.Device_Type=="Laptop" else 0, 1 if t.Device_Type=="Mobile" else 0, 1 if t.Device_Type=="Tablet" else 0,
                 1 if t.Merchant_Category=="Clothing" else 0, 1 if t.Merchant_Category=="Electronics" else 0,
                 1 if t.Merchant_Category=="Groceries" else 0, 1 if t.Merchant_Category=="Restaurants" else 0,
                 1 if t.Merchant_Category=="Travel" else 0,
                 1 if t.Card_Type=="Amex" else 0, 1 if t.Card_Type=="Discover" else 0,
                 1 if t.Card_Type=="Mastercard" else 0, 1 if t.Card_Type=="Visa" else 0,
                 "Fraud", 1.0, "Fraud", 1.0, "Fraud", 1.0, "Fraud", 3, 1.0))
            conn.commit(); conn.close()
            return TransactionResponse(
                username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
                final_verdict="Fraud", fraud_votes=3, avg_probability=1.0,
                rf_verdict="Fraud", rf_probability=1.0,
                xgb_verdict="Fraud", xgb_probability=1.0,
                lgb_verdict="Fraud", lgb_probability=1.0,
                cumulative={**cum, "rule": "كارت جديد مع مبلغ عالي"})

    # القاعدة 2 — حساب جديد + مبلغ فوق 20000
    if total_txns == 0 and t.Transaction_Amount > 20000:
        now = datetime.now()
        transaction_id = "TXN-" + t.username + "-" + str(int(now.timestamp()))
        is_weekend = 1 if now.weekday() >= 5 else 0
        cum = calc_cumulative(t.username, t.Transaction_Amount, now)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.username, transaction_id, now.isoformat(), t.Transaction_Amount, t.Account_Balance,
             t.Device_Type, t.Merchant_Category, t.Card_Type, t.Card_Age, is_weekend,
             cum["Previous_Fraudulent_Activity"], cum["Daily_Transaction_Count"],
             cum["Avg_Transaction_Amount_7d"], cum["Failed_Transaction_Count_7d"],
             DEVICE_RISK.get(t.Device_Type,0), MERCHANT_RISK.get(t.Merchant_Category,0), CARD_RISK.get(t.Card_Type,0),
             1 if t.Device_Type=="Laptop" else 0, 1 if t.Device_Type=="Mobile" else 0, 1 if t.Device_Type=="Tablet" else 0,
             1 if t.Merchant_Category=="Clothing" else 0, 1 if t.Merchant_Category=="Electronics" else 0,
             1 if t.Merchant_Category=="Groceries" else 0, 1 if t.Merchant_Category=="Restaurants" else 0,
             1 if t.Merchant_Category=="Travel" else 0,
             1 if t.Card_Type=="Amex" else 0, 1 if t.Card_Type=="Discover" else 0,
             1 if t.Card_Type=="Mastercard" else 0, 1 if t.Card_Type=="Visa" else 0,
             "Fraud", 1.0, "Fraud", 1.0, "Fraud", 1.0, "Fraud", 3, 1.0))
        conn.commit(); conn.close()
        return TransactionResponse(
            username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
            final_verdict="Fraud", fraud_votes=3, avg_probability=1.0,
            rf_verdict="Fraud", rf_probability=1.0,
            xgb_verdict="Fraud", xgb_probability=1.0,
            lgb_verdict="Fraud", lgb_probability=1.0,
            cumulative={**cum, "rule": "حساب جديد مع مبلغ كبير جداً"})

    # القاعدة 3 — تجاوز التصنيف السعري
    if total_txns > 0 and tier_order[txn_tier] > tier_order[user_tier]:
        now = datetime.now()
        transaction_id = "TXN-" + t.username + "-" + str(int(now.timestamp()))
        is_weekend = 1 if now.weekday() >= 5 else 0
        cum = calc_cumulative(t.username, t.Transaction_Amount, now)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.username, transaction_id, now.isoformat(), t.Transaction_Amount, t.Account_Balance,
             t.Device_Type, t.Merchant_Category, t.Card_Type, t.Card_Age, is_weekend,
             cum["Previous_Fraudulent_Activity"], cum["Daily_Transaction_Count"],
             cum["Avg_Transaction_Amount_7d"], cum["Failed_Transaction_Count_7d"],
             DEVICE_RISK.get(t.Device_Type,0), MERCHANT_RISK.get(t.Merchant_Category,0), CARD_RISK.get(t.Card_Type,0),
             1 if t.Device_Type=="Laptop" else 0, 1 if t.Device_Type=="Mobile" else 0, 1 if t.Device_Type=="Tablet" else 0,
             1 if t.Merchant_Category=="Clothing" else 0, 1 if t.Merchant_Category=="Electronics" else 0,
             1 if t.Merchant_Category=="Groceries" else 0, 1 if t.Merchant_Category=="Restaurants" else 0,
             1 if t.Merchant_Category=="Travel" else 0,
             1 if t.Card_Type=="Amex" else 0, 1 if t.Card_Type=="Discover" else 0,
             1 if t.Card_Type=="Mastercard" else 0, 1 if t.Card_Type=="Visa" else 0,
             "Fraud", 1.0, "Fraud", 1.0, "Fraud", 1.0, "Fraud", 3, 1.0))
        conn.commit(); conn.close()
        return TransactionResponse(
            username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
            final_verdict="Fraud", fraud_votes=3, avg_probability=1.0,
            rf_verdict="Fraud", rf_probability=1.0,
            xgb_verdict="Fraud", xgb_probability=1.0,
            lgb_verdict="Fraud", lgb_probability=1.0,
            cumulative={**cum, "rule": "تجاوز التصنيف السعري المعتاد"})
    if t.Transaction_Amount > t.Account_Balance:
        now = datetime.now()
        transaction_id = "TXN-" + t.username + "-" + str(int(now.timestamp()))
        is_weekend = 1 if now.weekday() >= 5 else 0
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.username, transaction_id, now.isoformat(), t.Transaction_Amount, t.Account_Balance,
             t.Device_Type, t.Merchant_Category, t.Card_Type, t.Card_Age, is_weekend,
             0, 1, t.Transaction_Amount, 0,
             DEVICE_RISK.get(t.Device_Type, 0), MERCHANT_RISK.get(t.Merchant_Category, 0), CARD_RISK.get(t.Card_Type, 0),
             1 if t.Device_Type=="Laptop" else 0, 1 if t.Device_Type=="Mobile" else 0, 1 if t.Device_Type=="Tablet" else 0,
             1 if t.Merchant_Category=="Clothing" else 0, 1 if t.Merchant_Category=="Electronics" else 0,
             1 if t.Merchant_Category=="Groceries" else 0, 1 if t.Merchant_Category=="Restaurants" else 0,
             1 if t.Merchant_Category=="Travel" else 0,
             1 if t.Card_Type=="Amex" else 0, 1 if t.Card_Type=="Discover" else 0,
             1 if t.Card_Type=="Mastercard" else 0, 1 if t.Card_Type=="Visa" else 0,
             "Fraud", 1.0, "Fraud", 1.0, "Fraud", 1.0, "Fraud", 3, 1.0))
        conn.commit()
        conn.close()
        return TransactionResponse(
            username=t.username, transaction_id=transaction_id, timestamp=now.isoformat(),
            final_verdict="Fraud", fraud_votes=3, avg_probability=1.0,
            rf_verdict="Fraud", rf_probability=1.0,
            xgb_verdict="Fraud", xgb_probability=1.0,
            lgb_verdict="Fraud", lgb_probability=1.0,
            cumulative={"message": "لا يمكن اتمام المعاملة: المبلغ أكبر من رصيد الحساب",
                       "Daily_Transaction_Count": 1, "Avg_Transaction_Amount_7d": t.Transaction_Amount,
                       "Failed_Transaction_Count_7d": 1, "Previous_Fraudulent_Activity": 0})
    if t.Device_Type not in DEVICE_RISK: raise HTTPException(400, f"Device_Type: {list(DEVICE_RISK)}")
    if t.Merchant_Category not in MERCHANT_RISK: raise HTTPException(400, f"Merchant_Category: {list(MERCHANT_RISK)}")
    if t.Card_Type not in CARD_RISK: raise HTTPException(400, f"Card_Type: {list(CARD_RISK)}")
    now=datetime.now(); transaction_id="TXN-"+t.username+"-"+str(int(now.timestamp())); is_weekend=1 if now.weekday()>=5 else 0
    cum=calc_cumulative(t.username,t.Transaction_Amount,now)
    full={**t.__dict__,"Is_Weekend":is_weekend,**cum}
    df=build_features(full)
    v=run_voting(df)
    conn=sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO transactions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (t.username,transaction_id,now.isoformat(),t.Transaction_Amount,t.Account_Balance,t.Device_Type,
         t.Merchant_Category,t.Card_Type,t.Card_Age,is_weekend,
         cum["Previous_Fraudulent_Activity"],cum["Daily_Transaction_Count"],cum["Avg_Transaction_Amount_7d"],cum["Failed_Transaction_Count_7d"],
         DEVICE_RISK[t.Device_Type],MERCHANT_RISK[t.Merchant_Category],CARD_RISK[t.Card_Type],
         1 if t.Device_Type=="Laptop" else 0,1 if t.Device_Type=="Mobile" else 0,1 if t.Device_Type=="Tablet" else 0,
         1 if t.Merchant_Category=="Clothing" else 0,1 if t.Merchant_Category=="Electronics" else 0,
         1 if t.Merchant_Category=="Groceries" else 0,1 if t.Merchant_Category=="Restaurants" else 0,
         1 if t.Merchant_Category=="Travel" else 0,
         1 if t.Card_Type=="Amex" else 0,1 if t.Card_Type=="Discover" else 0,
         1 if t.Card_Type=="Mastercard" else 0,1 if t.Card_Type=="Visa" else 0,
         v["rf"]["verdict"],v["rf"]["prob"],v["xgb"]["verdict"],v["xgb"]["prob"],
         v["lgb"]["verdict"],v["lgb"]["prob"],v["final_verdict"],v["fraud_votes"],v["avg_probability"]))
    conn.commit(); conn.close()
    return TransactionResponse(username=t.username,transaction_id=transaction_id,timestamp=now.isoformat(),
        final_verdict=v["final_verdict"],fraud_votes=v["fraud_votes"],avg_probability=v["avg_probability"],
        rf_verdict=v["rf"]["verdict"],rf_probability=v["rf"]["prob"],
        xgb_verdict=v["xgb"]["verdict"],xgb_probability=v["xgb"]["prob"],
        lgb_verdict=v["lgb"]["verdict"],lgb_probability=v["lgb"]["prob"],cumulative=cum)

@app.get("/user/{username}/history")
def get_user_history(username:str):
    conn=sqlite3.connect(DB_PATH)
    rows=conn.execute("SELECT * FROM transactions WHERE username=? ORDER BY id DESC",(username,)).fetchall()
    cols=[d[0] for d in conn.execute("PRAGMA table_info(transactions)").fetchall()]
    conn.close()
    if not rows: raise HTTPException(404,"المستخدم غير موجود")
    return {"username":username,"total_transactions":len(rows),"transactions":[dict(zip(cols,r)) for r in rows]}

@app.get("/stats")
def get_stats():
    conn=sqlite3.connect(DB_PATH)
    total=conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud=conn.execute("SELECT COUNT(*) FROM transactions WHERE final_verdict='Fraud'").fetchone()[0]
    users=conn.execute("SELECT COUNT(DISTINCT username) FROM transactions").fetchone()[0]
    conn.close()
    rate=str(round(fraud/total*100,1))+"%" if total else "0%"
    return {"total_transactions":total,"total_fraud":fraud,"total_safe":total-fraud,"fraud_rate":rate,"total_users":users}
