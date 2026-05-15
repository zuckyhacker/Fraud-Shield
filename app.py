"""
╔══════════════════════════════════════════════════════════════╗
║   FraudShield — Call & SMS Fraud Detection System           ║
║   Streamlit Application  |  Python 3.10+                   ║
║   Paste this file + requirements.txt into your GitHub repo  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import re
import io
import csv
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="FraudShield — AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════

CC_MAP  = {"+1": 0, "+44": 1, "+61": 2, "+49": 3, "+91": 4, "+234": 5, "+880": 6}
TOD_MAP = {"early_morning": 0, "morning": 1, "afternoon": 2, "evening": 3, "late_night": 4}

_CC_REGIONS = [
    ("+880", "Bangladesh 🇧🇩",   True),
    ("+234", "Nigeria 🇳🇬",      True),
    ("+256", "Uganda 🇺🇬",       True),
    ("+933", "Afghanistan 🇦🇫",  True),
    ("+7",   "Russia 🇷🇺",       True),
    ("+966", "Saudi Arabia 🇸🇦", False),
    ("+971", "UAE 🇦🇪",          False),
    ("+255", "Tanzania 🇹🇿",     False),
    ("+254", "Kenya 🇰🇪",        False),
    ("+233", "Ghana 🇬🇭",        False),
    ("+353", "Ireland 🇮🇪",      False),
    ("+380", "Ukraine 🇺🇦",      False),
    ("+39",  "Italy 🇮🇹",        False),
    ("+33",  "France 🇫🇷",       False),
    ("+34",  "Spain 🇪🇸",        False),
    ("+31",  "Netherlands 🇳🇱",  False),
    ("+49",  "Germany 🇩🇪",      False),
    ("+44",  "UK 🇬🇧",           False),
    ("+61",  "Australia 🇦🇺",    False),
    ("+65",  "Singapore 🇸🇬",    False),
    ("+81",  "Japan 🇯🇵",        False),
    ("+82",  "South Korea 🇰🇷",  False),
    ("+86",  "China 🇨🇳",        False),
    ("+91",  "India 🇮🇳",        False),
    ("+92",  "Pakistan 🇵🇰",     False),
    ("+1",   "USA / Canada 🇺🇸", False),
]

_PREMIUM_PREFIXES = ["0900", "0976", "0970", "0871", "0872",
                     "0873", "0844", "0845", "0870", "976", "900"]

CHART_STYLE = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#1c1c2e",
    "axes.edgecolor":   "#2d2d4e",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#a0aec0",
    "ytick.color":      "#a0aec0",
    "text.color":       "#e2e8f0",
    "grid.color":       "#2d2d4e",
    "grid.alpha":       0.4,
}
PALETTE = ["#6c63ff", "#ff6584"]

# ════════════════════════════════════════════════════════════════════
#  NLTK DOWNLOAD
# ════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _download_nltk():
    for r in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass
    return True

_download_nltk()

# ════════════════════════════════════════════════════════════════════
#  DATASET GENERATION
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_datasets():
    np.random.seed(42)

    spam_messages = [
        "WINNER!! You have been selected to receive a $1000 Walmart gift card. Call now!",
        "Congratulations! You've won a FREE iPhone 15. Click here to claim your prize!",
        "URGENT: Your bank account has been compromised. Call 1-800-FRAUD now!",
        "You have been pre-approved for a $5000 loan. No credit check required!",
        "FREE ringtones! Text YES to 12345 to get unlimited downloads. $9.99/month.",
        "Your PayPal account is limited. Verify your identity at http://paypal-verify.xyz",
        "Claim your FREE vacation package to Bahamas! Limited time offer. Reply YES",
        "ALERT: Suspicious login to your account. Verify immediately: bit.ly/secure123",
        "You are our lucky customer! Win $500 Amazon voucher. Text CLAIM to 5555",
        "Get rich quick! Invest $100 today and earn $10000 in 30 days guaranteed!",
        "Your mobile number has won £750,000 in the UK National Lottery! Call now",
        "Congratulations, you have been chosen for our VIP membership. Click to activate",
        "FREE ENTRY: Win a top of the range 2006 Yamaha Enduro. Text RIDE to 87121",
        "Urgent reply needed: You have won a Nokia 3250. 1 year guarantee.",
        "DOUBLE your money in 24hrs! Guaranteed returns. WhatsApp +1234567890",
        "Your parcel could not be delivered. Pay £1.99 to redeliver: parcel-uk.xyz",
        "CASH PRIZE: You have been selected! Call 0800-XXX-XXXX to claim your £5000",
        "Lose 10kg in 2 weeks! Our miracle pill is proven to work. Order now free shipping",
        "IMPORTANT: Your Google account was accessed from another device. Verify: g00gle.xyz",
        "You have pending $347 federal tax refund. Claim now: irs-refund.net",
        "Earn $500 daily from home! No experience needed. Apply: earn-fast.xyz",
        "CRYPTO ALERT: Bitcoin is about to surge. Invest now before it's too late!",
        "Your debit card has been blocked. Unblock it immediately: bank-secure.xyz",
        "FREE gift for completing our survey. Click here: survey-prize.xyz. Ends tonight!",
        "JACKPOT: $2.5 million awaits you! Your email was randomly selected. Reply NOW",
        "Warning: Virus detected on your phone! Download our FREE cleaner app now!",
        "Hot singles in your area! Meet them now for FREE. Text MEET to 44444",
        "Your subscription is about to expire. Renew at discounted price: renew-now.xyz",
        "LIMITED OFFER: Grow your business online! Social media followers package $29",
        "You have 1 unread secret message. Click to read: secret-msg.xyz. Free trial!",
        "FINAL NOTICE: You owe $847. Pay now to avoid legal action: pay-debt.xyz",
        "Exclusive: Win a brand new BMW! Enter now. Text BMW to 70070. T&C apply",
        "FREE mortgage quote. Save thousands! Visit loans-fast.com or call 0800-LOAN",
        "PRIZE NOTIFICATION: You have won a luxury cruise. Call 0901-PRIZE to claim",
        "Special offer for you! Buy 1 get 3 FREE on all vitamins. Shop: health-deals.xyz",
        "Your account will be suspended. Update billing info: secure-update.xyz/pay",
        "You've qualified for our exclusive reward. Collect your £100 voucher now!",
        "CONTEST: You entered and WON! Claim your Samsung TV. Text WIN to 83738",
        "Urgent security alert. Your device is infected. Install protection: safe-now.xyz",
        "FREE iPhone for survey completion! Visit iphone-survey.xyz before midnight!",
    ]

    ham_messages = [
        "Hey, are you coming to the party tonight? Let me know by 6pm.",
        "Can you pick up some milk on your way home? Also need bread.",
        "The meeting has been moved to 3pm tomorrow. Conference room B.",
        "Happy birthday! Hope you have an amazing day filled with joy.",
        "Just finished the report. Will send it over by end of day.",
        "Running 10 minutes late. Start without me, be there soon.",
        "Great job on the presentation today! The client was impressed.",
        "Did you see the game last night? Incredible finish!",
        "Dinner reservation is confirmed for 7:30pm at La Bella.",
        "Your doctor appointment is scheduled for Monday at 2pm. Please confirm.",
        "The package has been delivered to your front door.",
        "Can we reschedule our call to Thursday? I have a conflict tomorrow.",
        "Mom wants to know if you're coming for Thanksgiving. Let her know.",
        "The wifi password is FamilyHome2024. Uppercase F and H.",
        "Just landed. Getting my luggage now. Will call when I'm outside.",
        "Reminder: Your car service is due next week. Call us to book.",
        "Thanks for the help yesterday! Really appreciated it.",
        "The kids' school play is on Friday at 6pm. Don't forget!",
        "Are you free for lunch this week? Would love to catch up.",
        "Your Amazon order has shipped. Expected delivery: Thursday.",
        "Good morning! Just checking in to see how you're feeling today.",
        "Meeting notes from today's session are attached. Please review.",
        "We need to talk about the project timeline. Can you call me?",
        "Don't forget to submit your expense report before Friday.",
        "Watched that movie you recommended. It was really good!",
        "Your electricity bill for this month is $127.45. Due by the 15th.",
        "Library book return reminder. Books due in 3 days to avoid fines.",
        "Flight confirmed: AA123 departs at 8:45am. Terminal 2, Gate B14.",
        "Gym closes at 9pm today due to maintenance. Normal hours resume tomorrow.",
        "Hi, this is Dr. Smith's office confirming your appointment for tomorrow.",
        "The report you requested is ready. You can pick it up at reception.",
        "Congrats on your promotion! Well deserved. Let's celebrate this weekend.",
        "Just to confirm, we're meeting at the coffee shop at 11am tomorrow?",
        "Your tax documents are ready for review. Please log in to your account.",
        "Hope you're feeling better! Let us know if you need anything.",
        "The book club meeting is at Sarah's place on Wednesday. Bring snacks!",
        "I left my keys at your place. Can I swing by to grab them?",
        "Reminder: HOA meeting this Thursday at 7pm in the community center.",
        "Your insurance renewal is coming up. Contact us to review your coverage.",
        "Just checking — did you get my email about the project update?",
    ]

    n_spam, n_ham = 800, 3200
    sms_df = pd.DataFrame({
        "message": [spam_messages[i % len(spam_messages)] for i in range(n_spam)] +
                   [ham_messages[i % len(ham_messages)]  for i in range(n_ham)],
        "label":   ["spam"] * n_spam + ["ham"] * n_ham,
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    rows = []
    for _ in range(1000):
        rows.append({
            "call_duration": float(np.random.choice([
                np.random.uniform(0.1, 5),
                np.random.uniform(300, 1800),
            ], p=[0.6, 0.4])),
            "frequency":        int(np.random.randint(5, 30)),
            "country_code":     np.random.choice(["+880", "+234", "+91", "+44", "+1"], p=[0.3, 0.3, 0.2, 0.1, 0.1]),
            "is_international": int(np.random.choice([0, 1], p=[0.2, 0.8])),
            "unknown_number":   int(np.random.choice([0, 1], p=[0.1, 0.9])),
            "time_of_day":      np.random.choice(["early_morning", "late_night"]),
            "label": "fraud",
        })
    for _ in range(3000):
        rows.append({
            "call_duration":    float(np.random.uniform(30, 600)),
            "frequency":        int(np.random.randint(1, 6)),
            "country_code":     np.random.choice(["+1", "+44", "+61", "+49"], p=[0.5, 0.2, 0.2, 0.1]),
            "is_international": int(np.random.choice([0, 1], p=[0.8, 0.2])),
            "unknown_number":   int(np.random.choice([0, 1], p=[0.8, 0.2])),
            "time_of_day":      np.random.choice(["morning", "afternoon", "evening"], p=[0.4, 0.4, 0.2]),
            "label": "safe",
        })
    call_df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return sms_df, call_df

# ════════════════════════════════════════════════════════════════════
#  NLP HELPERS
# ════════════════════════════════════════════════════════════════════

def _preprocess_text(text, stemmer, stop_words):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _train_models():
    plt.rcParams.update(CHART_STYLE)
    stemmer    = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    sms_df, call_df = _build_datasets()

    # SMS
    sms_df["cleaned"]   = sms_df["message"].apply(lambda t: _preprocess_text(t, stemmer, stop_words))
    le_sms              = LabelEncoder()
    sms_df["label_enc"] = le_sms.fit_transform(sms_df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        sms_df["cleaned"], sms_df["label_enc"], test_size=0.2, random_state=42, stratify=sms_df["label_enc"]
    )
    tfidf   = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr_vec = tfidf.fit_transform(X_tr)
    Xte_vec = tfidf.transform(X_te)

    sms_candidates = {
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    sms_results = {}
    for name, m in sms_candidates.items():
        m.fit(Xtr_vec, y_tr)
        preds = m.predict(Xte_vec)
        sms_results[name] = {"model": m, "accuracy": accuracy_score(y_te, preds), "preds": preds}
    best_sms_name = max(sms_results, key=lambda k: sms_results[k]["accuracy"])
    best_sms      = sms_results[best_sms_name]

    # Call
    cc_enc_map  = {"+1": 0, "+44": 1, "+61": 2, "+49": 3, "+91": 4, "+234": 5, "+880": 6}
    tod_enc_map = {"early_morning": 0, "morning": 1, "afternoon": 2, "evening": 3, "late_night": 4}
    call_df["cc_enc"]  = call_df["country_code"].map(cc_enc_map).fillna(7)
    call_df["tod_enc"] = call_df["time_of_day"].map(tod_enc_map).fillna(2)
    le_call   = LabelEncoder()
    y_call    = le_call.fit_transform(call_df["label"])
    feat_cols = ["call_duration", "frequency", "cc_enc", "is_international", "unknown_number", "tod_enc"]
    X_call    = call_df[feat_cols].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_call, y_call, test_size=0.2, random_state=42, stratify=y_call)
    scaler   = StandardScaler()
    Xc_tr_sc = scaler.fit_transform(Xc_tr)
    Xc_te_sc = scaler.transform(Xc_te)

    call_candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    call_results = {}
    for name, m in call_candidates.items():
        m.fit(Xc_tr_sc, yc_tr)
        preds = m.predict(Xc_te_sc)
        call_results[name] = {"model": m, "accuracy": accuracy_score(yc_te, preds), "preds": preds}
    best_call_name = max(call_results, key=lambda k: call_results[k]["accuracy"])
    best_call      = call_results[best_call_name]

    # Charts
    charts = {}

    def fig(w=6, h=4):
        f, a = plt.subplots(figsize=(w, h)); return f, a

    # 1. SMS Distribution
    f, a = fig(); vc = sms_df["label"].value_counts()
    bars = a.bar(["Ham (Legit)", "Spam"], [vc.get("ham", 0), vc.get("spam", 0)], color=PALETTE, width=0.5)
    for b, c in zip(bars, [vc.get("ham", 0), vc.get("spam", 0)]):
        a.text(b.get_x()+b.get_width()/2, b.get_height()+30, f"{c:,}", ha="center", fontweight="bold")
    a.set_title("SMS Distribution", fontweight="bold"); a.set_ylabel("Count"); a.yaxis.grid(True, alpha=0.3)
    plt.tight_layout(); charts["sms_dist"] = f; plt.close()

    # 2. SMS Accuracy
    f, a = fig(); names = list(sms_results.keys()); accs = [sms_results[n]["accuracy"]*100 for n in names]
    bars = a.bar(names, accs, color=["#6c63ff","#48bb78","#f6ad55"], width=0.5)
    for b, acc in zip(bars, accs):
        a.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{acc:.2f}%", ha="center", fontweight="bold")
    a.set_ylim(80, 102); a.set_title("SMS Accuracy Comparison", fontweight="bold"); a.set_ylabel("Accuracy (%)")
    a.yaxis.grid(True, alpha=0.3); plt.tight_layout(); charts["sms_acc"] = f; plt.close()

    # 3. SMS Confusion Matrix
    f, a = fig(5, 4); cm = confusion_matrix(y_te, best_sms["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="plasma",
                xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"], ax=a)
    a.set_title(f"SMS Confusion Matrix\n({best_sms_name})", fontweight="bold")
    a.set_ylabel("Actual"); a.set_xlabel("Predicted")
    plt.tight_layout(); charts["sms_cm"] = f; plt.close()

    # 4. Word Cloud
    spam_text = " ".join(sms_df[sms_df["label"]=="spam"]["cleaned"].tolist())
    wc = WordCloud(width=800, height=380, background_color="#1c1c2e", colormap="cool", max_words=100).generate(spam_text)
    f, a = fig(9, 4); a.imshow(wc, interpolation="bilinear"); a.axis("off")
    a.set_title("Spam Word Cloud", fontweight="bold"); plt.tight_layout(); charts["wordcloud"] = f; plt.close()

    # 5. Call Distribution
    f, a = fig(); vc2 = call_df["label"].value_counts()
    bars = a.bar(["Safe Call","Fraud Call"], [vc2.get("safe",0), vc2.get("fraud",0)], color=PALETTE, width=0.5)
    for b, c in zip(bars, [vc2.get("safe",0), vc2.get("fraud",0)]):
        a.text(b.get_x()+b.get_width()/2, b.get_height()+30, f"{c:,}", ha="center", fontweight="bold")
    a.set_title("Call Distribution", fontweight="bold"); a.set_ylabel("Count"); a.yaxis.grid(True, alpha=0.3)
    plt.tight_layout(); charts["call_dist"] = f; plt.close()

    # 6. Call Accuracy
    f, a = fig(); c_names = list(call_results.keys()); c_accs = [call_results[n]["accuracy"]*100 for n in c_names]
    bars = a.bar(c_names, c_accs, color=["#6c63ff","#48bb78"], width=0.5)
    for b, acc in zip(bars, c_accs):
        a.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{acc:.2f}%", ha="center", fontweight="bold")
    a.set_ylim(80, 102); a.set_title("Call Accuracy Comparison", fontweight="bold"); a.set_ylabel("Accuracy (%)")
    a.yaxis.grid(True, alpha=0.3); plt.tight_layout(); charts["call_acc"] = f; plt.close()

    # 7. Call Confusion Matrix
    f, a = fig(5, 4); cm2 = confusion_matrix(yc_te, best_call["preds"])
    sns.heatmap(cm2, annot=True, fmt="d", cmap="plasma",
                xticklabels=le_call.classes_, yticklabels=le_call.classes_, ax=a)
    a.set_title(f"Call Confusion Matrix\n({best_call_name})", fontweight="bold")
    a.set_ylabel("Actual"); a.set_xlabel("Predicted")
    plt.tight_layout(); charts["call_cm"] = f; plt.close()

    return {
        "sms_model": best_sms["model"], "tfidf": tfidf, "sms_le": le_sms,
        "sms_results": {k: v["accuracy"] for k, v in sms_results.items()},
        "sms_best": best_sms_name, "sms_acc": best_sms["accuracy"],
        "call_model": best_call["model"], "call_scaler": scaler, "call_le": le_call,
        "call_results": {k: v["accuracy"] for k, v in call_results.items()},
        "call_best": best_call_name, "call_acc": best_call["accuracy"],
        "stemmer": stemmer, "stop_words": stop_words, "charts": charts,
        "sms_df": sms_df, "call_df": call_df,
    }

# ════════════════════════════════════════════════════════════════════
#  PHONE NUMBER PARSER
# ════════════════════════════════════════════════════════════════════

def parse_phone_number(raw):
    cleaned = re.sub(r"[\s\-().]", "", raw)
    digits  = re.sub(r"\D", "", cleaned)
    detected_code, detected_region, high_risk = None, None, False
    if cleaned.startswith("+"):
        for prefix, region, risk in _CC_REGIONS:
            if cleaned.startswith(prefix):
                detected_code, detected_region, high_risk = prefix, region, risk
                break
    is_repeated  = len(digits) >= 6 and len(set(digits)) == 1
    is_too_short = 0 < len(digits) < 7
    is_premium   = any(cleaned.lstrip("+0123456789").startswith(p) for p in _PREMIUM_PREFIXES)
    suspicious   = is_repeated or is_too_short or is_premium or high_risk
    return {
        "detected_code":    detected_code,
        "detected_region":  detected_region,
        "high_risk":        high_risk,
        "suspicious":       suspicious,
        "is_international": detected_code is not None and detected_code != "+1",
    }

# ════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state["history"] = []

def _add_history(record):
    st.session_state["history"].append(record)
    if len(st.session_state["history"]) > 200:
        st.session_state["history"].pop(0)

# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.caption("AI-Powered Fraud Detection")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "💬 SMS Detection", "📞 Call Detection", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python · Scikit-learn · NLTK · Streamlit")

with st.spinner("🧠 Training ML models on first run… (~30 seconds)"):
    M = _train_models()

# ════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.title("🛡️ FraudShield — AI Fraud Detection System")
    st.markdown(
        "Real-time detection of **spam SMS messages** and **fraudulent phone calls** "
        "using Machine Learning and NLP. Trained on 8,000 samples across 3 algorithms."
    )
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📩 SMS Accuracy",     f"{M['sms_acc']*100:.2f}%")
    c2.metric("📞 Call Accuracy",    f"{M['call_acc']*100:.2f}%")
    c3.metric("🤖 ML Models",        "3")
    c4.metric("🔍 Predictions Made", len(st.session_state["history"]))
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💬 SMS Spam Detection")
        st.markdown("Uses NLP to classify any SMS as **Spam** or **Legitimate**.\n\n"
                    "**Pipeline:** Tokenize → Remove stopwords → Stem → TF-IDF → Classifier")
        for name, acc in M["sms_results"].items():
            st.progress(acc, text=f"{name}: {acc*100:.2f}%")
    with col2:
        st.subheader("📞 Call Fraud Detection")
        st.markdown("Analyses call metadata to classify calls as **Fraud** or **Safe**.\n\n"
                    "**Features:** Duration · Frequency · Country Code · International · Time")
        for name, acc in M["call_results"].items():
            st.progress(acc, text=f"{name}: {acc*100:.2f}%")
    st.divider()
    st.subheader("📈 How It Works")
    for title, desc in [
        ("1. Data Collection",  "8,000 records — SMS Spam Collection + simulated call fraud data"),
        ("2. Preprocessing",    "Text cleaning, tokenization, stemming, stopword removal, TF-IDF"),
        ("3. Model Training",   "3 SMS + 2 Call classifiers compared; best selected automatically"),
        ("4. Prediction",       "New input preprocessed identically → instant classification + confidence"),
    ]:
        with st.expander(title, expanded=True):
            st.write(desc)

# ════════════════════════════════════════════════════════════════════
#  PAGE: SMS DETECTION
# ════════════════════════════════════════════════════════════════════

elif page == "💬 SMS Detection":
    st.title("💬 SMS Spam Detection")
    st.markdown("Paste any SMS message. The model classifies it as **Spam** or **Legitimate** with confidence.")
    st.divider()
    col_form, col_info = st.columns([3, 2])

    with col_form:
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("⚠️ Spam Sample"):
            st.session_state["sms_input"] = (
                "WINNER!! You have been selected to receive a $1000 Walmart gift card. "
                "Call 1-800-FAKE-WIN now to claim your prize! Limited time offer."
            )
        if b2.button("✅ Legit Sample"):
            st.session_state["sms_input"] = (
                "Hey, are you coming to the party tonight? Let me know by 6pm "
                "so I can give you the address. Looking forward to seeing you!"
            )

        message = st.text_area(
            "SMS Message",
            value=st.session_state.get("sms_input", ""),
            placeholder="Type or paste an SMS message here…",
            height=160,
        )
        st.caption(f"{len(message)} characters")

        if st.button("🔍 Analyse Message", type="primary", use_container_width=True):
            if not message.strip():
                st.warning("Please enter an SMS message.")
            else:
                cleaned  = _preprocess_text(message, M["stemmer"], M["stop_words"])
                vec      = M["tfidf"].transform([cleaned])
                pred_enc = M["sms_model"].predict(vec)[0]
                proba    = M["sms_model"].predict_proba(vec)[0]
                label    = M["sms_le"].inverse_transform([pred_enc])[0]
                conf     = float(max(proba)) * 100
                is_spam  = label == "spam"

                st.divider()
                if is_spam:
                    st.error(f"🚨 **Spam Message** — Confidence: {conf:.1f}%")
                else:
                    st.success(f"✅ **Legitimate Message** — Confidence: {conf:.1f}%")
                st.progress(conf / 100, text=f"Confidence: {conf:.1f}%")

                with st.expander("📋 Details"):
                    d1, d2 = st.columns(2)
                    d1.metric("Prediction", "Spam" if is_spam else "Ham")
                    d2.metric("Confidence", f"{conf:.2f}%")
                    st.write(f"**Preprocessed:** `{cleaned[:200]}`")

                _add_history({"type": "sms", "prediction": "Spam" if is_spam else "Legitimate",
                              "confidence": round(conf, 2), "is_fraud": is_spam,
                              "detail": message[:80] + ("…" if len(message) > 80 else "")})

    with col_info:
        st.subheader("Active Model")
        st.info(f"**{M['sms_best']}**\nAccuracy: {M['sms_acc']*100:.2f}%")
        st.subheader("All Models")
        for name, acc in M["sms_results"].items():
            st.progress(acc, text=f"{name}: {acc*100:.2f}%")
        st.subheader("NLP Pipeline")
        for i, s in enumerate(["Lowercase", "Remove URLs & numbers", "Tokenize",
                                "Remove stopwords", "Stem words", "TF-IDF vectorize"], 1):
            st.markdown(f"**{i}.** {s}")
        st.subheader("📊 Charts")
        st.pyplot(M["charts"]["sms_dist"])
        st.pyplot(M["charts"]["wordcloud"])
        st.pyplot(M["charts"]["sms_cm"])

# ════════════════════════════════════════════════════════════════════
#  PAGE: CALL DETECTION
# ════════════════════════════════════════════════════════════════════

elif page == "📞 Call Detection":
    st.title("📞 Call Fraud Detection")
    st.markdown("Enter a **phone number** and call details to classify the call as **Fraud** or **Safe**.")
    st.divider()
    col_form, col_info = st.columns([3, 2])

    with col_form:
        pb1, pb2, _ = st.columns([1, 1, 2])
        if pb1.button("⚠️ Fraud Sample"):
            st.session_state.update({"c_phone": "+2348012345678", "c_duration": 2,
                                     "c_freq": 18, "c_intl": "Yes", "c_unk": "Yes",
                                     "c_tod": "late_night (10pm – 4am)"})
        if pb2.button("✅ Safe Sample"):
            st.session_state.update({"c_phone": "+15551234567", "c_duration": 245,
                                     "c_freq": 2, "c_intl": "No", "c_unk": "No",
                                     "c_tod": "afternoon (12pm – 6pm)"})

        phone_input = st.text_input(
            "📱 Phone Number",
            value=st.session_state.get("c_phone", ""),
            placeholder="e.g. +15551234567 or +2348012345678",
            help="Include country code (+1, +44, +234…). Auto-detects region and flags suspicious patterns.",
        )

        phone_info = {}
        if phone_input.strip():
            phone_info = parse_phone_number(phone_input.strip())
            fi1, fi2, fi3 = st.columns(3)
            if phone_info.get("detected_code"):
                fi1.success(f"**Code:** {phone_info['detected_code']}")
                fi2.info(f"**Region:** {phone_info['detected_region']}")
            else:
                fi1.warning("**Code:** Unknown")
                fi2.warning("**Region:** Unrecognised")
            if phone_info.get("suspicious"):
                fi3.error("⚠️ Suspicious")
            else:
                fi3.success("✅ No flags")

        st.divider()
        r1, r2 = st.columns(2)
        call_duration = r1.number_input("⏱ Call Duration (seconds)", 0, 7200,
                                         value=st.session_state.get("c_duration", 245))
        frequency     = r2.number_input("🔄 Calls Per Day", 1, 50,
                                         value=st.session_state.get("c_freq", 3))

        cc_options = ["+1 (USA / Canada)", "+44 (UK)", "+61 (Australia)",
                      "+49 (Germany)", "+91 (India)", "+234 (Nigeria)", "+880 (Bangladesh)"]
        default_cc = 0
        if phone_info.get("detected_code"):
            for i, opt in enumerate(cc_options):
                if opt.startswith(phone_info["detected_code"]):
                    default_cc = i; break

        tod_options = {
            "morning (6am – 12pm)":      "morning",
            "afternoon (12pm – 6pm)":    "afternoon",
            "evening (6pm – 10pm)":      "evening",
            "early_morning (4am – 6am)": "early_morning",
            "late_night (10pm – 4am)":   "late_night",
        }
        r3, r4 = st.columns(2)
        cc_sel  = r3.selectbox("🌐 Country Code", cc_options, index=default_cc,
                               help="Auto-filled from phone number")
        tod_lbl = r4.selectbox("🕐 Time of Day", list(tod_options.keys()))

        r5, r6 = st.columns(2)
        default_intl = "Yes" if phone_info.get("is_international") else \
                       st.session_state.get("c_intl", "No")
        intl_val = r5.radio("✈️ International?", ["Yes", "No"],
                             index=0 if default_intl == "Yes" else 1, horizontal=True)
        unk_val  = r6.radio("❓ Unknown Number?", ["Yes", "No"],
                             index=0 if st.session_state.get("c_unk", "No") == "Yes" else 1,
                             horizontal=True)

        st.divider()
        if st.button("🛡️ Analyse Call", type="primary", use_container_width=True):
            cc_raw = phone_info.get("detected_code") or cc_sel.split(" ")[0]
            is_int = 1 if (phone_info.get("is_international") if phone_input else intl_val == "Yes") else 0
            unk    = 1 if unk_val == "Yes" else 0
            tod    = tod_options[tod_lbl]
            eff_f  = frequency + 5 if (phone_info.get("suspicious") and frequency < 8) else frequency

            feats    = np.array([[call_duration, eff_f, CC_MAP.get(cc_raw, 7),
                                   is_int, unk, TOD_MAP.get(tod, 2)]])
            feats_sc = M["call_scaler"].transform(feats)
            pred_enc = M["call_model"].predict(feats_sc)[0]
            proba    = M["call_model"].predict_proba(feats_sc)[0]
            label    = M["call_le"].inverse_transform([pred_enc])[0]
            conf     = float(max(proba)) * 100
            is_fraud = label == "fraud"

            st.divider()
            if is_fraud:
                st.error(f"🚨 **Fraudulent Call** — Confidence: {conf:.1f}%")
            else:
                st.success(f"✅ **Safe Call** — Confidence: {conf:.1f}%")
            st.progress(conf / 100, text=f"Confidence: {conf:.1f}%")

            if phone_input:
                st.markdown(f"📱 **Number:** `{phone_input}`")
            if phone_info.get("detected_region"):
                st.markdown(f"🌐 **Region:** {phone_info['detected_region']}")
            if phone_info.get("suspicious"):
                st.warning("⚠️ Suspicious number pattern detected.")

            with st.expander("📋 Feature Summary"):
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Duration",   f"{call_duration}s")
                fc2.metric("Frequency",  f"{frequency}×/day")
                fc3.metric("Confidence", f"{conf:.2f}%")

            _add_history({"type": "call", "prediction": "Fraud" if is_fraud else "Safe",
                          "confidence": round(conf, 2), "is_fraud": is_fraud,
                          "detail": f"{phone_input or cc_raw} · {call_duration}s · {frequency}×/day"})

    with col_info:
        st.subheader("Active Model")
        st.info(f"**{M['call_best']}**\nAccuracy: {M['call_acc']*100:.2f}%")
        st.subheader("All Models")
        for name, acc in M["call_results"].items():
            st.progress(acc, text=f"{name}: {acc*100:.2f}%")
        st.subheader("🚩 Suspicious Patterns")
        for f, d in [("Repeated digits", "e.g. 0000000000"),
                     ("Too short (<7 digits)", "Robocall pattern"),
                     ("Premium-rate prefix", "0900, 0976, 0871…"),
                     ("High-risk region", "+234 NG, +880 BD, +7 RU"),
                     ("No country code", "Unverified origin")]:
            st.markdown(f"⚠️ **{f}** — {d}")
        st.subheader("📊 Charts")
        st.pyplot(M["charts"]["call_dist"])
        st.pyplot(M["charts"]["call_acc"])
        st.pyplot(M["charts"]["call_cm"])

# ════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")
    st.divider()
    history  = st.session_state["history"]
    sms_hist = [h for h in history if h["type"] == "sms"]
    cal_hist = [h for h in history if h["type"] == "call"]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total",          len(history))
    c2.metric("SMS Analysed",   len(sms_hist))
    c3.metric("SMS Spam",       sum(1 for h in sms_hist if h["is_fraud"]))
    c4.metric("Calls Analysed", len(cal_hist))
    c5.metric("Fraud Calls",    sum(1 for h in cal_hist if h["is_fraud"]))
    c6.metric("Safe / Legit",   sum(1 for h in history if not h["is_fraud"]))

    st.divider()
    st.subheader("🤖 Model Performance")
    mp1, mp2 = st.columns(2)
    with mp1:
        st.markdown(f"**SMS — {M['sms_best']}**")
        st.metric("Accuracy", f"{M['sms_acc']*100:.2f}%")
        for name, acc in M["sms_results"].items():
            st.progress(acc, text=f"{name}: {acc*100:.2f}%")
    with mp2:
        st.markdown(f"**Call — {M['call_best']}**")
        st.metric("Accuracy", f"{M['call_acc']*100:.2f}%")
        for name, acc in M["call_results"].items():
            st.progress(acc, text=f"{name}: {acc*100:.2f}%")

    st.divider()
    st.subheader("📈 Visualisations")
    ch1, ch2 = st.columns(2)
    ch1.pyplot(M["charts"]["sms_dist"]); ch2.pyplot(M["charts"]["call_dist"])
    ch3, ch4 = st.columns(2)
    ch3.pyplot(M["charts"]["sms_acc"]);  ch4.pyplot(M["charts"]["call_acc"])
    ch5, ch6 = st.columns(2)
    ch5.pyplot(M["charts"]["sms_cm"]);   ch6.pyplot(M["charts"]["call_cm"])
    st.pyplot(M["charts"]["wordcloud"])

    st.divider()
    st.subheader("🕐 Prediction History")
    if history:
        df_h = pd.DataFrame(list(reversed(history)))
        df_h["Result"] = df_h.apply(
            lambda r: ("🚨 " if r["is_fraud"] else "✅ ") + r["prediction"], axis=1)
        st.dataframe(df_h[["type","Result","confidence","detail"]].rename(columns={
            "type":"Type","confidence":"Confidence (%)","detail":"Details"}),
            use_container_width=True)
        buf = io.StringIO()
        pd.DataFrame(history).to_csv(buf, index=False)
        st.download_button("⬇️ Download Report (CSV)", buf.getvalue().encode(),
                           file_name="fraudshield_report.csv", mime="text/csv")
    else:
        st.info("No predictions yet. Use SMS or Call Detection to get started.")

# ════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("**FraudShield** — Call and SMS Fraud Detection System for internship submission.")
    st.divider()
    ab1, ab2 = st.columns([3, 2])
    with ab1:
        st.subheader("🎯 Objective")
        st.markdown("Detect fraudulent phone calls and spam SMS using ML and NLP.")
        st.subheader("🤖 ML Models")
        for name, desc in [
            ("Naive Bayes", "Probabilistic classifier, excellent with TF-IDF features."),
            ("Logistic Regression", "Linear classifier with L2 regularisation."),
            ("Random Forest", "Ensemble of decision trees, best on call metadata."),
        ]:
            with st.expander(name): st.write(desc)
        st.subheader("🔤 NLP Pipeline")
        for i, s in enumerate(["Clean text (remove URLs, numbers)", "Lowercase",
                                "Tokenize", "Remove stopwords", "Stem (Porter)",
                                "TF-IDF vectorize (5000 features, bigrams)"], 1):
            st.markdown(f"**{i}.** {s}")
        st.subheader("⚙️ Run Locally")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    with ab2:
        st.subheader("🧰 Tech Stack")
        for name, desc in [("Python 3.10+","Core"), ("Streamlit","UI"), ("Scikit-learn","ML"),
                            ("NLTK","NLP"), ("Pandas","Data"), ("NumPy","Math"),
                            ("Matplotlib","Charts"), ("Seaborn","Heatmaps"), ("WordCloud","Viz")]:
            st.markdown(f"**{name}** — {desc}")
        st.divider()
        st.subheader("📊 Datasets")
        st.markdown("**SMS:** 4,000 messages (800 spam / 3,200 ham)\n\n**Call:** 4,000 records (1,000 fraud / 3,000 safe)")
        st.divider()
        st.subheader("🚀 Future Scope")
        for f in ["BERT / LSTM deep learning", "Live phone blacklist DB",
                  "Audio call analysis", "Multi-language SMS", "Mobile app"]:
            st.markdown(f"→ {f}")
        if M:
            st.divider()
            st.metric("SMS Best", M["sms_best"],  f"{M['sms_acc']*100:.2f}%")
            st.metric("Call Best", M["call_best"], f"{M['call_acc']*100:.2f}%")
