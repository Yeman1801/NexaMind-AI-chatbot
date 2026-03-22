import streamlit as st
import random
import json
import base64
import re
import string
import emoji
import os

from datetime import datetime

# NLP libs
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- NLTK SETUP ----------------
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ---------------- NORMALIZE (GEN-Z + HINGLISH) ----------------
def normalize(text):
    text = text.lower()

    # extra letters control (heyyyy → heyy)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    replacements = {
        # greetings
        "hlo":"hello","hlw":"hello","helo":"hello","hii":"hi","heyy":"hey",
        "yo":"hello","sup":"what is up",

        # chat
        "hw":"how","r":"are","u":"you","ur":"your",
        "wru":"where are you","wyd":"what are you doing",

        # short forms
        "brb":"be right back","ttyl":"talk to you later",
        "idk":"i do not know","btw":"by the way","omg":"oh my god",

        # thanks / sorry
        "thx":"thanks","tnx":"thanks","ty":"thank you",
        "sry":"sorry",

        # genz
        "bro":"bro","bruh":"bro",
        "lit":"awesome","fire":"awesome",
        "sus":"suspicious",
        "noob":"beginner","pro":"expert",

        # latest
        "fr":"for real","ngl":"not gonna lie",
        "tbh":"to be honest","rn":"right now",

        # shortcuts
        "pls":"please","plz":"please",
        "bcz":"because","coz":"because",

        # about bot
        "who r u":"who are you",
        "wat r u":"what are you",
        "wat can u do":"what can you do",
        "ur name":"your name",
        "about u":"about you"
    }

    words = text.split()
    words = [replacements.get(w, w) for w in words]

    return " ".join(words)


# ---------------- PREPROCESS ----------------
def preprocess(text):
    text = normalize(text)  # 🔥 IMPORTANT (pehle normalize)

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = emoji.demojize(text)

    # punctuation remove
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = []
    for w in text.split():
        if w not in stop_words:
            w = lemmatizer.lemmatize(w)
            w = ps.stem(w)
            words.append(w)

    return " ".join(words)


# ---------------- LOAD DATA ----------------
questions = [
    "What is the latest news about AI?"
]

answers = {
    0: [
        "AI adoption is increasing rapidly across industries worldwide.",
        "Major AI companies are investing heavily in generative AI research.",
        "AI is influencing job markets and automation strategies."
    ]
}

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform([preprocess(q) for q in questions])


# ---------------- GREETING ----------------
greet_words = ["hi","hello","hey","namaste","hola","salam"]
bye_words = ["bye","exit","goodbye","alvida"]


# ---------------- ABOUT BOT HANDLING ----------------
bot_keywords = [
    "who are you", "what are you", "your name", "about you", "yourself",
    "what can you do", "your capabilities", "your purpose", "what do you do",
    "tell me about yourself", "introduce yourself", "kaun ho tum", "tum kya ho",
    "tum kya kar sakte ho", "tumhara naam kya hai", "about bot"
]

def is_about_bot(text):
    # text is already normalized
    if any(phrase in text for phrase in bot_keywords):
        return True
    return False


def get_reply(user_text):
    user_text = normalize(user_text)
    text = preprocess(user_text)

    # Greeting
    if any(word in text for word in greet_words):
        return random.choice([
            "👋 Hello! Kaise help kar sakta hu tumhari?",
            "😊 Hi! Main tumhara AI assistant hu.",
            "🙏 Namaste! Batao kya help chahiye?"
        ])

    # Bye
    if any(word in text for word in bye_words):
        return random.choice([
            "👋 Bye! Milte hain jaldi!",
            "😊 Take care! Fir milenge!",
            "🌙 Good bye! Apna khayal rakhna!"
        ])

    # ABOUT BOT
    if is_about_bot(user_text):
        return random.choice([
            "🤖 I am an AI assistant who can provide information on various topics – technology, economy, health, education, sports, space, and much more.",
            "🙋‍♂️ I am your assistant. I can answer questions in 10+ categories (AI, business, science, sports, etc.). Just ask!",
            "✨ I am an AI chatbot here to help you. I can give you the latest news, trends, and information.",
            "📌 My name is Premium AI Chatbot. I provide updates and information on various subjects.",
            "💬 I am a premium AI assistant. You can ask me about anything – AI, healthcare, business, space, etc."
        ])

    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, matrix)[0]
    idx = sim.argmax()

    if sim[idx] >= 0.4:  # increased threshold for better matching
        return random.choice(answers[idx])

    return random.choice([
        "🤔 I didn't understand. Could you please clarify?",
        "💡 I'm still learning. Your question seems different; can you give more details?",
        "🧠 I'm not updated on this topic right now. Ask me something else?",
        "🌟 Great question! I'll learn about this soon. Anything else for now?",
        "📚 I don't have information on that topic. Please ask another question."
    ])


# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d%H%M%S")

# ---------------- FILE STORAGE ----------------
CHAT_DIR = "all_chats"
os.makedirs(CHAT_DIR, exist_ok=True)

def save_chat(chat_id, data):
    with open(f"{CHAT_DIR}/{chat_id}.json", "w") as f:
        json.dump(data, f, indent=4)

def load_chat(chat_id):
    with open(f"{CHAT_DIR}/{chat_id}.json", "r") as f:
        return json.load(f)

# ---------------- BACKGROUND ----------------
uploaded_bg = st.sidebar.file_uploader("Upload Background", type=["jpg","png"])

def get_base64(file):
    return base64.b64encode(file.read()).decode()

if uploaded_bg:
    img_base64 = get_base64(uploaded_bg)
    bg_css = f"data:image/jpg;base64,{img_base64}"
else:
    bg_css = "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa"

# ---------------- UI ----------------
st.markdown(f"""
<style>

/* Background */
.stApp {{
    background-image: url("{bg_css}");
    background-size: cover;
    background-attachment: fixed;
}}

/* REMOVE DARK BAR COMPLETELY */
[data-testid="stBottomBlockContainer"] {{
    background: transparent !important;
}}

/* INPUT GLASS EFFECT */
.stChatInputContainer {{
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(20px);
    border-radius: 50px;
    padding: 10px 20px;
    margin: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}}

/* TEXT COLOR */
textarea {{
    color: white !important;
}}

/* USER */
.chat-user {{
    background: linear-gradient(135deg,#6c63ff,#8f94fb);
    padding: 12px;
    border-radius: 20px 20px 5px 20px;
    color: white;
    margin: 8px;
    max-width: 60%;
    margin-left: auto;
}}

/* BOT */
.chat-bot {{
    background: rgba(0,0,0,0.45);
    backdrop-filter: blur(10px);
    padding: 12px;
    border-radius: 20px 20px 20px 5px;
    color: white;
    margin: 8px;
    max-width: 60%;
}}

/* TIME */
.time {{
    font-size: 10px;
    color: #ccc;
}}

/* TYPING */
.typing {{
    font-size: 12px;
    color: #aaa;
    margin-left: 10px;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📜 Chat History")

files = os.listdir(CHAT_DIR)
files.sort(reverse=True)

for i, f in enumerate(files):
    if st.sidebar.button(f"Chat {f[:8]}", key=f"chat_{i}"):
        st.session_state.chat = load_chat(f.replace(".json",""))
        st.session_state.chat_id = f.replace(".json","")

# ---------------- TITLE ----------------
st.title("🚀 Premium AI Chatbot")

# ---------------- DISPLAY ----------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>{msg['text']}<div class='time'>{msg['time']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>{msg['text']}<div class='time'>{msg['time']}</div></div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.chat_input("💬 Type message...")

if user_input:
    now = datetime.now().strftime("%H:%M")

    st.session_state.chat.append({"role":"user","text":user_input,"time":now})

    # typing effect
    typing_placeholder = st.empty()
    typing_placeholder.markdown("<div class='typing'>🤖 typing...</div>", unsafe_allow_html=True)

    reply = get_reply(user_input)

    typing_placeholder.empty()

    st.session_state.chat.append({"role":"bot","text":reply,"time":now})

    save_chat(st.session_state.chat_id, st.session_state.chat)
    st.rerun()

# ---------------- NEW CHAT ----------------
if st.sidebar.button("➕ New Chat", key="new_chat"):
    st.session_state.chat = []
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
    st.rerun()