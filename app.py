"""
=============================================================================
🎬 YOUTUBE VIDEO CHATBOT PRO  —  Enhanced College Edition  v2.0
=============================================================================
Stack  : Streamlit + LangChain + Google Gemini + FAISS + YouTube Transcript API + Whisper
New    : ✅ Concept Revision (timestamp + LLM explanation)
         ✅ Parallel chunked Whisper for long videos (near-realtime first answers)
         ✅ Completely revamped glassmorphism UI
=============================================================================
"""

import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import json
import os
import concurrent.futures
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# 🔍  AUTO-DETECT optional dependencies (Whisper + yt-dlp + ffmpeg)
# ---------------------------------------------------------------------------
def _check_whisper_available() -> tuple[bool, str]:
    """Returns (is_available, reason_if_not)."""
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False, "openai-whisper not installed"
    try:
        from yt_dlp import YoutubeDL  # noqa: F401
    except ImportError:
        return False, "yt-dlp not installed"
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return False, "ffmpeg not found on PATH"
    except Exception:
        return False, "ffmpeg not found on PATH"
    return True, ""

WHISPER_AVAILABLE, WHISPER_MISSING_REASON = _check_whisper_available()

# ---------------------------------------------------------------------------
# 🔑  API KEY — loaded in priority order:
#   1. Streamlit Secrets  (recommended for cloud deployment)
#   2. Environment variable GOOGLE_API_KEY
#   3. Hardcoded value below (leave blank for sidebar-only entry)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = ""   # ← hardcode here OR leave blank and use sidebar/secrets

# Auto-load from Streamlit Secrets if available (Streamlit Cloud / HF Spaces)
try:
    _secret_key = st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
    if _secret_key:
        GEMINI_API_KEY = _secret_key
        os.environ["GOOGLE_API_KEY"] = _secret_key
except Exception:
    pass  # st.secrets not available (local run without secrets.toml)

# Auto-load from environment variable (Railway, Render, Docker, etc.)
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Page config (MUST be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎬 YouTube Chatbot Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 🎨  CUSTOM CSS  — Glassmorphism + Neon Accent Dark Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #060918 0%, #0d1333 40%, #0a0a1a 100%);
    color: #e8eaf6;
}
/* Animated noise overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,19,51,0.98) 0%, rgba(6,9,24,0.98) 100%);
    border-right: 1px solid rgba(99,102,241,0.25);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #c7d2fe !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* ── Headings ── */
h1,h2,h3,h4 { color: #ffffff !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.10) 50%, rgba(236,72,153,0.12) 100%);
    border: 1px solid rgba(99,102,241,0.35);
    padding: 28px 36px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: -2px;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899, #6366f1);
    background-size: 300% 300%;
    border-radius: 22px;
    z-index: -1;
    animation: borderSpin 4s linear infinite;
    opacity: 0.4;
}
@keyframes borderSpin { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero-banner h1 { font-size: 32px; font-weight: 800; margin: 0; background: linear-gradient(135deg,#a5b4fc,#f0abfc,#fbcfe8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-banner p  { color: #a5b4fc; font-size: 15px; margin: 8px 0 0; }
.hero-badge { display:inline-block; background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.4); color:#a5b4fc; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; margin-top:8px; letter-spacing:1px; }

/* ── Glass Cards (metric, msg, quiz, etc.) ── */
.glass {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
}
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(168,85,247,0.12) 100%);
    border: 1px solid rgba(99,102,241,0.30);
    padding: 20px 16px;
    border-radius: 16px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(99,102,241,0.25); }
.metric-card .stat-num { font-size: 36px; font-weight: 800; background: linear-gradient(135deg,#a5b4fc,#f0abfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.metric-card .stat-label { font-size: 12px; color: #a5b4fc; margin-top: 4px; font-weight: 500; letter-spacing: 0.5px; }

/* ── Chat Bubbles ── */
.user-msg {
    background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(139,92,246,0.30));
    border: 1px solid rgba(99,102,241,0.4);
    color: #e8eaf6;
    padding: 14px 20px;
    border-radius: 20px 20px 4px 20px;
    margin: 10px 0;
    max-width: 82%;
    margin-left: auto;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.25);
}
.bot-msg {
    background: linear-gradient(135deg, rgba(15,30,80,0.60), rgba(30,50,120,0.50));
    border: 1px solid rgba(99,102,241,0.20);
    color: #e8eaf6;
    padding: 14px 20px;
    border-radius: 20px 20px 20px 4px;
    margin: 10px 0;
    max-width: 82%;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
.msg-label { font-size: 11px; color: #a5b4fc; margin-bottom: 6px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── Quiz ── */
.quiz-card {
    background: rgba(13,19,51,0.70);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    padding: 22px;
    border-radius: 16px;
    margin: 14px 0;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
.quiz-card:hover { border-color: rgba(168,85,247,0.5); }
.quiz-q { font-size: 17px; font-weight: 700; color: #e8eaf6; margin-bottom: 14px; line-height: 1.5; }
.quiz-correct { background: rgba(16,185,129,0.15) !important; border-color: #10b981 !important; }
.quiz-wrong   { background: rgba(239,68,68,0.12) !important;  border-color: #ef4444 !important; }

/* ── Concept Revision Card ── */
.revision-card {
    background: linear-gradient(135deg, rgba(236,72,153,0.10), rgba(99,102,241,0.12));
    border: 1px solid rgba(236,72,153,0.30);
    border-left: 5px solid #ec4899;
    padding: 22px;
    border-radius: 16px;
    margin: 14px 0;
    backdrop-filter: blur(10px);
}
.revision-ts-link {
    display: inline-block;
    background: rgba(236,72,153,0.20);
    border: 1px solid rgba(236,72,153,0.40);
    color: #f9a8d4;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    margin: 4px 4px 4px 0;
    transition: background 0.2s;
}
.revision-ts-link:hover { background: rgba(236,72,153,0.40); }

/* ── Summary ── */
.summary-card {
    background: rgba(255,255,255,0.030);
    border: 1px solid rgba(168,85,247,0.25);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
}
.topic-chip {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.30), rgba(168,85,247,0.25));
    border: 1px solid rgba(99,102,241,0.35);
    color: #c7d2fe;
    padding: 5px 14px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
}
.topic-chip:hover { background: rgba(99,102,241,0.50); color: #fff; }

/* ── Transcript row ── */
.ts-row {
    background: rgba(255,255,255,0.025);
    border-left: 3px solid rgba(99,102,241,0.40);
    padding: 10px 16px;
    border-radius: 8px;
    margin: 4px 0;
    transition: background 0.15s;
}
.ts-row:hover { background: rgba(99,102,241,0.08); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.45) !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(99,102,241,0.20); }
.stTabs [data-baseweb="tab"] {
    padding: 10px 22px;
    background: rgba(13,19,51,0.60);
    border-radius: 10px 10px 0 0;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.15);
    border-bottom: none;
    font-weight: 500;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(99,102,241,0.15); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.40), rgba(168,85,247,0.30)) !important;
    color: white !important;
    border-color: rgba(99,102,241,0.50) !important;
}

/* ── Inputs ── */
.stTextInput  > div > div > input,
.stTextArea   > div > div > textarea {
    background: rgba(13,19,51,0.80) !important;
    color: #e8eaf6 !important;
    border: 1px solid rgba(99,102,241,0.40) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {
    border-color: rgba(168,85,247,0.70) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stNumberInput > div > div > input {
    background: rgba(13,19,51,0.80) !important;
    color: #e8eaf6 !important;
    border: 1px solid rgba(99,102,241,0.40) !important;
    border-radius: 12px !important;
}
.stSelectbox > div > div { background: rgba(13,19,51,0.80) !important; border: 1px solid rgba(99,102,241,0.35) !important; border-radius: 12px !important; }
.stSelectbox > div > div > div { color: #e8eaf6 !important; }
.stRadio > div[role="radiogroup"] > label { color: #c7d2fe !important; padding: 5px 0; }
.stSpinner > div { color: #6366f1 !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] > div { background: rgba(99,102,241,0.25) !important; }
.stSlider [data-baseweb="thumb"]  { background: #6366f1 !important; }

/* ── Alerts / Info ── */
.stAlert { background: rgba(99,102,241,0.12) !important; border: 1px solid rgba(99,102,241,0.30) !important; border-radius: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(13,19,51,0.50); }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }

/* ── Live progress badge ── */
.progress-badge {
    display: inline-block;
    background: rgba(16,185,129,0.20);
    border: 1px solid rgba(16,185,129,0.40);
    color: #6ee7b7;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 4px 0;
}

/* ── Footer ── */
.footer { text-align:center; padding:20px; color:#6366f1; font-size:12px; margin-top:30px; opacity:0.7; }

/* ── Section divider ── */
.section-divider { border: none; border-top: 1px solid rgba(99,102,241,0.20); margin: 20px 0; }

/* ── Whisper stream progress ── */
.whisper-progress {
    background: rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 13px;
    color: #a5b4fc;
}
</style>
""",
    unsafe_allow_html=True,
)


# ===========================================================================
# 🛠️  HELPER FUNCTIONS
# ===========================================================================

def extract_video_id(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from any YouTube URL format."""
    pattern = r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def format_timestamp(seconds: float) -> str:
    """Convert seconds → MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_transcript(video_id: str) -> tuple[str, list[dict]]:
    """Fetch the transcript for a YouTube video."""
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)

    chunks = []
    parts = []
    for snippet in transcript_list:
        parts.append(snippet.text)
        chunks.append({
            "text": snippet.text,
            "start": float(snippet.start),
            "duration": float(snippet.duration),
        })

    full_text = " ".join(parts)
    return full_text, chunks


# ---------------------------------------------------------------------------
# ⚡ FAST PARALLEL WHISPER — processes audio in segments concurrently
# ---------------------------------------------------------------------------
def transcribe_with_whisper(
    video_id: str,
    model_size: str = "base",
    progress_callback=None,           # called with (pct: int, msg: str)
    fast_mode: bool = True,           # use parallel segment processing
) -> tuple[str, list[dict]]:
    """
    Fallback: Download audio from YouTube and transcribe with OpenAI Whisper.

    Fast-mode strategy:
      1. Download audio in best quality.
      2. If fast_mode=True, split audio into N_SEGMENTS segments using ffmpeg
         and transcribe each segment in parallel threads, adjusting timestamps.
      3. Reassemble results in order.

    Requires: pip install openai-whisper yt-dlp  + ffmpeg on PATH.
    """
    import tempfile, glob, subprocess

    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "Whisper is not installed. Run: pip install openai-whisper\n"
            "Also install ffmpeg: https://ffmpeg.org/download.html"
        )
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    url = f"https://youtube.com/watch?v={video_id}"
    tmp_dir = tempfile.mkdtemp(prefix="yt_audio_")

    if progress_callback:
        progress_callback(5, "📥 Downloading audio from YouTube…")

    audio_template = os.path.join(tmp_dir, "audio.%(ext)s")

    # ── Anti-403: realistic browser headers ────────────────────────────────
    _BASE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.youtube.com/",
    }

    def _make_ydl_opts(cookie_browser=None):
        opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_template,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "96"}],
            "quiet": True,
            "no_warnings": True,
            "http_headers": _BASE_HEADERS,
            "socket_timeout": 30,
            "retries": 3,
            "noplaylist": True,
        }
        if cookie_browser:
            opts["cookiesfrombrowser"] = (cookie_browser,)
        return opts

    # ── Try Chrome → Edge → Firefox → no cookies (graceful fallback) ───────
    last_error = None
    downloaded = False
    import glob as _glob

    for browser in ["chrome", "edge", "firefox", None]:
        try:
            label = browser if browser else "no-cookie fallback"
            if progress_callback:
                progress_callback(5, f"📥 Downloading audio ({label})…")
            with YoutubeDL(_make_ydl_opts(browser)) as ydl:
                ydl.download([url])
            downloaded = True
            break
        except Exception as exc:
            last_error = exc
            # Remove any partial files before retrying
            for f in _glob.glob(os.path.join(tmp_dir, "audio.*")):
                try: os.remove(f)
                except: pass

    if not downloaded:
        raise RuntimeError(
            f"YouTube blocked the download (HTTP 403 Forbidden).\n"
            f"Details: {last_error}\n\n"
            "How to fix:\n"
            "  1. Open Chrome/Edge and log into YouTube, then retry\n"
            "  2. Update yt-dlp:  pip install -U yt-dlp\n"
            "  3. Try a different video — some videos block all downloads"
        )

    mp3_files = _glob.glob(os.path.join(tmp_dir, "audio.mp3"))
    if not mp3_files:
        raise RuntimeError("Failed to download audio from YouTube.")
    mp3_path = mp3_files[0]

    if progress_callback:
        progress_callback(20, "🔍 Loading Whisper model…")

    model = whisper.load_model(model_size)

    # ── Get audio duration via ffprobe ──────────────────────────────────────
    def get_duration(path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    duration = get_duration(mp3_path)

    chunks: list[dict] = []
    parts: list[str] = []

    if fast_mode and duration and duration > 300:
        # ── Split into segments and process in parallel ───────────────────
        NUM_SEGMENTS = min(8, max(2, int(duration // 180)))   # ~3-min segments, max 8
        seg_duration = duration / NUM_SEGMENTS

        seg_paths = []
        for i in range(NUM_SEGMENTS):
            seg_path = os.path.join(tmp_dir, f"seg_{i:03d}.mp3")
            start_t = i * seg_duration
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start_t), "-t", str(seg_duration),
                 "-i", mp3_path, "-c:a", "libmp3lame", "-q:a", "5", seg_path],
                capture_output=True, timeout=120,
            )
            if os.path.exists(seg_path):
                seg_paths.append((i, seg_path, i * seg_duration))

        if progress_callback:
            progress_callback(35, f"🎙️ Transcribing {len(seg_paths)} segments in parallel…")

        completed = threading.Event()
        lock = threading.Lock()
        results = {}

        def transcribe_segment(idx, path, time_offset):
            try:
                r = model.transcribe(path, verbose=False)
                segs = []
                for seg in r.get("segments", []):
                    text = seg["text"].strip()
                    if text:
                        segs.append({
                            "text": text,
                            "start": float(seg["start"]) + time_offset,
                            "duration": float(seg.get("end", seg["start"])) - float(seg["start"]),
                        })
                with lock:
                    results[idx] = segs
                    done_pct = 35 + int((len(results) / len(seg_paths)) * 55)
                    if progress_callback:
                        progress_callback(done_pct, f"✅ Segment {idx+1}/{len(seg_paths)} done…")
            except Exception as e:
                with lock:
                    results[idx] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(NUM_SEGMENTS, 4)) as executor:
            futures = [
                executor.submit(transcribe_segment, i, path, offset)
                for i, path, offset in seg_paths
            ]
            concurrent.futures.wait(futures)

        # Merge in order
        for i in range(NUM_SEGMENTS):
            chunks.extend(results.get(i, []))

        parts = [c["text"] for c in chunks]

        # Clean up segments
        for _, path, _ in seg_paths:
            try: os.remove(path)
            except: pass

    else:
        # ── Single-pass transcription (short video) ───────────────────────
        if progress_callback:
            progress_callback(30, "🎙️ Transcribing with Whisper (single pass)…")
        result = model.transcribe(mp3_path, verbose=False)
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if not text:
                continue
            parts.append(text)
            chunks.append({
                "text": text,
                "start": float(seg["start"]),
                "duration": float(seg.get("end", seg["start"])) - float(seg["start"]),
            })

    # Cleanup
    try: os.remove(mp3_path)
    except: pass
    try: os.rmdir(tmp_dir)
    except: pass

    if progress_callback:
        progress_callback(100, "✅ Transcription complete!")

    full_text = " ".join(parts) if parts else ""
    return full_text, chunks


# ===========================================================================
# 🔧  VECTOR-STORE
# ===========================================================================

def build_vectorstore(transcript_text: str, chunk_size: int, chunk_overlap: int, embedding_model: str = "embedding-001"):
    """Split transcript → embed → build FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([transcript_text])
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, len(docs)


def retrieve_context(vectorstore, question: str, k: int = 5) -> str:
    """Retrieve the most relevant chunks for a question."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])


# ===========================================================================
# 🤖  LLM-BACKED FEATURES
# ===========================================================================

def get_llm(model_name: str = "gemini-2.0-flash", temperature: float = 0.3):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def chat_with_video(vectorstore, question: str, model_name: str) -> str:
    context = retrieve_context(vectorstore, question, k=5)
    prompt = PromptTemplate(
        template="""You are an expert AI tutor analyzing a YouTube video transcript.

INSTRUCTIONS:
- Answer ONLY using the context below.
- Use clean Markdown: bullet points (- ), **bold**, and short headings (##).
- Add relevant emojis to make the response engaging.
- If the answer is NOT in the context, reply exactly: "I don't know based on the video."
- Keep answers concise but complete (3-8 bullet points).

CONTEXT FROM VIDEO:
{context}

USER QUESTION:
{question}

YOUR ANSWER:""",
        input_variables=["context", "question"],
    )
    chain = prompt | get_llm(model_name, temperature=0.3) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def generate_summary(vectorstore, model_name: str) -> str:
    context = retrieve_context(vectorstore, "summary overview main points", k=12)
    prompt = PromptTemplate(
        template="""You are an expert content summarizer. Create a structured summary of this YouTube video.

Use this EXACT format:

## 📌 One-Line Summary
(One sentence capturing the video's core message)

## 🎯 Key Takeaways
- (5-7 bullet points, each one a complete idea)

## 📚 Main Topics Covered
- (List the major topics/themes)

## 💡 Important Insights
- (2-3 deeper insights or "aha" moments)

## ⚡ Actionable Advice
- (If applicable, practical tips the viewer can apply)

Transcript context:
{context}""",
        input_variables=["context"],
    )
    chain = prompt | get_llm(model_name, temperature=0.4) | StrOutputParser()
    return chain.invoke({"context": context})


def extract_topics(vectorstore, model_name: str) -> list[str]:
    context = retrieve_context(vectorstore, "main topics themes keywords", k=10)
    prompt = PromptTemplate(
        template="""From the video transcript below, extract 8-12 key topics as short keyword phrases.
Return ONLY a JSON array of strings. No markdown, no explanation.

Example output:
["machine learning", "neural networks", "backpropagation", "training data"]

Transcript:
{context}""",
        input_variables=["context"],
    )
    chain = prompt | get_llm(model_name, temperature=0.2) | StrOutputParser()
    raw = chain.invoke({"context": context})

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        topics = json.loads(raw)
        if isinstance(topics, list):
            return [str(t) for t in topics][:12]
    except Exception:
        pass
    return []


def generate_quiz(vectorstore, num_questions: int, difficulty: str, model_name: str) -> list[dict]:
    context = retrieve_context(vectorstore, f"quiz questions {difficulty}", k=12)
    difficulty_guide = {
        "Easy": "Basic recall and definitions. Straightforward questions.",
        "Medium": "Understanding and application. Slightly tricky options.",
        "Hard": "Deep analysis, edge cases, and multi-step reasoning.",
    }[difficulty]

    prompt = PromptTemplate(
        template="""You are an exam creator for college students. Generate {n} multiple-choice questions
based ONLY on the video transcript below.

Difficulty: {difficulty}
Guidance: {guide}

Return STRICT JSON with this structure (no markdown, no extra text):
[
  {{
    "question": "The question text",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_index": 0,
    "explanation": "Why the correct answer is right, in 1-2 sentences."
  }}
]

Rules:
- correct_index is 0-based (0=A, 1=B, 2=C, 3=D).
- All 4 options must be plausible.
- Questions must be answerable from the transcript.

Transcript:
{context}""",
        input_variables=["n", "difficulty", "guide", "context"],
    )

    chain = prompt | get_llm(model_name, temperature=0.5) | StrOutputParser()
    raw = chain.invoke({
        "n": num_questions,
        "difficulty": difficulty,
        "guide": difficulty_guide,
        "context": context,
    })

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        quiz = json.loads(raw)
        if isinstance(quiz, list) and len(quiz) > 0:
            return quiz[:num_questions]
    except Exception:
        pass
    return []


# ===========================================================================
# ✨  NEW FEATURE: CONCEPT REVISION (Timestamp + LLM Explanation)
# ===========================================================================

def find_concept_timestamps(
    transcript_chunks: list[dict],
    concept: str,
) -> list[dict]:
    """
    Fast local search: return chunks whose text mentions the concept keyword(s).
    Returns up to 8 most relevant chunks with start-time info.
    """
    concept_lower = concept.lower()
    keywords = [w for w in re.split(r"\s+", concept_lower) if len(w) > 2]

    scored = []
    for chunk in transcript_chunks:
        text_lower = chunk["text"].lower()
        # Score: number of keyword hits (weighted for phrase match)
        score = sum(text_lower.count(kw) for kw in keywords)
        if concept_lower in text_lower:
            score += 5          # bonus for exact phrase match
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: -x[0])

    # Remove near-duplicate timestamps (within 10 seconds)
    seen_times = []
    filtered = []
    for _, chunk in scored:
        t = chunk["start"]
        if all(abs(t - st) > 10 for st in seen_times):
            seen_times.append(t)
            filtered.append(chunk)
        if len(filtered) >= 8:
            break

    # Sort chronologically
    filtered.sort(key=lambda x: x["start"])
    return filtered


def explain_concept(
    vectorstore,
    concept: str,
    model_name: str,
    timestamp_contexts: list[dict],
) -> str:
    """LLM explanation of the concept grounded in the video's transcript."""
    # Build focused context from timestamp regions + semantic retrieval
    ts_texts = " ".join([c["text"] for c in timestamp_contexts[:4]])
    semantic_ctx = retrieve_context(vectorstore, concept, k=6)
    combined = f"[Timestamp context]\n{ts_texts}\n\n[Semantic context]\n{semantic_ctx}"

    prompt = PromptTemplate(
        template="""You are an expert educator helping a student revise a concept from a YouTube video.

The student wants to revise: **"{concept}"**

Using ONLY the video context below, provide a rich revision guide:

## 🧠 What is {concept}?
(Clear definition in 2-3 sentences, as explained in the video)

## 📖 How the Video Explains It
(3-5 bullet points covering the video's specific explanation, examples, or demonstrations)

## 🔑 Key Points to Remember
(3-4 crisp bullet points — the most important things to memorize)

## 💡 Example / Analogy from the Video
(If any example or analogy was mentioned, describe it)

## ⚡ Quick Revision Summary
(One-paragraph executive summary in simple language)

Video context:
{context}

If the concept is not covered in the video, say so clearly.""",
        input_variables=["concept", "context"],
    )

    chain = prompt | get_llm(model_name, temperature=0.35) | StrOutputParser()
    return chain.invoke({"concept": concept, "context": combined})


# ===========================================================================
# 🧠  SESSION STATE INITIALIZATION
# ===========================================================================
_defaults = {
    "vectorstore": None,
    "transcript_text": "",
    "transcript_chunks": [],
    "chat_history": [],
    "video_id": None,
    "video_processed": False,
    "num_chunks": 0,
    "quiz": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "summary": "",
    "topics": [],
    "transcript_source": "",
    "revision_concept": "",
    "revision_result": None,      # dict: {explanation, timestamps}
    "whisper_progress": 0,
    "whisper_msg": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===========================================================================
# 📺  SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown(
        "<div style='padding:14px 0 6px;'>"
        "<span style='font-size:22px;'>🎬</span> "
        "<span style='font-size:17px;font-weight:700;color:#a5b4fc;'>YouTube Chatbot Pro</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── API Key ──────────────────────────────────────────────────────────────
    st.markdown("**🔑 Gemini API Key**")
    api_key_input = st.text_input(
        "API Key",
        value=GEMINI_API_KEY,
        type="password",
        label_visibility="collapsed",
        help="Get a free key at https://aistudio.google.com/apikey",
        placeholder="AIza…",
    )
    effective_api_key = api_key_input.strip() or GEMINI_API_KEY.strip()
    if effective_api_key:
        os.environ["GOOGLE_API_KEY"] = effective_api_key

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── URL ──────────────────────────────────────────────────────────────────
    video_url = st.text_input(
        "🔗 YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        help="Paste any YouTube video URL that has captions/transcript.",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**⚙️ Model Settings**")

    model_choice = st.selectbox(
        "🤖 Gemini Model",
        options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
    )

    col_cs, col_co = st.columns(2)
    with col_cs:
        chunk_size = st.slider("Chunk Size", 400, 2000, 900, step=100)
    with col_co:
        chunk_overlap = st.slider("Overlap", 50, 400, 180, step=50)

    embedding_model = st.selectbox(
        "🧠 Embedding Model",
        options=["embedding-001", "text-embedding-004", "gemini-embedding-001"],
        index=0,
        help="Switch to 'embedding-001' if you get a 404 error.",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**🎙️ Transcript Source**")

    if WHISPER_AVAILABLE:
        transcript_mode = st.radio(
            "Source",
            options=["Auto (YouTube first, then Whisper)", "YouTube captions only", "Whisper AI only"],
            index=0,
            label_visibility="collapsed",
        )
        whisper_model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="tiny=fastest, large=best quality. 'base' is recommended.",
        )
        fast_whisper = st.checkbox(
            "⚡ Parallel Whisper (faster for long videos)",
            value=True,
            help="Splits audio into segments and transcribes them in parallel threads. ~2-4x faster for long videos.",
        )
    else:
        # Whisper not available — lock to YouTube captions only
        transcript_mode = "YouTube captions only"
        whisper_model_size = "base"
        fast_whisper = False
        st.markdown(
            "<div style='background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.30);"
            "border-radius:10px;padding:12px 14px;font-size:12px;color:#a5b4fc;'>"
            "<b>🎙️ Whisper AI — Not available</b><br>"
            f"<span style='opacity:0.75;'>{WHISPER_MISSING_REASON}.<br>"
            "YouTube captions will be used automatically (covers 95%+ of videos).<br>"
            "To enable Whisper: install <code>openai-whisper</code>, <code>yt-dlp</code> + ffmpeg.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**📝 Quiz Settings**")
    num_quiz_questions = st.slider("Questions", 3, 15, 5, step=1)
    quiz_difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Process Button ────────────────────────────────────────────────────────
    if st.button("🚀 Process Video", type="primary"):
        if not video_url:
            st.warning("Please paste a YouTube URL first.")
        elif not effective_api_key:
            st.error("❌ Please paste your Gemini API key above.")
        else:
            vid = extract_video_id(video_url)
            if not vid:
                st.error("❌ Invalid YouTube URL. Check the URL and try again.")
            else:
                transcript_text = ""
                chunks = []
                source_label = ""

                use_youtube = transcript_mode.startswith("Auto") or transcript_mode.startswith("YouTube")
                use_whisper = transcript_mode.startswith("Whisper") or transcript_mode.startswith("Auto")

                if use_youtube:
                    with st.spinner("📡 Fetching YouTube captions…"):
                        try:
                            transcript_text, chunks = get_transcript(vid)
                            source_label = "YouTube Captions"
                        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
                            transcript_text = ""
                        except Exception:
                            transcript_text = ""

                if (not transcript_text.strip()) and use_whisper:
                    progress_bar = st.progress(0, text="Initializing Whisper…")

                    def whisper_progress_cb(pct, msg):
                        progress_bar.progress(pct, text=msg)

                    try:
                        transcript_text, chunks = transcribe_with_whisper(
                            vid,
                            model_size=whisper_model_size,
                            progress_callback=whisper_progress_cb,
                            fast_mode=fast_whisper,
                        )
                        source_label = f"Whisper AI ({whisper_model_size})"
                        progress_bar.empty()
                    except RuntimeError as re_err:
                        st.error(f"❌ Whisper error: {re_err}")
                        st.info("💡 Install: `pip install openai-whisper yt-dlp` + ffmpeg")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Whisper error: {e}")
                        st.stop()

                if not transcript_text.strip():
                    st.error("❌ No transcript found. Try 'Whisper AI only'.")
                    st.stop()

                with st.spinner("🧠 Building semantic index…"):
                    vs, n_chunks = build_vectorstore(transcript_text, chunk_size, chunk_overlap, embedding_model)

                # Save to session
                st.session_state.vectorstore = vs
                st.session_state.transcript_text = transcript_text
                st.session_state.transcript_chunks = chunks
                st.session_state.video_id = vid
                st.session_state.video_processed = True
                st.session_state.num_chunks = n_chunks
                st.session_state.transcript_source = source_label
                st.session_state.chat_history = []
                st.session_state.quiz = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.summary = ""
                st.session_state.topics = []
                st.session_state.revision_result = None
                st.session_state.revision_concept = ""

                st.success(f"✅ Ready via {source_label}!")
                st.rerun()

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**📊 Status**")
    if st.session_state.video_processed:
        st.markdown(
            f"<span class='progress-badge'>● READY — {st.session_state.transcript_source}</span>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Chunks", st.session_state.num_chunks)
        c2.metric("Words", f"{len(st.session_state.transcript_text.split()):,}")
        c1.metric("Q&A", len(st.session_state.chat_history) // 2)
        c2.metric("Quiz Qs", len(st.session_state.quiz))
    else:
        st.info("No video loaded yet.")

    st.markdown("<div class='footer'>v2.0 · Powered by Gemini + LangChain</div>", unsafe_allow_html=True)


# ===========================================================================
# 🏠  MAIN AREA
# ===========================================================================

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <h1>🎬 YouTube Video Chatbot Pro</h1>
  <p>Chat • Summarize • Revise Concepts • Quiz • Learn — Powered by RAG + Google Gemini</p>
  <span class="hero-badge">v2.0 · COLLEGE EDITION</span>
</div>
""",
    unsafe_allow_html=True,
)

# ===========================================================================
if st.session_state.video_processed and st.session_state.video_id:
    # ── Embedded Video + Metrics ─────────────────────────────────────────────
    vid_col, meta_col = st.columns([2, 3])
    with vid_col:
        st.video(f"https://youtube.com/watch?v={st.session_state.video_id}")
    with meta_col:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.markdown(
            f"<div class='metric-card'><div class='stat-num'>{st.session_state.num_chunks}</div>"
            "<div class='stat-label'>📚 Chunks Indexed</div></div>", unsafe_allow_html=True)
        m2.markdown(
            f"<div class='metric-card'><div class='stat-num'>{len(st.session_state.transcript_text.split()):,}</div>"
            "<div class='stat-label'>📝 Total Words</div></div>", unsafe_allow_html=True)
        m3.markdown(
            f"<div class='metric-card'><div class='stat-num'>{len(st.session_state.chat_history)//2}</div>"
            "<div class='stat-label'>💬 Q&A Exchanges</div></div>", unsafe_allow_html=True)
        m4.markdown(
            f"<div class='metric-card'><div class='stat-num'>{len(st.session_state.quiz)}</div>"
            "<div class='stat-label'>📝 Quiz Questions</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_chat, tab_revision, tab_quiz, tab_summary, tab_transcript = st.tabs(
        ["💬 Chat", "🔁 Concept Revision", "📝 Quiz", "📋 Summary", "📜 Transcript"]
    )

    # =========================================================================
    # TAB 1 — CHAT
    # =========================================================================
    with tab_chat:
        st.markdown("### 💬 Chat with the Video")
        st.caption("Ask any question about the video content. All answers are grounded in the transcript.")

        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_question = st.text_input(
                "Question",
                placeholder="e.g. What is the main topic discussed at the beginning?",
                label_visibility="collapsed",
                key="chat_input",
            )
        with btn_col:
            ask_clicked = st.button("🔍 Ask")

        # Quick suggestion pills
        st.markdown("<div style='margin:8px 0;'>", unsafe_allow_html=True)
        sug_cols = st.columns(4)
        suggestions = [
            "What is this video about?",
            "Explain the main concept simply",
            "What are the key takeaways?",
            "Give me an example from the video",
        ]
        for i, sug in enumerate(suggestions):
            if sug_cols[i].button(sug, key=f"sug_{i}"):
                user_question = sug
                ask_clicked = True
        st.markdown("</div>", unsafe_allow_html=True)

        if ask_clicked and user_question.strip():
            with st.spinner("🤔 Thinking…"):
                try:
                    answer = chat_with_video(
                        st.session_state.vectorstore, user_question, model_choice
                    )
                    st.session_state.chat_history.append(("user", user_question))
                    st.session_state.chat_history.append(("bot", answer))
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Render chat history (most recent first)
        if st.session_state.chat_history:
            for role, msg in reversed(st.session_state.chat_history):
                if role == "user":
                    st.markdown(
                        f"<div class='user-msg'>"
                        f"<div class='msg-label'>🧑 You</div>{msg}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='bot-msg'>"
                        f"<div class='msg-label'>🤖 AI Tutor</div>{msg}</div>",
                        unsafe_allow_html=True,
                    )
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("💡 Ask your first question above to start chatting with the video!")

    # =========================================================================
    # TAB 2 — CONCEPT REVISION  (NEW ✨)
    # =========================================================================
    with tab_revision:
        st.markdown("### 🔁 Concept Revision")
        st.caption(
            "Enter any concept or topic from the video. "
            "We'll find the exact timestamps where it's discussed **and** give you a full LLM-powered explanation."
        )

        rev_col, rev_btn = st.columns([5, 1])
        with rev_col:
            concept_input = st.text_input(
                "Concept",
                placeholder="e.g. backpropagation, photosynthesis, Newton's second law…",
                label_visibility="collapsed",
                key="concept_input",
            )
        with rev_btn:
            revise_clicked = st.button("🔁 Revise")

        # Quick concept suggestions from extracted topics
        if st.session_state.topics:
            st.markdown("**💡 Quick pick from extracted topics:**")
            topic_cols = st.columns(min(len(st.session_state.topics), 4))
            for i, tp in enumerate(st.session_state.topics[:4]):
                if topic_cols[i % 4].button(f"#{tp}", key=f"tp_rev_{i}"):
                    concept_input = tp
                    revise_clicked = True

        if revise_clicked and concept_input.strip():
            with st.spinner(f"🔍 Finding '{concept_input}' in the video…"):
                ts_chunks = find_concept_timestamps(
                    st.session_state.transcript_chunks,
                    concept_input,
                )

            with st.spinner("🧠 Generating revision guide…"):
                try:
                    explanation = explain_concept(
                        st.session_state.vectorstore,
                        concept_input,
                        model_choice,
                        ts_chunks,
                    )
                    st.session_state.revision_concept = concept_input
                    st.session_state.revision_result = {
                        "explanation": explanation,
                        "timestamps": ts_chunks,
                    }
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")

        # ── Show revision result ──────────────────────────────────────────────
        if st.session_state.revision_result:
            res = st.session_state.revision_result
            concept_disp = st.session_state.revision_concept

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Timestamp section
            st.markdown(f"#### ⏱️ Where **\"{concept_disp}\"** appears in the video")
            if res["timestamps"]:
                ts_html = ""
                for chunk in res["timestamps"]:
                    ts = format_timestamp(chunk["start"])
                    ts_sec = int(chunk["start"])
                    video_id = st.session_state.video_id
                    url = f"https://youtube.com/watch?v={video_id}&t={ts_sec}s"
                    preview = chunk["text"][:90].replace("<", "&lt;").replace(">", "&gt;")
                    ts_html += (
                        f"<div class='ts-row'>"
                        f"<a href='{url}' target='_blank' class='revision-ts-link'>▶ {ts}</a>"
                        f"<span style='color:#c7d2fe;font-size:13px;'> — {preview}…</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div class='revision-card'>{ts_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    f"⚠️ The keyword **\"{concept_disp}\"** wasn't found verbatim in the transcript. "
                    "The LLM explanation below is based on semantic search instead."
                )

            # Explanation section
            st.markdown(f"#### 📖 LLM Revision Guide: *{concept_disp}*")
            st.markdown(
                f"<div class='revision-card'>{res['explanation']}</div>",
                unsafe_allow_html=True,
            )

            # Clear button
            if st.button("🗑️ Clear Revision"):
                st.session_state.revision_result = None
                st.session_state.revision_concept = ""
                st.rerun()
        else:
            st.markdown(
                """
<div class="glass" style="padding:30px;text-align:center;margin-top:20px;">
  <div style="font-size:48px;">🔁</div>
  <h3 style="color:#a5b4fc;">Concept Revision Studio</h3>
  <p style="color:#818cf8;">
    Type any concept above (or pick one from your extracted topics).<br>
    We'll pinpoint the exact moments in the video <strong>and</strong> generate a full revision guide.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 3 — QUIZ
    # =========================================================================
    with tab_quiz:
        st.markdown("### 📝 MCQ Quiz Generator")
        st.caption("Test your understanding with auto-generated, auto-graded multiple-choice questions.")

        qcol1, qcol2, qcol3 = st.columns([2, 2, 2])
        with qcol1:
            if st.button("🎯 Generate New Quiz"):
                with st.spinner("Creating quiz questions…"):
                    quiz = generate_quiz(
                        st.session_state.vectorstore,
                        num_quiz_questions,
                        quiz_difficulty,
                        model_choice,
                    )
                    if quiz:
                        st.session_state.quiz = quiz
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate quiz. Try again.")

        with qcol3:
            if st.session_state.quiz and st.session_state.quiz_submitted:
                if st.button("🔄 Reset Quiz"):
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()

        if st.session_state.quiz:
            st.markdown(
                f"<span class='hero-badge'>Difficulty: {quiz_difficulty}</span>"
                f"<span class='hero-badge' style='margin-left:8px;'>Questions: {len(st.session_state.quiz)}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            with st.form("quiz_form"):
                for i, q in enumerate(st.session_state.quiz):
                    st.markdown(
                        f"<div class='quiz-card'><div class='quiz-q'>Q{i+1}. {q['question']}</div></div>",
                        unsafe_allow_html=True,
                    )
                    options = q.get("options", [])
                    choice = st.radio(
                        f"Answer Q{i+1}",
                        options=options,
                        index=st.session_state.quiz_answers.get(i, 0),
                        key=f"q_{i}",
                        label_visibility="collapsed",
                    )
                    if choice and options:
                        st.session_state.quiz_answers[i] = options.index(choice)

                submitted = st.form_submit_button("✅ Submit & Grade Quiz")

            if submitted:
                st.session_state.quiz_submitted = True
                st.rerun()

            if st.session_state.quiz_submitted:
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                st.markdown("## 📊 Quiz Results")

                correct_count = 0
                for i, q in enumerate(st.session_state.quiz):
                    user_ans = st.session_state.quiz_answers.get(i, -1)
                    correct_idx = q.get("correct_index", 0)
                    is_correct = user_ans == correct_idx
                    if is_correct:
                        correct_count += 1

                    options = q.get("options", [])
                    user_letter = chr(65 + user_ans) if user_ans >= 0 else "—"
                    correct_letter = chr(65 + correct_idx)
                    status_emoji = "✅" if is_correct else "❌"
                    card_class = "quiz-card quiz-correct" if is_correct else "quiz-card quiz-wrong"

                    options_html = "".join(
                        f"<div style='padding:5px 10px;margin:3px 0;border-radius:6px;"
                        f"background:rgba(255,255,255,0.04);font-size:13px;'>{opt}</div>"
                        for opt in options
                    )
                    st.markdown(
                        f"<div class='{card_class}'>"
                        f"<div class='quiz-q'>{status_emoji} Q{i+1}. {q['question']}</div>"
                        f"{options_html}"
                        f"<div style='margin-top:10px;font-size:13px;'>"
                        f"<b style='color:#a5b4fc;'>Your:</b> {user_letter} &nbsp;&nbsp;"
                        f"<b style='color:#6ee7b7;'>Correct:</b> {correct_letter}</div>"
                        f"<div style='margin-top:8px;font-size:13px;color:#fcd34d;'>"
                        f"💡 <b>Explanation:</b> {q.get('explanation','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                total = len(st.session_state.quiz)
                pct = (correct_count / total) * 100 if total else 0
                grade = (
                    "🏆 Excellent!" if pct >= 80
                    else "👍 Good job!" if pct >= 60
                    else "📚 Keep studying!" if pct >= 40
                    else "🔄 Review the video again."
                )

                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(
                    f"<div class='metric-card'><div class='stat-num'>{correct_count}/{total}</div>"
                    "<div class='stat-label'>✅ Correct</div></div>",
                    unsafe_allow_html=True,
                )
                sc2.markdown(
                    f"<div class='metric-card'><div class='stat-num'>{pct:.0f}%</div>"
                    "<div class='stat-label'>📊 Score</div></div>",
                    unsafe_allow_html=True,
                )
                sc3.markdown(
                    f"<div class='metric-card'><div class='stat-num' style='font-size:22px;'>{grade}</div>"
                    "<div class='stat-label'>🎯 Result</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
<div class="glass" style="padding:30px;text-align:center;margin-top:20px;">
  <div style="font-size:48px;">📝</div>
  <h3 style="color:#a5b4fc;">MCQ Quiz Generator</h3>
  <p style="color:#818cf8;">
    Click <strong>"Generate New Quiz"</strong> above to create customized multiple-choice questions.<br>
    Set difficulty and question count in the sidebar.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 4 — SUMMARY
    # =========================================================================
    with tab_summary:
        st.markdown("### 📋 Video Summary & Key Topics")
        st.caption("Get an instant structured summary and key topics extracted from the video.")

        s1, s2 = st.columns(2)
        with s1:
            if st.button("📋 Generate Summary"):
                with st.spinner("Summarizing video…"):
                    try:
                        summary = generate_summary(st.session_state.vectorstore, model_choice)
                        st.session_state.summary = summary
                    except Exception as e:
                        st.error(f"Error: {e}")
        with s2:
            if st.button("🏷️ Extract Topics"):
                with st.spinner("Extracting key topics…"):
                    try:
                        topics = extract_topics(st.session_state.vectorstore, model_choice)
                        st.session_state.topics = topics
                        if not topics:
                            st.warning("No topics extracted. Try again.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.topics:
            st.markdown("#### 🏷️ Key Topics")
            chips = "".join(
                f"<span class='topic-chip'>#{t}</span>"
                for t in st.session_state.topics
            )
            st.markdown(
                f"<div style='margin:12px 0;line-height:2.4;'>{chips}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        if st.session_state.summary:
            st.markdown("#### 📋 Structured Summary")
            st.markdown(
                f"<div class='summary-card'>{st.session_state.summary}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="glass" style="padding:30px;text-align:center;margin-top:20px;">
  <div style="font-size:48px;">📋</div>
  <h3 style="color:#a5b4fc;">Video Summary</h3>
  <p style="color:#818cf8;">
    Click <strong>"Generate Summary"</strong> for a structured overview.<br>
    Click <strong>"Extract Topics"</strong> to see keyword chips you can also use in Concept Revision.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 5 — TRANSCRIPT
    # =========================================================================
    with tab_transcript:
        st.markdown("### 📜 Full Transcript with Timestamps")
        st.caption("Browse the full transcript. Click any timestamp to jump to that moment in the video.")

        search_term = st.text_input(
            "🔍 Search transcript",
            placeholder="Type a keyword to filter segments…",
            key="ts_search",
        )

        if st.session_state.transcript_chunks:
            visible = [
                c for c in st.session_state.transcript_chunks
                if not search_term or search_term.lower() in c["text"].lower()
            ]
            st.markdown(
                f"<span class='progress-badge'>Showing {len(visible)} / {len(st.session_state.transcript_chunks)} segments</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Highlight search term in results
            def highlight(text, term):
                if not term:
                    return text
                escaped = re.escape(term)
                return re.sub(
                    f"({escaped})",
                    r"<mark style='background:rgba(99,102,241,0.40);color:#e8eaf6;border-radius:3px;padding:0 2px;'>\1</mark>",
                    text,
                    flags=re.IGNORECASE,
                )

            for chunk in visible:
                ts = format_timestamp(chunk["start"])
                ts_sec = int(chunk["start"])
                video_id = st.session_state.video_id
                url = f"https://youtube.com/watch?v={video_id}&t={ts_sec}s"
                text_hl = highlight(chunk["text"].replace("<", "&lt;").replace(">", "&gt;"), search_term)
                st.markdown(
                    f"<div class='ts-row'>"
                    f"<a href='{url}' target='_blank' style='color:#818cf8;font-weight:700;font-size:12px;text-decoration:none;'>"
                    f"⏱ {ts}</a>"
                    f"<span style='color:#c7d2fe;font-size:14px;margin-left:10px;'>{text_hl}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No transcript segments available.")

# ===========================================================================
# WELCOME SCREEN (no video loaded)
# ===========================================================================
else:
    st.markdown(
        """
<div class="glass" style="padding:50px 40px;text-align:center;margin-top:20px;">
  <div style="font-size:64px;margin-bottom:16px;">🎬</div>
  <h2 style="font-size:26px;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#f0abfc);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Welcome to YouTube Video Chatbot Pro
  </h2>
  <p style="color:#818cf8;font-size:16px;max-width:520px;margin:12px auto;">
    Paste a YouTube URL in the <strong style="color:#a5b4fc;">sidebar</strong> 
    and click <strong style="color:#a5b4fc;">🚀 Process Video</strong> to get started.
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:32px;max-width:680px;margin-left:auto;margin-right:auto;">
    <div class="metric-card"><div style="font-size:28px;">💬</div><div style="font-size:13px;color:#a5b4fc;margin-top:6px;">Chat with any video using RAG</div></div>
    <div class="metric-card"><div style="font-size:28px;">🔁</div><div style="font-size:13px;color:#a5b4fc;margin-top:6px;">Concept Revision with Timestamps</div></div>
    <div class="metric-card"><div style="font-size:28px;">📝</div><div style="font-size:13px;color:#a5b4fc;margin-top:6px;">Auto-graded MCQ Quizzes</div></div>
    <div class="metric-card"><div style="font-size:28px;">📋</div><div style="font-size:13px;color:#a5b4fc;margin-top:6px;">Structured Summaries & Topics</div></div>
    <div class="metric-card"><div style="font-size:28px;">📜</div><div style="font-size:13px;color:#a5b4fc;margin-top:6px;">Searchable Timestamped Transcript</div></div>
    <div class="metric-card"><div style="font-size:28px;">🎙️</div><div style="font-size:13px;color:#a5b4fc;margin-top:6px;">⚡ Parallel Whisper for Long Videos</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    "<div class='footer'>🎬 YouTube Video Chatbot Pro v2.0 · Streamlit + LangChain + Google Gemini + FAISS<br>"
    "Built for Educational Use · © 2025</div>",
    unsafe_allow_html=True,
)
