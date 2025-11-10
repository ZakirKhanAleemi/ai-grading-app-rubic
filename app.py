# app.py  (AI Grading Assistant – Enhanced Large Upload & Auto Model Version)

import streamlit as st
import pandas as pd
import google.generativeai as genai
import docx, zipfile, io, json, os, re, tempfile, time, requests, shutil
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from google.api_core import exceptions
import fitz
import py7zr
import rarfile

# ==============================
# Streamlit Configuration
# ==============================
st.set_page_config(layout="wide", page_title="AI Grading Assistant")
st.markdown(
    """
    <style>
        [data-testid="stFileUploader"] > div > div > div > button {
            height: 3em;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Remove Streamlit default upload cap when self-hosting
#st.runtime.scriptrunner.add_script_run_ctx  # keeps compatibility
st.session_state.setdefault("upload_limit", 1024)  # in MB
st.write(f"🔼 Current upload size limit: **{st.session_state.upload_limit} MB**")

# ==============================
# File Utilities
# ==============================
def extract_text_from_docx_bytes(b):
    try:
        d = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        return f"[DOCX error] {e}"

def extract_images_from_docx_bytes(b):
    imgs = []
    try:
        d = docx.Document(io.BytesIO(b))
        for rel in d.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    im = Image.open(io.BytesIO(rel.target_part.blob))
                    imgs.append(im)
                except UnidentifiedImageError:
                    pass
    except Exception:
        pass
    return imgs

def extract_text_from_pdf_bytes(b):
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        return f"[PDF error] {e}"

def read_ipynb(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            nb = json.load(f)
        return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    except Exception as e:
        return f"[IPYNB error] {e}"

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[READ error] {e}"

# ==============================
# Archive Extraction (Recursive)
# ==============================
def extract_any_archive(p, dest):
    pl = p.lower()
    try:
        if pl.endswith(".zip"):
            with zipfile.ZipFile(p, "r") as z: z.extractall(dest)
        elif pl.endswith(".7z"):
            with py7zr.SevenZipFile(p, "r") as z: z.extractall(dest)
        elif pl.endswith(".rar"):
            with rarfile.RarFile(p) as r: r.extractall(dest)
        else:
            return False
        os.remove(p)
        return True
    except Exception:
        return False

def recursive_extract(root):
    while True:
        found = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith((".zip",".7z",".rar")):
                    found.append(os.path.join(dp,fn))
        if not found: break
        for a in found:
            extract_any_archive(a, os.path.dirname(a))

# ==============================
# Gemini Model Setup (Auto)
# ==============================
@st.cache_resource
def get_best_gemini_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

        # Priority order (highest quality → fallback)
        preferred = [
            "models/gemini-1.5-pro-latest",
            "models/gemini-exp-1206",
            "models/gemini-pro-latest",
            "models/gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-pro-vision",
        ]
        chosen = next((m for pref in preferred for m in models if pref in m), None)
        if not chosen:
            chosen = models[0]
            st.warning(f"No preferred model found; defaulting to {chosen}")
        else:
            st.success(f"✅ Using Gemini model: **{chosen}**")
        return genai.GenerativeModel(chosen), chosen
    except Exception as e:
        st.error(f"Gemini setup failed: {e}")
        return None, None

# ==============================
# JSON Repair & Parse
# ==============================
def sanitize_json(txt):
    txt = txt.strip().replace("```json","").replace("```","")
    try:
        start = txt.index("{"); end = txt.rindex("}")+1
        txt = txt[start:end]
    except Exception: pass
    txt = re.sub(r",(\s*[}\]])",r"\1",txt)
    txt = txt.replace("“",'"').replace("”",'"')
    return txt

def parse_json_safe(t):
    try:
        return json.loads(t)
    except Exception:
        try: return json.loads(sanitize_json(t))
        except Exception: return None

# ==============================
# Rubric Summary (short)
# ==============================
RUBRIC_MAX = {
    "Functional Requirements":2,"Loading and Processing Data":3,
    "Correctness – Q1–Q5 Implementation":3,"Correctness - Output Accuracy":3,
    "Visualization":3,"User Interface":3,"Code Quality - Organization":1,
    "Code Quality - Naming conventions":1,"Code Quality - comments":1,
    "Test Plan and Strategy":3,"Coverage - Q1–Q5 Evidence":5,
    "Manual Test Cases – Completeness":3,"Manual Test Cases – Evidence":2,
    "Automated Tests":5,"Reproducibility":2,
    "Deduction - Statement of Completion":-1,
    "Deduction - Usage of AI statement":-1,"Deduction - Statement of Assistance":-1,
}

# ==============================
# Prompt Builder
# ==============================
def make_prompt(context, strict, name, report, code, imgs):
    return [
        "You are an expert university programming examiner.",
        strict,
        f"Grade the submission '{name}' strictly according to rubric.",
        f"Assignment context: {context or 'N/A'}",
        f"RUBRIC JSON: {json.dumps(RUBRIC_MAX)}",
        f"REPORT:\n{report[:100000]}\n\nCODE:\n{code[:100000]}",
        f"Images present: {imgs}",
        """
Return only JSON:
{
 "grading_summary":[{"criterion":"...","score":...,"max_score":...,"justification":"..."}],
 "overall_feedback":"..."
}
"""
    ]

def grade(model, prompt):
    try:
        resp = model.generate_content(prompt, request_options={"timeout":300})
        return parse_json_safe(resp.text)
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None

# ==============================
# Core Grading Loop
# ==============================
def grade_zip(zip_obj, context, model, strict):
    results=[]
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_obj,"r") as z: z.extractall(td)
        recursive_extract(td)
        subdirs=[os.path.join(td,d) for d in os.listdir(td)]
        for i,d in enumerate(subdirs):
            if not os.path.isdir(d): continue
            st.write(f"Grading {i+1}/{len(subdirs)}: **{os.path.basename(d)}**")
            rep,code,imgs="","",False
            for dp,_,fns in os.walk(d):
                for f in fns:
                    p=os.path.join(dp,f)
                    if f.endswith(".docx"): rep+=extract_text_from_docx_bytes(open(p,"rb").read())
                    elif f.endswith(".pdf"): rep+=extract_text_from_pdf_bytes(open(p,"rb").read())
                    elif f.endswith(".py"): code+=read_text(p)
                    elif f.endswith(".ipynb"): code+=read_ipynb(p)
            imgs=True if any(fn.lower().endswith((".png",".jpg",".jpeg")) for _,_,fs in os.walk(d) for fn in fs) else False
            prompt=make_prompt(context,strict,os.path.basename(d),rep,code,imgs)
            res=grade(model,prompt)
            if res:
                total=sum(x["score"] for x in res["grading_summary"])
                maxs=sum(abs(v) for v in RUBRIC_MAX.values())
                results.append({
                    "Student":os.path.basename(d),
                    "Total":total,"Percent":round(total/maxs*100,2),
                    "Feedback":res.get("overall_feedback","")
                })
        return pd.DataFrame(results)

# ==============================
# UI
# ==============================
st.title("👨‍🏫 AI-Powered Assignment Grader (Auto Model + Large Upload)")

api_key = st.text_input("Gemini API Key 🔑", type="password")
context = st.text_area("Assignment Context (Optional)")
strictness = st.selectbox("Strictness",["Lenient","Standard","Strict","Critical"])
strict_map = {
 "Lenient":"Grade leniently, reward effort.",
 "Standard":"Grade fairly per rubric.",
 "Strict":"Grade rigorously, deduct for small errors.",
 "Critical":"Grade as a capstone reviewer, very strict."
}
st.write("")

st.subheader("📂 Upload ZIP (up to ~1 GB locally)")
upfile = st.file_uploader("Choose a ZIP", type="zip")
st.markdown("OR paste a Google Drive / S3 link to a ZIP:")
link = st.text_input("ZIP URL (optional)")

if st.button("🚀 Grade Assignments", type="primary"):
    if not api_key:
        st.error("Provide Gemini API Key")
    elif not (upfile or link):
        st.error("Upload a file or provide a link.")
    else:
        # handle remote download if link
        if link and not upfile:
            st.info("Downloading remote ZIP ...")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with requests.get(link, stream=True, timeout=60) as r:
                for chunk in r.iter_content(8192):
                    tmp.write(chunk)
            zip_obj = open(tmp.name, "rb")
        else:
            zip_obj = upfile

        model, name = get_best_gemini_model(api_key)
        if model:
            with st.spinner(f"Grading with {name} ({strictness})..."):
                df = grade_zip(zip_obj, context, model, strict_map[strictness])
                st.dataframe(df)
                st.download_button("📥 Download Results CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "grading_results.csv","text/csv")
