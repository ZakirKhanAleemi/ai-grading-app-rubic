import streamlit as st
import pandas as pd
import google.generativeai as genai
import docx, zipfile, io, json, os, re, tempfile, time, requests
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from google.api_core import exceptions
import fitz, py7zr, rarfile

# ==============================
# Streamlit setup
# ==============================
st.set_page_config(layout="wide", page_title="AI Grading Assistant")
st.markdown(
    """
    <style>
    [data-testid="stFileUploader"] button {height:3em;}
    </style>
    """,
    unsafe_allow_html=True
)
st.session_state.setdefault("upload_limit", 1024)   # MB, effective only when self-hosted
st.write(f"🔼 Local upload limit (if self-hosted): **{st.session_state.upload_limit} MB**")

# ==============================
# File utilities
# ==============================
def extract_text_from_docx_bytes(b):
    try:
        d = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e: return f"[DOCX error] {e}"

def extract_text_from_pdf_bytes(b):
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e: return f"[PDF error] {e}"

def read_ipynb(p):
    try:
        nb = json.load(open(p, encoding="utf-8", errors="ignore"))
        return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    except Exception as e: return f"[IPYNB error] {e}"

def read_text(p):
    try: return open(p, encoding="utf-8", errors="ignore").read()
    except Exception as e: return f"[READ error] {e}"

# ==============================
# Archive extraction
# ==============================
def extract_any_archive(p, dest):
    try:
        pl=p.lower()
        if pl.endswith(".zip"):
            with zipfile.ZipFile(p) as z:z.extractall(dest)
        elif pl.endswith(".7z"):
            with py7zr.SevenZipFile(p) as z:z.extractall(dest)
        elif pl.endswith(".rar"):
            with rarfile.RarFile(p) as r:r.extractall(dest)
        else:return False
        os.remove(p);return True
    except Exception:return False

def recursive_extract(root):
    while True:
        found=[os.path.join(dp,f) for dp,_,fs in os.walk(root) for f in fs if f.lower().endswith((".zip",".7z",".rar"))]
        if not found:break
        for f in found:extract_any_archive(f,os.path.dirname(f))

# ==============================
# Gemini model auto-selection
# ==============================
@st.cache_resource
def get_best_gemini_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models=[m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        preferred=[
            "models/gemini-1.5-pro-latest","models/gemini-exp","models/gemini-pro-latest",
            "models/gemini-1.5-flash-latest","gemini-1.5-pro","gemini-pro","gemini-pro-vision"
        ]
        chosen=next((m for pref in preferred for m in models if pref in m),models[0])
        st.success(f"✅ Using Gemini model: **{chosen}**")
        return genai.GenerativeModel(chosen), chosen
    except Exception as e:
        st.error(f"Gemini setup failed: {e}")
        return None,None

# ==============================
# JSON helpers
# ==============================
def sanitize_json(txt):
    txt=txt.replace("```json","").replace("```","").strip()
    try:
        s=txt.index("{");e=txt.rindex("}")+1;txt=txt[s:e]
    except:pass
    txt=re.sub(r",(\s*[}\]])",r"\1",txt)
    txt=txt.replace("“",'"').replace("”",'"')
    return txt

def parse_json_safe(t):
    try:return json.loads(t)
    except: 
        try:return json.loads(sanitize_json(t))
        except:return None

# ==============================
# Rubric
# ==============================
RUBRIC_MAX={
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
RUBRIC_ORDER=list(RUBRIC_MAX.keys())

# ==============================
# Prompt builder
# ==============================
def make_prompt(context,strict,name,report,code,imgs):
    return [
        "You are an expert programming examiner.",
        strict,
        f"Grade '{name}' using this rubric JSON:\n{json.dumps(RUBRIC_MAX,indent=2)}",
        f"Assignment context:\n{context or 'N/A'}",
        f"REPORT:\n{report[:100000]}\n\nCODE:\n{code[:100000]}",
        f"Images present: {imgs}",
        "Return only JSON:\n"
        "{'grading_summary':[{'criterion':'...','score':...,'max_score':...,'justification':'...'}],"
        "'overall_feedback':'...'}"
    ]

def grade(model,prompt):
    try:
        r=model.generate_content(prompt,request_options={"timeout":300})
        return parse_json_safe(r.text)
    except Exception as e:
        st.error(f"Gemini error: {e}");return None

# ==============================
# Grading core
# ==============================
def grade_zip(zip_obj,context,model,strict):
    out=[]
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_obj) as z:z.extractall(td)
        recursive_extract(td)
        subs=[os.path.join(td,d) for d in os.listdir(td) if os.path.isdir(os.path.join(td,d))]
        for i,d in enumerate(subs,1):
            st.write(f"Grading {i}/{len(subs)}: **{os.path.basename(d)}**")
            rep,code,imgs="","",False
            for dp,_,fs in os.walk(d):
                for f in fs:
                    p=os.path.join(dp,f)
                    if f.endswith(".docx"):rep+=extract_text_from_docx_bytes(open(p,"rb").read())
                    elif f.endswith(".pdf"):rep+=extract_text_from_pdf_bytes(open(p,"rb").read())
                    elif f.endswith(".py"):code+=read_text(p)
                    elif f.endswith(".ipynb"):code+=read_ipynb(p)
            imgs=any(fn.lower().endswith((".png",".jpg",".jpeg")) for _,_,fs in os.walk(d) for fn in fs)
            res=grade(model,make_prompt(context,strict,os.path.basename(d),rep,code,imgs))
            if res:
                total=sum(x["score"] for x in res["grading_summary"])
                maxt=sum(abs(v) for v in RUBRIC_MAX.values())
                out.append({
                    "Student":os.path.basename(d),
                    "Total":total,"Percent":round(total/maxt*100,2),
                    "Feedback":res.get("overall_feedback","")
                })
        return pd.DataFrame(out)

# ==============================
# Streamlit UI
# ==============================
st.title("👨‍🏫 AI-Powered Assignment Grader – Full Rubric + Auto Gemini Model")

api_key=st.text_input("Gemini API Key",type="password")
context=st.text_area("Assignment Context (Optional)")
strictness=st.selectbox("Strictness",["Lenient","Standard","Strict","Critical"])
strict_map={
 "Lenient":"Grade leniently, reward effort.",
 "Standard":"Grade fairly and accurately per rubric.",
 "Strict":"Grade rigorously; deduct for minor issues.",
 "Critical":"Grade as a capstone examiner, very strict."
}

st.subheader("📂 Upload ZIP (up to 1 GB locally)")
upfile=st.file_uploader("Choose ZIP",type="zip")
link=st.text_input("or paste Google Drive / S3 ZIP link")

if st.button("🚀 Grade Assignments"):
    if not api_key:
        st.error("Enter Gemini API Key.")
    elif not (upfile or link):
        st.error("Upload a ZIP or provide a link.")
    else:
        if link and not upfile:
            st.info("Downloading remote ZIP …")
            tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".zip")
            with requests.get(link,stream=True,timeout=60) as r:
                for chunk in r.iter_content(8192):tmp.write(chunk)
            zip_obj=open(tmp.name,"rb")
        else: zip_obj=upfile
        model,name=get_best_gemini_model(api_key)
        if model:
            with st.spinner(f"Grading with {name} ({strictness}) …"):
                df=grade_zip(zip_obj,context,model,strict_map[strictness])
                st.dataframe(df)
                st.download_button("📥 Download Results CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "grading_results.csv","text/csv")
