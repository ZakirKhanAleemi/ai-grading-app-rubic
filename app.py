import streamlit as st
import pandas as pd
import google.generativeai as genai
import docx, zipfile, io, json, os, re, tempfile, time, requests
from PIL import Image, UnidentifiedImageError
import fitz, py7zr, rarfile
from google.api_core import exceptions

# ==============================
# Page Setup
# ==============================
st.set_page_config(layout="wide", page_title="AI Grading Assistant")
st.markdown("<h3>👨‍🏫 AI-Powered Assignment Grader (Full Rubric + Auto Gemini Model)</h3>", unsafe_allow_html=True)

# ==============================
# File Handling Helpers
# ==============================
def extract_text_from_docx_bytes(b):
    try:
        d = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        return f"[DOCX error] {e}"

def extract_text_from_pdf_bytes(b):
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        return f"[PDF error] {e}"

def read_ipynb(p):
    try:
        nb = json.load(open(p, encoding="utf-8", errors="ignore"))
        return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    except Exception as e:
        return f"[IPYNB error] {e}"

def read_text(p):
    try: return open(p, encoding="utf-8", errors="ignore").read()
    except Exception as e: return f"[READ error] {e}"

# ==============================
# Archive Extraction
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
# Gemini Model Auto-Selection
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
# JSON Repair
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
# Rubric Defaults
# ==============================
DEFAULT_RUBRIC = """Grade
Functional Requirements
No genuine attempt to define functional requirements
0 points
Fewer than 8 FRs, vague or not clearly mapped to Q1–Q5.
0.5 points
8–15 FRs, some mapping to Q1–Q5
1 points
8–15 clear, testable FRs, well-mapped to Q1–Q5.
2 points

Loading and Processing Data
No genuine attempt to load or process data.
0 points
Loads data but fails to handle invalid data, cancelled buses, or date/time formats correctly.
1 points
Fully correct parsing, including “none”, date/time, and cancelled buses.
3 points

Correctness – Q1–Q5 Implementation
No genuine attempt to implement requirements.
0 points
1–2 questions implemented with major issues.
1 points
3–4 questions implemented correctly
2 points
All 5 questions implemented correctly and appropriately.
3 points

Visualization
No visualization provided.
0 points
Visualization present but unclear, incorrect, or unhelpful.
1 points
Clear visualization, but lacks labels or readability
2 points
Clear, labelled, readable charts that help answer at least one question.
3 points

... (etc. include all criteria you listed) ...
"""

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
RUBRIC_ORDER = list(RUBRIC_MAX.keys())

# ==============================
# Prompt builder and grader
# ==============================
def make_prompt(context, strict, name, rubric, report, code, imgs):
    return [
        "You are an expert programming examiner.",
        strict,
        f"Grade '{name}' according to this rubric text:\n{rubric}",
        f"Rubric JSON (max scores): {json.dumps(RUBRIC_MAX)}",
        f"Assignment context:\n{context or 'N/A'}",
        f"REPORT:\n{report[:80000]}\n\nCODE:\n{code[:80000]}",
        f"Images present: {imgs}",
        "Return only JSON:\n"
        "{'grading_summary':[{'criterion':'...','score':...,'max_score':...,'justification':'...'}],'overall_feedback':'...'}"
    ]

def grade(model,prompt):
    try:
        r=model.generate_content(prompt,request_options={"timeout":300})
        return parse_json_safe(r.text)
    except Exception as e:
        st.error(f"Gemini error: {e}");return None

# ==============================
# Grading loop
# ==============================
def grade_zip(zip_obj, context, rubric, model, strict):
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
            res=grade(model,make_prompt(context,strict,os.path.basename(d),rubric,rep,code,imgs))
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
# Sidebar Configuration
# ==============================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    strictness = st.selectbox("Strictness",["Lenient","Standard","Strict","Critical"])
    strict_map={
        "Lenient":"Grade leniently, reward effort.",
        "Standard":"Grade fairly and accurately per rubric.",
        "Strict":"Grade rigorously; deduct for minor issues.",
        "Critical":"Grade as a capstone examiner, very strict."
    }
    st.header("📋 Editable Rubric")
    rubric_text = st.text_area("Paste or edit the rubric here:", value=DEFAULT_RUBRIC, height=400)
    context = st.text_area("Assignment Context (optional)", height=120)

# ==============================
# Upload section
# ==============================
st.subheader("📂 Upload ZIP (up to 1 GB locally or use link on Streamlit Cloud)")
upfile=st.file_uploader("Choose ZIP file",type="zip")
link=st.text_input("Or paste Google Drive / S3 link")

# ==============================
# Main Actions
# ==============================
if st.button("🚀 Grade Assignments"):
    if not api_key:
        st.error("Please enter Gemini API Key.")
    elif not (upfile or link):
        st.error("Upload a ZIP or provide a link.")
    else:
        if link and not upfile:
            st.info("Downloading ZIP from link ...")
            tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".zip")
            with requests.get(link,stream=True,timeout=60) as r:
                for chunk in r.iter_content(8192): tmp.write(chunk)
            zip_obj=open(tmp.name,"rb")
        else: zip_obj=upfile
        model,name=get_best_gemini_model(api_key)
        if model:
            with st.spinner(f"Grading with {name} ({strictness})..."):
                df=grade_zip(zip_obj,context,rubric_text,model,strict_map[strictness])
                st.dataframe(df)
                st.download_button("📥 Download Results CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "grading_results.csv","text/csv")

# ==============================
# Rubric summary display
# ==============================
st.subheader("📊 Rubric Summary")
st.dataframe(pd.DataFrame(
    [{"Criterion":k,"Max Score":v} for k,v in RUBRIC_MAX.items()]
), use_container_width=True)
