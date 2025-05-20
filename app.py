import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import io
from tenacity import retry, stop_after_attempt, wait_exponential

# --- CONFIG ---
MODEL_NAME_GEMINI = "models/gemini-2.5-pro"
MODEL_NAME_GPT = "gpt-4.1"
BATCH_SIZE = 40
MAX_ATTEMPTS_API = 5
MAX_RETRY_ROUNDS = 2
owner_col = "OwnerName"

# --- API KEYS ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not GEMINI_API_KEY or not OPENAI_API_KEY:
    st.error("Missing GEMINI_API_KEY or OPENAI_API_KEY in environment!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY
generation_config = genai.types.GenerationConfig(
    temperature=0.0,
    max_output_tokens=2048
)
llm_gemini = genai.GenerativeModel(
    MODEL_NAME_GEMINI,
    generation_config=generation_config
)

# --- PROMPT TEMPLATES ---
def get_classification_prompt(names_batch):
    names_str = "\n".join(names_batch)
    prompt = (
        "You are an expert at US name classification and splitting.\n"
        "For each of the following names:\n"
        "- Classify as 'Person' or 'Business'.\n"
        "- If 'Person', provide First, Middle (if any), and Last names. ENSURE THE CORRECT ORDER.\n"
        "- Handle ALL possible US name order permutations (LAST FIRST MIDDLE, FIRST LAST, FIRST MIDDLE LAST, FIRST LAST SUFFIX, FIRST MIDDLE MIDDLE2 LAST, FIRST LAST SUFFIX, etc; can be >3 words and not always in the same order).\n"
        "- If not absolutely certain, classify as Business and put the full name in LastName.\n"
        "- If 'Business', put the entire name in the 'LastName' column and 'Business' in 'Type'. Leave 'FirstName' and 'MiddleName' empty.\n"
        "- For missing fields, ALWAYS use empty string \"\" (never nan/null/N/A/none).\n"
        "- KEEP THE 'Name' FIELD EXACTLY AS INPUT, INCLUDING ALL UPPERCASE and punctuation.\n"
        "- NEVER output markdown/code blocks or explanations—only plain CSV.\n"
        "- If the input has more than three words, treat as 'Business' unless it is clearly a person name.\n"
        "- If the input contains business indicators like INC, LLC, CORP, CO, COMPANY, BANK, ESTATE, TRUST, GROUP, AND, &, or numbers, classify as Business unless a clear person pattern is detected.\n"
        "- For one-word names, classify as Business unless it matches a common US person last name (e.g. SMITH, JOHNSON, WILLIAMS).\n"
        "Output a CSV with these headers: Name,Type,FirstName,MiddleName,LastName\n"
        "EXAMPLES:\n"
        "SMITH,Person,,,SMITH\n"
        "DOE JANE,Person,JANE,,DOE\n"
        "BROWN JOHN Q JR,Person,JOHN,Q JR,BROWN\n"
        "MC DONALD,Person,,,MC DONALD\n"
        "CAFÉ DEL MAR,Business,,,CAFÉ DEL MAR\n"
        "THE SMITH COMPANY,Business,,,THE SMITH COMPANY\n"
        "MAMA FUS NOODLE HOUSE,Business,,,MAMA FUS NOODLE HOUSE\n"
        "SMALLS WILLIE R,Person,WILLIE,R,SMALLS\n"
        "GARRETT ALDA ESTATE & HEIRS,Business,,,GARRETT ALDA ESTATE & HEIRS\n"
        "PIERCE & YOUNG ATTORNEYS AT LAW,Business,,,PIERCE & YOUNG ATTORNEYS AT LAW\n"
        "Reply as a CSV with exactly these headers as the first row: Name,Type,FirstName,MiddleName,LastName\n"
        "Do not output any other text or explanations.\n\n"
        f"{names_str}"
    )
    return prompt

def get_audit_prompt(audit_lines):
    names_str = "\n".join(audit_lines)
    prompt = (
        "You are an expert at US name classification and splitting.\n"
        "For each entry below (OriginalName,CurrentType,CurrentFirstName,CurrentMiddleName,CurrentLastName):\n"
        "- If 'CurrentType' is 'Person' but the split is not plausible (wrong order, business words in person fields, more than 4 words, or wrong permutation), re-split. Be cautious about ALL possible orders (LAST FIRST, FIRST LAST, FIRST MIDDLE LAST, FIRST LAST SUFFIX, etc). If not certain, classify as Business.\n"
        "- If 'CurrentType' is 'Business' but it looks like a person, change to Person and split.\n"
        "- If 'CurrentType' starts with 'Unknown', try to classify and split accordingly.\n"
        "- KEEP THE Name field EXACTLY as input. For missing fields, always use \"\" (never nan/null/N/A/none).\n"
        "- Output CSV ONLY with these headers: Name,Type,FirstName,MiddleName,LastName\n"
        "- NEVER output markdown/code blocks or explanations.\n\n"
        f"{names_str}"
    )
    return prompt

# --- MODEL CALLS ---
@retry(wait=wait_exponential(multiplier=2, min=3, max=30), stop=stop_after_attempt(MAX_ATTEMPTS_API))
def generate_with_gemini(prompt_text):
    response = llm_gemini.generate_content(prompt_text)
    text_response = ""
    if hasattr(response, "parts") and response.parts:
        text_response = "".join(part.text for part in response.parts if hasattr(part, "text"))
    elif hasattr(response, "text"):
        text_response = response.text
    else:
        try:
            text_response = response.candidates[0].content.parts[0].text
        except Exception as e:
            st.warning(f"Could not extract text from Gemini response. {e}")
            text_response = ""
    return text_response.strip()

@retry(wait=wait_exponential(multiplier=2, min=3, max=30), stop=stop_after_attempt(MAX_ATTEMPTS_API))
def generate_with_gpt(prompt_text):
    response = openai.chat.completions.create(
        model=MODEL_NAME_GPT,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        max_tokens=1800
    )
    return response.choices[0].message.content.strip()

def parse_csv_response(csv_text, expected_headers):
    if not csv_text:
        return pd.DataFrame(columns=expected_headers)
    try:
        lines = csv_text.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean_csv_text = "\n".join(lines)
        data = io.StringIO(clean_csv_text)
        df = pd.read_csv(data, dtype=str, on_bad_lines='warn')
        df.columns = [str(col).strip().replace('"', '').replace("'", "") for col in df.columns]
        for col in expected_headers:
            df[col] = df.get(col, "").fillna("").replace(["nan", "NaN", "N/A", "na", "null", "None"], "")
        if list(df.columns) == expected_headers:
            pass
        elif all(eh in df.columns for eh in expected_headers):
            df = df[expected_headers]
        elif len(df.columns) == len(expected_headers):
            df.columns = expected_headers
        else:
            st.warning(f"CSV headers mismatch: {df.columns.tolist()}")
            return pd.DataFrame(columns=expected_headers)
        return df
    except Exception as e:
        st.error(f"Error parsing CSV: {e}\n{csv_text[:500]}")
        return pd.DataFrame(columns=expected_headers)

# --- MAIN LOGIC FOR STREAMLIT ---
def classify_names(names):
    expected_headers = ["Name", "Type", "FirstName", "MiddleName", "LastName"]
    # 1. Gemini classification (batched)
    all_results = []
    for i in range(0, len(names), BATCH_SIZE):
        batch = names[i:i + BATCH_SIZE]
        prompt = get_classification_prompt(batch)
        csv_resp = generate_with_gemini(prompt)
        df = parse_csv_response(csv_resp, expected_headers)
        if df.empty:
            df = pd.DataFrame([{h: "" for h in expected_headers} for _ in batch])
            df["Name"] = batch
        all_results.append(df)
    out_df = pd.concat(all_results, ignore_index=True)
    out_df["SplitSource"] = "Gemini"
    # 2. Find uncertain rows for GPT audit: (Unknown, or Person with missing First/Last)
    audit_rows = []
    idxs = []
    for idx, row in out_df.iterrows():
        if (
            str(row["Type"]).startswith("Unknown")
            or (row["Type"] == "Person" and (row["FirstName"] == "" or row["LastName"] == ""))
            or any(str(row.get(c, "")).lower() in ["nan", "n/a", "null", "none"] for c in ["FirstName", "MiddleName", "LastName"])
        ):
            fields = [
                row["Name"],
                row["Type"],
                row["FirstName"],
                row["MiddleName"],
                row["LastName"],
            ]
            audit_rows.append(",".join(str(x) for x in fields))
            idxs.append(idx)
    if audit_rows:
        # Only audit the actual problematic ones with GPT-4o
        for i in range(0, len(audit_rows), BATCH_SIZE):
            batch = audit_rows[i:i+BATCH_SIZE]
            prompt = get_audit_prompt(batch)
            gpt_csv = generate_with_gpt(prompt)
            df_gpt = parse_csv_response(gpt_csv, expected_headers)
            for j, gpt_row in df_gpt.iterrows():
                real_idx = idxs[i+j] if i+j < len(idxs) else None
                if real_idx is not None:
                    for col in ["Type", "FirstName", "MiddleName", "LastName"]:
                        out_df.at[real_idx, col] = gpt_row[col]
                    out_df.at[real_idx, "SplitSource"] = "GPT-4o"
    return out_df

# --- STREAMLIT UI ---
st.set_page_config(page_title="US Name Splitter & Classifier", layout="wide")
st.title("US Name Classifier & Splitter (Gemini + GPT-4o)")
st.write("Input a list of names below (or upload a CSV/Excel). You'll get each name's type and best split using AI.")
uploaded_file = st.file_uploader("Upload CSV or Excel file with names (column: OwnerName or names)", type=["csv", "xlsx"])
sample = st.text_area("Paste names (one per line)", height=120)
names = []

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)
        name_col = None
        for c in df.columns:
            if c.lower() in ["ownername", "name", "names"]:
                name_col = c
                break
        if not name_col:
            st.warning("No suitable column (OwnerName/name/names) found in uploaded file.")
        else:
            names = df[name_col].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        st.error(f"Could not read file: {e}")
elif sample.strip():
    names = [n.strip() for n in sample.splitlines() if n.strip()]

names = [n for n in names if n]
if names:
    st.write(f"Found {len(names)} names.")
    with st.spinner("Processing..."):
        df_results = classify_names(names)
    st.success("Done!")
    st.dataframe(df_results, use_container_width=True)
    st.download_button(
        "Download Results as CSV",
        df_results.to_csv(index=False).encode(),
        "name_splits_results.csv",
        "text/csv"
    )
else:
    st.info("Input or upload a list of names to begin.")

# Address lookup (optional, if you want)
if st.checkbox("Address lookup from master file (upload your parsed data CSV)", value=False):
    master = st.file_uploader("Upload main parsed data CSV with addresses", type=["csv"])
    if master:
        mdf = pd.read_csv(master, dtype=str)
        address_col = None
        for c in mdf.columns:
            if c.lower() in ["address", "addr", "location"]:
                address_col = c
                break
        if address_col:
            # Do a left join
            merged = pd.merge(df_results, mdf[[owner_col, address_col]], how="left", left_on="Name", right_on=owner_col)
            st.write("Results with address:")
            st.dataframe(merged, use_container_width=True)
            st.download_button(
                "Download With Addresses",
                merged.to_csv(index=False).encode(),
                "name_splits_with_addresses.csv",
                "text/csv"
            )
        else:
            st.warning("No address column found in master file.")
