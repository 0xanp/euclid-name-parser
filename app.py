import streamlit as st
import pandas as pd
import os
import openai
import google.generativeai as genai
import io
from tenacity import retry, stop_after_attempt, wait_exponential

# --- CONFIGURATION ---
PARSED_DATA = "parsed_names_gemini_gpt_final.csv"
MODEL_GEMINI = "models/gemini-2.0-flash-lite"
MODEL_GPT = "gpt-4.1-nano"
BATCH_SIZE = 25
MAX_ATTEMPTS_API = 5
MAX_RETRY_ROUNDS = 2
OWNER_COL = "OwnerName"

# --- API KEYS (Streamlit Cloud: Set in Secrets) ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini + OpenAI model instances ---
generation_config = genai.types.GenerationConfig(
    temperature=0.0,
    max_output_tokens=2048
)
gemini_llm = genai.GenerativeModel(
    MODEL_GEMINI,
    generation_config=generation_config
)

def load_parsed_data():
    return pd.read_csv(PARSED_DATA, dtype=str).fillna("")

def get_classification_prompt(names_batch):
    names_str = "\n".join(names_batch)
    prompt = (
        "You are an expert at name classification and splitting.\n"
        "For each of the following names, classify as 'Person' or 'Business'.\n"
        "If 'Person', split into First, Middle (if any), and Last name (all caps, order as in input; beware: name order may be Last First Middle, First Last, or First Middle Last; DO NOT ASSUME order!).\n"
        "If unsure, classify as Business (put full name in LastName, other fields empty).\n"
        "NEVER output markdown/code blocks or explanations—PLAIN CSV ONLY: Name,Type,FirstName,MiddleName,LastName.\n"
        "If field is missing, use \"\".\n"
        "Examples:\n"
        "SMITH,Person,,,SMITH\n"
        "DOE JANE,Person,JANE,,DOE\n"
        "BROWN JOHN Q JR,Person,JOHN,Q JR,BROWN\n"
        "CAFÉ DEL MAR,Business,,,CAFÉ DEL MAR\n"
        "THE SMITH COMPANY,Business,,,THE SMITH COMPANY\n"
        f"{names_str}"
    )
    return prompt

def get_gpt_audit_prompt(entries):
    batch_str = "\n".join(entries)
    prompt = (
        "You are an expert at US name splitting and business detection.\n"
        "For each CSV entry: Name,Type,FirstName,MiddleName,LastName\n"
        "Audit split, keeping fields as is if correct; if incorrect or ambiguous, re-split cautiously (permutations possible: Last First, First Last, First Middle Last, First Last Middle).\n"
        "If not a person, Type should be Business and all names except LastName must be blank. If ambiguous or more than 4 name words, classify as Business.\n"
        "Plain CSV ONLY: Name,Type,FirstName,MiddleName,LastName. No explanations.\n"
        f"{batch_str}"
    )
    return prompt

@retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(MAX_ATTEMPTS_API))
def gemini_generate(prompt_text):
    response = gemini_llm.generate_content(prompt_text)
    text_response = ""
    if hasattr(response, "parts") and response.parts:
        text_response = "".join(part.text for part in response.parts if hasattr(part, "text"))
    elif hasattr(response, "text"):
        text_response = response.text
    else:
        try:
            text_response = response.candidates[0].content.parts[0].text
        except Exception:
            text_response = ""
    return text_response.strip()

@retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(MAX_ATTEMPTS_API))
def gpt_generate(prompt_text):
    response = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

def parse_llm_csv(csv_text, expected_headers):
    # Remove markdown code blocks if present
    lines = csv_text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    clean_csv_text = "\n".join(lines)
    data = io.StringIO(clean_csv_text)
    df = pd.read_csv(data, dtype=str, on_bad_lines='warn')
    df.columns = [str(col).strip().replace('"', '').replace("'", "") for col in df.columns]
    # Standardize columns
    for col in expected_headers:
        if col not in df.columns:
            df[col] = ""
    df = df[expected_headers]
    df = df.fillna("")
    for col in expected_headers:
        df[col] = df[col].replace(["nan", "NaN", "N/A", "na", "null", "None"], "")
    return df

def classify_with_gemini(name_list):
    # Batching
    expected_headers = ["Name", "Type", "FirstName", "MiddleName", "LastName"]
    results = []
    for i in range(0, len(name_list), BATCH_SIZE):
        batch = name_list[i:i+BATCH_SIZE]
        prompt = get_classification_prompt(batch)
        csv_text = gemini_generate(prompt)
        df = parse_llm_csv(csv_text, expected_headers)
        results.append(df)
    out_df = pd.concat(results, ignore_index=True).fillna("")
    return out_df

def audit_with_gpt(df_persons):
    # Audit only Type=Person, or those with empty/missing split
    expected_headers = ["Name", "Type", "FirstName", "MiddleName", "LastName"]
    mask_audit = (df_persons["Type"] == "Person") & (
        (df_persons["FirstName"] == "") | (df_persons["LastName"] == "") |
        (df_persons["Name"].str.split().str.len() > 4)
    )
    if not mask_audit.any():
        return df_persons
    audit_df = df_persons[mask_audit].copy()
    entries = []
    for _, row in audit_df.iterrows():
        fields = [row["Name"], row["Type"], row["FirstName"], row["MiddleName"], row["LastName"]]
        entries.append(",".join(fields))
    results = []
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i+BATCH_SIZE]
        prompt = get_gpt_audit_prompt(batch)
        csv_text = gpt_generate(prompt)
        df_batch = parse_llm_csv(csv_text, expected_headers)
        results.append(df_batch)
    audited_df = pd.concat(results, ignore_index=True).fillna("")
    # Update main df
    df_persons_update = df_persons.set_index("Name")
    for _, row in audited_df.iterrows():
        df_persons_update.loc[row["Name"], ["Type", "FirstName", "MiddleName", "LastName"]] = \
            row[["Type", "FirstName", "MiddleName", "LastName"]].values
    df_persons_update = df_persons_update.reset_index()
    return df_persons_update

def lookup_address(df_classified, df_parsed):
    # Merge on Name to get address fields
    address_fields = [c for c in df_parsed.columns if "address" in c.lower()]
    df_merged = pd.merge(df_classified, df_parsed[[OWNER_COL]+address_fields], left_on="Name", right_on=OWNER_COL, how="left")
    return df_merged

# --- STREAMLIT APP ---
st.set_page_config(page_title="Name Classification & Address Lookup", layout="wide")
st.title("Name Classification, Person Split & Address Lookup")
st.write("Input a list of names (paste or upload), run multi-layer LLM classification, and get address lookup.")

# User input
with st.form("name_input"):
    input_method = st.radio("Input names by:", ["Paste/Type", "Upload .csv or .txt"])
    name_list = []
    if input_method == "Paste/Type":
        user_input = st.text_area("Enter names (one per line):", height=200)
        if user_input:
            name_list = [line.strip().upper() for line in user_input.splitlines() if line.strip()]
    else:
        uploaded_file = st.file_uploader("Upload a .csv or .txt file (one name per row):")
        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "csv":
                df_up = pd.read_csv(uploaded_file, header=None)
                name_list = df_up[0].astype(str).str.upper().tolist()
            else:
                name_list = [line.decode("utf-8").strip().upper() for line in uploaded_file.readlines() if line.strip()]
    submit = st.form_submit_button("Run Classification")

if submit and name_list:
    st.info(f"Processing {len(name_list)} names via Gemini…")
    # 1. Classify/split with Gemini
    df_classified = classify_with_gemini(name_list)
    # 2. Audit with GPT-4.1-nano for Person splits with issues or >4 words
    df_classified = audit_with_gpt(df_classified)
    # 3. Address lookup
    df_parsed = load_parsed_data()
    df_out = lookup_address(df_classified, df_parsed)
    # Reorder for best visibility
    out_cols = ["Name", "Type", "FirstName", "MiddleName", "LastName"] + [c for c in df_out.columns if c not in ["Name", "Type", "FirstName", "MiddleName", "LastName", OWNER_COL]]
    st.dataframe(df_out[out_cols].fillna("").replace("nan", ""))
    # Download button
    csv = df_out[out_cols].to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "name_classification_results.csv", "text/csv")
else:
    st.info("Awaiting input…")
