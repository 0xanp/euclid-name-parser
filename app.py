import os
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
import openai
import io
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# --- LOAD SECRETS / ENV ---
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY or not OPENAI_API_KEY:
    st.error("Missing Gemini or OpenAI API key. Please set in .env or Streamlit secrets.")
    st.stop()

openai.api_key = OPENAI_API_KEY

MODEL_NAME_GEMINI = "models/gemini-2.5-pro-preview-05-06"
MODEL_NAME_GPT = "gpt-4.1"
BATCH_SIZE = 50  # lower for interactive app

@st.cache_resource
def load_existing_parsed():
    try:
        return pd.read_csv("parsed_names_gemini_gpt_final.csv", dtype=str, low_memory=False).fillna("")
    except Exception:
        return pd.DataFrame()

@retry(wait=wait_exponential(multiplier=2, min=5, max=30), stop=stop_after_attempt(3))
def generate_with_gemini(prompt_text):
    import os
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-preview-04-17"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)],
        ),
    ]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    text = getattr(response, "text", None)
    if not text and hasattr(response, "candidates"):
        text = response.candidates[0].text
    return text or ""

@retry(wait=wait_exponential(multiplier=2, min=5, max=30), stop=stop_after_attempt(3))
def generate_with_gpt(prompt_text):
    resp = openai.chat.completions.create(
        model=MODEL_NAME_GPT,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        max_tokens=1800
    )
    return resp.choices[0].message.content.strip()

def get_classification_prompt(names_batch):
    names_str = "\n".join(names_batch)
    prompt = (
        "You are an expert at US name classification and splitting.\n"
        "For each of the following names:\n"
        "- Classify as 'Person' or 'Business'.\n"
        "- If 'Person', provide First, Middle (if any), and Last names. ENSURE THE CORRECT ORDER.\n"
        "- Handle ALL possible orders and all possible permutations and numbers of name parts:\n"
        "    - LAST FIRST MIDDLE, FIRST LAST, FIRST MIDDLE LAST, FIRST LAST SUFFIX, FIRST MIDDLE MIDDLE2 LAST, FIRST LAST SUFFIX, etc.\n"
        "    - The input may have 2, 3, 4 or even more words; US person names are not always in the same order or word count.\n"
        "- If not absolutely certain, classify as 'Business' and put the full name in LastName.\n"
        "- If 'Business', put the entire name in the 'LastName' column and 'Business' in 'Type'. Leave 'FirstName' and 'MiddleName' empty.\n"
        "- If a field is missing, ALWAYS use empty string \"\" (never nan/null/N/A/none).\n"
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
        "- If 'CurrentType' is 'Person', audit the split even if it looks plausible. Check for wrong order, business words in person fields, more than 4 words, or wrong permutation, and re-split if needed. Be cautious about ALL possible orders: LAST FIRST, FIRST LAST, FIRST MIDDLE LAST, FIRST LAST SUFFIX, FIRST MIDDLE MIDDLE2 LAST, FIRST LAST SUFFIX, etc. If not certain, classify as Business.\n"
        "- If 'CurrentType' is 'Business' but it looks like a person, change to Person and split.\n"
        "- If 'CurrentType' starts with 'Unknown', try to classify and split accordingly.\n"
        "- KEEP THE Name field EXACTLY as input. For missing fields, always use \"\" (never nan/null/N/A/none).\n"
        "- Output CSV ONLY with these headers: Name,Type,FirstName,MiddleName,LastName\n"
        "- NEVER output markdown/code blocks or explanations.\n\n"
        f"{names_str}"
    )
    return prompt

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
            st.warning(f"Critical Warning: CSV headers mismatch. Showing partial results.")
            return pd.DataFrame(columns=expected_headers)
        return df
    except Exception as e:
        st.warning(f"Error parsing CSV response: {e}")
        return pd.DataFrame(columns=expected_headers)

def classify_and_audit(names_batch):
    expected_headers = ["Name", "Type", "FirstName", "MiddleName", "LastName"]
    # 1. Gemini LLM classification
    prompt = get_classification_prompt(names_batch)
    csv_response = generate_with_gemini(prompt)
    df_llm = parse_csv_response(csv_response, expected_headers)

    # 2. AUDIT only uncertain/problematic rows with GPT-4o (not all names)
    audit_lines = [
        ",".join([
            row["Name"], row["Type"], row["FirstName"] or "", row["MiddleName"] or "", row["LastName"] or ""
        ])
        for _, row in df_llm.iterrows()
        if (
            str(row["Type"]).startswith("Unknown")
            or (row["Type"] == "Person" and (row["FirstName"] == "" or row["LastName"] == ""))
            or any(str(row.get(c, "")).lower() in ["nan", "n/a", "null", "none"] for c in ["FirstName", "MiddleName", "LastName"])
        )
    ]
    if audit_lines:
        audit_prompt = get_audit_prompt(audit_lines)
        audit_response = generate_with_gpt(audit_prompt)
        df_audit = parse_csv_response(audit_response, expected_headers)
        # Overwrite Gemini splits with audit splits for only those names
        for _, row in df_audit.iterrows():
            name = row.get("Name", "")
            if name in df_llm["Name"].values:
                df_llm.loc[df_llm["Name"] == name, ["Type", "FirstName", "MiddleName", "LastName"]] = [
                    row.get("Type", ""),
                    row.get("FirstName", ""),
                    row.get("MiddleName", ""),
                    row.get("LastName", ""),
                ]
    return df_llm

# --- STREAMLIT UI ---
st.title("Name Split & Type Classifier (Dual LLM Audit)")
existing_data = load_existing_parsed()
address_lookup_available = not existing_data.empty

st.write("Paste or upload a list of names. Each name is classified (Person/Business) and split (if Person) using Gemini 2.5 Pro, with uncertain cases re-audited by GPT-4.1. Address info is included if found in pre-parsed data.")

input_method = st.radio("Input method", ["Paste names", "Upload CSV"])

if input_method == "Paste names":
    name_input = st.text_area("Enter names (one per line):")
    if st.button("Classify and Split"):
        names = [n.strip() for n in name_input.strip().split("\n") if n.strip()]
        if names:
            with st.spinner("Classifying names..."):
                results_df = classify_and_audit(names)
                if address_lookup_available:
                    results_df = pd.merge(results_df, existing_data, left_on="Name", right_on="OwnerName", how="left")
                st.dataframe(results_df)
                csv_out = results_df.to_csv(index=False)
                st.download_button("Download CSV results", csv_out, "classified_names.csv")
        else:
            st.warning("Please enter at least one name.")
elif input_method == "Upload CSV":
    file = st.file_uploader("Upload CSV file with a column of names", type="csv")
    if file and st.button("Classify and Split"):
        df = pd.read_csv(file, dtype=str)
        name_col = st.selectbox("Select name column", list(df.columns))
        names = df[name_col].dropna().astype(str).tolist()
        with st.spinner("Classifying names..."):
            results_df = classify_and_audit(names)
            if address_lookup_available:
                results_df = pd.merge(results_df, existing_data, left_on="Name", right_on="OwnerName", how="left")
            st.dataframe(results_df)
            csv_out = results_df.to_csv(index=False)
            st.download_button("Download CSV results", csv_out, "classified_names.csv")
