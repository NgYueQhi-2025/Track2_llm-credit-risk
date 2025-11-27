# LLM-Based Credit Risk Assessment (Track 2)
### TEAM: THE ROOKIES
## 🔗 Quick Links
| Live Demo | Video Walkthrough | Pitch Deck |
| :---: | :---: | :---: |
| [**Launch Web App**](https://track2llm-credit-risk-9qhysbrs3gyvv34fxxvb2p.streamlit.app/) | [**Watch Demo**](https://drive.google.com/file/d/1wo5ZICngzGATV0njh2VrcXVVRxXCg6tV/view?usp=drivesdk) | [**View Slides**](https://www.canva.com/design/DAG54l7iTaU/WikI93oh_rpgqiTGHTlxPA/edit?utm_content=DAG54l7iTaU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) |


---

## ⭐ Overview

This repository contains a complete end-to-end prototype for **LLM-based credit risk assessment**, combining structured applicant data with unstructured text extracted from PDFs, images and CSVs. The system integrates OCR, regex-based parsing, LLM-powered behavioural analysis, rule-based overrides and transparent explainability through a Streamlit dashboard.

The app demonstrates how modern credit scoring can be improved using **Large Language Models (LLMs)** to capture behavioural risk signals beyond traditional numeric fields.

---

# 1. Industry Context

Traditional credit scoring models are often "blind" to context. They rely heavily on:
**Strict Numerics:** Income, FICO score, and liability counts.
**Missing Context:** They ignore explanations for late payments or behavioral cues in application letters.
**Black Box Logic:** Decisions are often hard to explain to the applicant 

But in real-world lending, **unstructured text** such as loan applications, explanations for late payments, or emails contains deeper behavioural cues:

- Tone and sentiment  
- Financial stress indicators  
- Contradictions  
- Risky behaviour clues  
- Repayment intention  

LLMs make it possible to extract those insights and fuse them with structured data to produce **more accurate, fair, and explainable** risk assessments.

---

# 2. Problem Statement

**How can we design a reliable and transparent LLM-based method to evaluate credit risk using unstructured text alongside structured financial fields?**

This prototype demonstrates a working solution.

---

# 3. Project Challenge

Build a prototype that:

- Uses an LLM to analyze unstructured text  
- Extracts behavioural and contextual risk features  
- Fuses text-based and numeric features  
- Produces interpretable credit risk scores  
- Visualizes results in an intuitive dashboard  

---

# 4. AI Opportunity

### ✔ LLMs can interpret unstructured text  
They detect sentiment, contradictions, behavioural patterns, and risky phrases.

### ✔ Multi-modal scoring  
The system merges:

- Structured fields (income, age, loan amount)  
- Extracted text fields (name, income from PDF, etc.)  
- LLM behavioural signals  

This mirrors modern underwriting practice where both **numbers and narrative** matter.

---

# 5. MVP Features

## 5.1 ⭐ File Upload & Text Extraction

Supported Formats:
- CSV  
- PDF  
- PNG / JPG  

Extraction Methods:
- `pdfplumber` for PDFs  
- `pytesseract` for OCR on images  
- Fallback text decoding  

(Handled inside `extract_text_from_file()`)

---

## 5.2 ⭐ Automatic Field Parsing (from documents)

Regex heuristics extract:

- Applicant name  
- Age  
- Annual income  
- Requested loan amount  

(via `parse_fields_from_text()`)

This allows applicants to upload raw files and still receive structured scoring.

---

## 5.3 ⭐ LLM-Based Behavioural Feature Extraction

Using:

```
integrations.run_feature_extraction()
```

The LLM extracts:
- Summary  
- Sentiment  
- Risky phrases  
- Contradictions  
- Credibility score  
- Narrative insights  
- Behavioural warning indicators  

These features are transformed into numerical inputs for risk scoring.

---

## 5.4 ⭐ Rule-Based Overrides

To mimic real underwriting, the system includes rules such as:
- Too many late payments → high risk  
- Too many new accounts → high risk  
- Stable employment + one late payment → moderate risk  

These ensure reliability even when LLM text is complex.

---

## 5.5 ⭐ Risk Score Prediction

Using:

```
integrations.predict()
```

The system outputs:
- risk_score (0 to 1)  
- risk_label (low / moderate / high)  
- explanation text  

---

## 5.6 ⭐ Recommendation Logic

| Risk Score | Decision |
|------------|----------|
| 0.70 – 1.00 | 🔴 Decline / Manual Review |
| 0.40 – 0.69 | 🟡 Conditional Approval |
| 0.00 – 0.39 | 🟢 Approve |

This appears inside the **Final Recommendation** panel.

---

## 5.7 ⭐ Explainable Dashboard (Streamlit)

Includes:
- KPI cards (Applicants Count, Avg Income, High Risk %)  
- Applicant table  
- Local explanation panel  
- Risk score breakdown  
- Sentiment evaluation  
- Extracted risk flags  

---

## 6. Tech Stack

| Layer             | Technology / Libraries                 | Purpose |
|------------------|----------------------------------------|---------|
| **Frontend**     | Streamlit, Altair                      | UI, interactive dashboard, charts & data visualization |
| **LLM Engine**   | google-genai                           | Behavioural analysis, feature extraction, LLM scoring |
| **Parsing**      | pdfplumber, pypdf, easyocr, pytesseract, Pillow | OCR, PDF parsing, and raw text extraction |
| **Data Handling**| pandas                                 | Data loading, cleaning, processing |
| **ML Core**      | scikit-learn, joblib                   | ML model training, scoring, and artifact saving |
| **Backend**      | Python 3.10+                           | Core logic & application pipeline |
| **Caching**      | JSON / Local Artifacts                 | Deterministic runs & API cost reduction |
 
---

# 7. Architecture Diagram

```
User Uploads File(s)
        ↓
Text Extraction (PDF/OCR)
        ↓
Field Parsing (Income, Name, etc.)
        ↓
LLM Behavioural Analysis
        ↓
Rule-Based Risk Overrides
        ↓
Risk Score Prediction
        ↓
Interactive Dashboard (Streamlit)
```

---

# 8. How To Use

## Step 1 — Install Requirements
```
pip install -r requirements.txt
```

## Step 2 — Run the App
```
streamlit run app.py
```

## Step 3 — Upload Applicants
Upload CSV or raw documents (PDF/JPG/PNG).

## Step 4 — Click **Run Model**
View:
- Summary  
- Sentiment  
- Extracted risky phrases  
- Risk flags  
- Final score  
- Recommendation  

---

# 9. Folder Structure

```
backend/app.py                 # Main Streamlit application
backend/integrations.py        # LLM + heuristic feature extraction + scoring
backend/ui_helpers.py          # KPI cards, tables, helpers
requirements.txt
data/
```

---

# 10. Deployment

This app can be deployed on:

- Streamlit Community Cloud  
- Render  
- Local Docker environment  

Deployment Steps:
1. Push repo to GitHub  
2. Open Streamlit Cloud  
3. Select repository and choose `app.py` as entry file  
4. Add secrets (API keys if needed)  

---

# 11. Judging Criteria Alignment

### ✔ Improved Credit Decision-Making
Uses behavioural + numeric indicators.

### ✔ Effective Use of LLM Textual Analysis
Extracts narrative meaning and risk signals.

### ✔ Interpretability
Shows evidence, summaries, and explanations.

### ✔ End-to-End Functionality
Complete pipeline from upload → extraction → scoring → dashboard.

---

# 12. Team Members
* Esther Kong Yuan Er (UI Lead): Streamlit layout, Visuals.
* Lim Hui Yun (Integrator): Wiring Backend to UI, Error handling.
* Lim Xuan Ning (AI/LLM): Prompts, Feature Engineering, Model Training.
* Ee Mei Xuan (Testing/Demo): QA, Fairness checks, Demo recording.
* Ng Yue Qhi (Backend Lead): Repo setup, Architecture.

---

# 12. License
MIT

---
