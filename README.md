# LLM-Based Credit Risk Assessment (Track 2)
### THE ROOKIES

## ⭐ Overview

This repository contains a complete end-to-end prototype for **LLM-based credit risk assessment**, combining structured applicant data with unstructured text extracted from PDFs, images and CSVs. The system integrates OCR, regex-based parsing, LLM-powered behavioural analysis, rule-based overrides and transparent explainability through a Streamlit dashboard.

The app demonstrates how modern credit scoring can be improved using **Large Language Models (LLMs)** to capture behavioural risk signals beyond traditional numeric fields.

---

# 1. Industry Context

Traditional credit scoring models rely heavily on:

- Income  
- Credit score  
- Liabilities  
- Payment history  

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
- **Story Playback** (step-by-step reasoning replay)

---

# 6. Architecture Diagram

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

# 7. How To Use

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

# 8. Folder Structure

```
backend/app.py                 # Main Streamlit application
backend/integrations.py        # LLM + heuristic feature extraction + scoring
backend/ui_helpers.py          # KPI cards, tables, helpers
requirements.txt
data/
```

---

# 9. Deployment

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

# 10. Judging Criteria Alignment

### ✔ Improved Credit Decision-Making
Uses behavioural + numeric indicators.

### ✔ Effective Use of LLM Textual Analysis
Extracts narrative meaning and risk signals.

### ✔ Interpretability
Shows evidence, summaries, and explanations.

### ✔ End-to-End Functionality
Complete pipeline from upload → extraction → scoring → dashboard.

---

# 11. Team Members
* Member A (Backend Lead): Repo setup, Architecture.
* Member B (UI/UX Lead): Streamlit layout, Visuals.
* Member C (Integrator): Wiring Backend to UI, Error handling.
* Member D (AI/LLM): Prompts, Feature Engineering, Model Training.
* Member E (Testing/Demo): QA, Fairness checks, Demo recording.

---

# 12. License
MIT

---
