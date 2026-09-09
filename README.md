# 🛡️ Clinical Risk Monitor: AI-Powered Pharmacovigilance System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clinicalriskmonitor-ai.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.0-green)

## 📋 Overview
The **Clinical Risk Monitor** is a real-time clinical decision support system (CDSS) designed to bridge the gap between raw patient data and actionable ICU insights. 

Unlike standard dashboards, this application combines **Deterministic Logic** (Clinical Rules like qSOFA) with **Probabilistic Machine Learning** (XGBoost & Generative AI) to predict patient deterioration before it happens.

### 🚀 Live Demo
**[Click here to launch the application](https://clinicalriskmonitor-ai.streamlit.app)**

---

## ⚙️ Key Features

### 1. 🧠 AI Clinical Consultant (Generative AI)
- Integrated **Google Gemini 2.0 Flash** to act as an automated medical resident.
- **Provider Mode:** Analyzes calculated risk scores (AKI, Bleeding) and generates a differential diagnosis and treatment plan.
- **Patient Mode:** A triage symptom checker that translates layperson terms into medical terminology.

### 2. 🩸 Bleeding Risk Prediction (Machine Learning)
- Deployed a pre-trained **XGBoost Regressor** (`bleeding_risk_model.json`).
- Predicts the probability of hemorrhage based on INR, Anticoagulant use, Age, and Comorbidities.
- Trained on synthetic acute care data.

### 3. ⚡ Real-Time Protocol Monitors (Clinical Logic)
- **Sepsis Watch:** Auto-calculates **qSOFA** scores based on Vitals (BP, Resp Rate, Mental Status).
- **AKI Monitor:** Tracks Creatinine spikes according to **KDIGO** guidelines.
- **Hypoglycemia Alert:** Flags critical glucose levels in diabetic patients.

### 4. 💊 Drug-Drug Interaction Checker
- Features a backend database of **High-Alert Medications** (Warfarin, Amiodarone, NSAIDs, etc.).
- Flags **Critical** (Contraindicated) and **Major** (Monitor closely) interactions instantly to prevent adverse drug events (ADEs).

---

## 🛠️ Technical Architecture

The application follows a modular **Model-View-Controller (MVC)** pattern:

* **Frontend (`app.py`):** Built with **Streamlit**. Handles UI rendering, session state, and input validation.
* **Backend (`backend.py`):** Pure Python logic layer. Handles:
    * **SQL Database:** SQLite integration for storing patient history.
    * **ML Inference:** Loading and querying the XGBoost model.
    * **API Gateway:** Secure connection to Google Gemini via Streamlit Secrets.
* **Data Visualization:** Used **Altair** for interactive vitals telemetry charts.

### 📂 Project Structure
```bash
├── app.py                 # Frontend Interface (Streamlit)
├── backend.py             # Logic & AI Controller (Python)
├── bleeding_risk_model.json # Pre-trained XGBoost Model
├── clinical_data.db       # SQLite Database (Patient History)
├── requirements.txt       # Dependencies
└── README.md              # Documentation

## 💻 How to Run Locally

If you want to run this on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Pravanith/ClinicalRiskApp.git](https://github.com/Pravanith/ClinicalRiskApp.git)
    cd ClinicalRiskApp
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    * Create a `.streamlit/secrets.toml` file.
    * Add your Google Gemini API Key: `GEMINI_API_KEY = "YOUR_KEY_HERE"`

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
## 📊 Methodology
1.  **Data Ingestion:** Simulating/Importing Electronic Health Record (EHR) data.
2.  **Preprocessing:** Handling missing vitals, normalizing lab values, and encoding medication classes.
3.  **Risk Logic:** Applying clinical rules (Pharm.D. domain knowledge) alongside statistical models.
4.  **Output:** Visual risk stratification (Low/Medium/High).

---

## 👨‍⚕️ About the Author
**Pravanith** | *Pharm.D. & Health Data Scientist*

This project demonstrates the intersection of **Clinical Expertise** and **Data Engineering**. It showcases the ability to not only analyze health data but to build deployed, scalable tools that improve patient safety.

* **Capstone Project:** Predicting Mental Health Severity from Digital Habits (R/Stats).
* **Engineering Project:** Clinical Risk Monitor (Python/ML).

---

*Disclaimer: This tool is a prototype for portfolio and educational purposes only. It uses [synthetic/anonymized] data and should not be used for actual medical diagnosis or treatment without clinical validation.*

## Version 4: voice-assisted workspace

The existing risk calculator, history, dashboard, CSV analysis, and medication tools remain available.

### Run

Use Python 3.9 or newer (Python 3.11 recommended):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Launch the dashboard and open **Risk Calculator**. Record an English note and press **Stop**. The recording is transcribed and recognized observations automatically populate the form. Recording does not stop automatically on silence. You can edit the transcript, apply it again, or undo the last fill. Confirm the observations, then run analysis; dictation never saves an assessment automatically.

Example:

> Age sixty eight. Gender female. Weight 70 kg. Height 170 cm. Blood pressure 120 over 80. Heart rate 76. Respiratory rate 18. Temperature 98.6 Fahrenheit. Oxygen saturation 98. Creatinine one point two. BUN 18. Potassium 4.2. Glucose 100. WBC 7.5. Hemoglobin 14. Platelets 220. INR 2.1. Lactate 1.2. Anticoagulant yes. No liver disease.

- Supports all numeric fields and explicit yes/no history flags; unmentioned values remain unchanged.
- Supports labelled numbers and English number words, Fahrenheit, pounds, and height in inches/meters. Use displayed form units for labs. Unsupported lab units and conflicting repeated values need manual review.
- This is a conservative parser, not unrestricted clinical language understanding. Use one observation per sentence. Do not dictate multiple patients or mix historical and current observations in one recording.
- Transcription uses `faster-whisper` (`base.en`, CPU). The first use downloads model weights and requires internet; subsequent use can run from the model cache. Audio stays on the app server for processing; on hosted deployments this means audio leaves your device. This feature does not send audio to Gemini or persist recordings/transcripts in the patient history database.
- Microphone capture requires browser permission and HTTPS or localhost. Text entry works if speech dependencies or model download are unavailable.

### Model upgrade

```sh
python train_model.py
```

The v4 estimator uses scikit-learn gradient boosting, avoiding the legacy XGBoost system runtime dependency. Training now uses reproducible stratified splits, seeded tuning, sigmoid probability calibration, and an untouched holdout. It writes `clinical_pipeline_v4.pkl` and `model_report.json`, preserving the original model. The app prefers the v4 artifact when present. The report compares calibrated and uncalibrated ROC-AUC, average precision, Brier score, and log loss. This retains the six existing model features; additional form fields feed existing rule calculations, not the bleeding estimator. All training data are synthetic; metrics are demonstration results and do not establish clinical performance.

qSOFA now returns its 0–3 point score consistently. AKI is labelled as a local heuristic, not a KDIGO diagnosis. Missing required observations block assessment saves. Existing clinical explanations and medication content still require independent clinical review.

### Verification

```sh
python -m unittest discover -s tests -v
```

Implementation references: [Streamlit recording](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input), [local transcription](https://github.com/SYSTRAN/faster-whisper), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), [Sepsis-3 qSOFA definition](https://pmc.ncbi.nlm.nih.gov/articles/4968574/).

## Version 5: admission-to-discharge workspace

Open **Hospital Workspace** after launching the dashboard:

1. **Register once:** Create a patient reference and demographic record. Select it for subsequent visits.
2. **Start an encounter:** Record the presenting complaint and ward. One open encounter per patient is supported; mark it admitted after reviewing intake.
3. **Voice intake:** Record and press Stop, or paste a note. Observations, explicitly stated disease history, and supported medication phrases populate an encounter-specific draft. Review each section and save it. Registration identity is kept separate from transcript extraction.
4. **Observations and risk:** Save timestamped observations. View the original bleeding model, the existing AKI heuristic, qSOFA, MAP, glucose context, and missing-input explanations. Previous diseases and the reviewed medication list inform applicable rules and AI review. The underlying bleeding model still uses its original six input features and synthetic training data.
5. **Medication reconciliation:** Enter dose per administration, unit, route, formulation, frequency, indication, and a clinician decision. Local checks identify exact-name allergy matches, duplicates, existing library interactions, and a limited adult immediate-release metformin label check. They do not verify arbitrary medications or provide comprehensive prescribing clearance.
6. **AI cross-check:** Review the outgoing clinical payload, then request Gemini review. Candidate openFDA product labels are retrieved by drug name. AI provides per-medication findings, missing information, qualitative risk considerations, nutrition considerations, and treatment questions for clinicians. Labels may not match the exact product; verify route, formulation, population, and the full label. An unavailable label cannot be treated as dosage verification. AI cannot clear deterministic label/allergy conflicts.
7. **Nutrition:** Record oral-intake clearance, food allergy context, preferences, and clinician/dietitian restrictions. Generic meal ideas are withheld for NPO/unknown intake, reported or unknown allergies, pediatric/pregnancy context, or renal/hepatic/heart-failure complexity. Recorded diet instructions take precedence over generic menus.
8. **Discharge:** Record diagnosis, hospital course, clinician-entered medication instructions, nutrition instructions, follow-up, return precautions, and clinician review. Download a Markdown draft or finalize the admitted encounter. Finalized encounters are read-only. New admissions prefill prior conditions, allergy text, and applicable discharge medications as unreviewed candidates, without copying old labs/vitals.

### AI configuration

In **Medications & AI → AI settings**, enter a Gemini API key and an available Gemini model ID for the session. Alternatively configure:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "your-available-model-id"
```

Keep these in `.streamlit/secrets.toml` (ignored by Git), or use environment variables. This machine was not configured for live Gemini requests during implementation; request/response handling is tested with mocked service responses. No AI review is fabricated when configuration or service access is unavailable.

Clicking **Send shown details for AI review** transmits the displayed clinical payload to Google and medication names to openFDA. Identity fields, raw transcripts, clinician names, and admission notes are excluded from the payload; free-text allergies and indications still need review for identifiers. Raw recordings are not stored by the encounter database. Saved narratives, encounter snapshots, and generated AI reviews are stored locally.

### Storage and limits

`care_patients`, `care_encounters`, and `care_events` are added to the existing SQLite database without migrating or deleting legacy `patient_history`. Stable encounter IDs, version checks, and transactional updates protect against accidental cross-patient saves and stale concurrent updates. Event snapshots preserve observation history. The timeline is not a tamper-proof audit log, and a typed clinician name is not an authenticated electronic signature.

This remains a local prototype: no authentication, role-based permissions, encryption-at-rest management, billing, bed allocation, laboratory/pharmacy interfaces, medication administration record, or validated clinical deployment is provided. Discharge completion records a clinician's documented decision; it is not an automated fitness-for-discharge decision. Use demonstration or appropriately authorized data. Production hospital use needs security, workflow, regulatory, and clinical validation.

Additional files: `care_store.py` (encounters), `care_logic.py` (intake and local reviews), `care_ai.py` (explicit AI calls and evidence), `care_ui.py` (workflow), and `tests/test_hospital.py`.

References: [openFDA labeling API](https://open.fda.gov/apis/drug/label/how-to-use-the-endpoint/), [metformin product labeling](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=54bb8030-8e80-4b38-8deb-89c99d73bf09), [CDC diabetes meal planning](https://www.cdc.gov/diabetes/healthy-eating/diabetes-meal-planning.html), [NIDDK kidney nutrition](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/healthy-eating-adults-chronic-kidney-disease), [NHLBI DASH](https://www.nhlbi.nih.gov/health/dash-eating-plan), and [Gemini API](https://ai.google.dev/api/generate-content).

## Broader admission observations

The Risk Calculator and Hospital Workspace observations now include optional pain score, Glasgow Coma Scale total, oxygen support/flow, measured urine volume and collection duration, sodium, chloride, bicarbonate/total CO2, total calcium, magnesium, phosphate, albumin, bilirubin, ALT, AST, alkaline phosphatase, and HbA1c. These values support documentation and clinician/AI context; they do not extend the trained bleeding model's feature set or create validated new disease predictions.

New fields start blank (not measured). Zero is preserved as an actual result for pain, oxygen flow, and urine volume. Urine output per kg per hour is calculated only when volume, collection duration, and weight are available. Use displayed units. Laboratory interpretation requires the local reference interval and patient context. Testing is selected by the treating team; these are not mandatory orders for every admission.

Voice example: “Sodium 138. Chloride 102. Bicarbonate 24. Calcium 9.2. Magnesium 2.0. Albumin 4.0. ALT 25. AST 24. HbA1c 6.5. Pain score zero. GCS 15. Urine volume 300. Urine duration six.” On stopping, review the transcript and populated fields as before.

The admission review table also records clinician-reviewed VTE/bleeding, falls/mobility, pressure-injury, nutrition/swallowing, and allergy/medication reviews. These start as “Not assessed” and are documentation prompts, not scored instruments. Relevant hospital-specific tools and clinical judgment still apply. The legacy screenshot's green “High” badges were replaced with neutral descriptions, qSOFA is explicitly labelled, and hypoglycemia heuristic points are no longer displayed as a probability in the risk-results panel.

References: [NICE admission observations](https://www.nice.org.uk/guidance/cg50/ifp/chapter/Arriving-on-the-ward-or-in-the-emergency-department), [MedlinePlus metabolic panel](https://medlineplus.gov/lab-tests/comprehensive-metabolic-panel-cmp/), [NICE VTE assessment](https://www.nice.org.uk/guidance/ng89/chapter/Recommendations), [NICE nutrition screening](https://www.nice.org.uk/guidance/cg32/chapter/Recommendations), [NICE pressure-injury assessment](https://www.nice.org.uk/guidance/cg179/chapter/Recommendations).
