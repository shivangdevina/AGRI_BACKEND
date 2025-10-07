# AgriNOVA Backend – Farmer-First Multilingual Farm Assistant

AgriNOVA is a farmer-first backend that turns scattered agricultural data — weather, soil, markets, advisories, local documents, and images — into one clear, safe, and actionable answer in the farmer’s own language.  
It combines Retrieval-Augmented Generation (RAG) with dedicated tools for crop advisory, fertilizer planning, pest & disease detection, weather forecasting, government advisories, and market prices.

---

## What This Backend Does (At a Glance)

- Understands any language & images: Auto-detects the user’s language, translates internally, and analyzes crop photos for pests or diseases.  
- Finds the right facts fast: Pulls verified information from a local agri knowledge base and supplements it with live data (weather, schemes, mandi prices).  
- Thinks like an agronomist: Merges outputs from all tools into timing-aware, actionable guidance (e.g., “don’t spray today—rain forecast”).  
- Personalizes safely: Learns from each farmer’s profile, history, and context; applies government advisories and safety checks before recommendations.  

---

## Problem Domains We Solved

### 1. Farmer & Farm Profiling
We capture essential farm data during initial interaction — location, land size, crop type, soil condition, irrigation method, and farmer preferences.  
This enables localized recommendations, faster follow-ups, and relevant default assumptions (e.g., rain-fed vs. irrigated).  
Over time, AgriNOVA refines this profile using past outcomes, creating more personalized and accurate guidance.

---

### 2. Conversational Interface (Malayalam-First, Multilingual)
AgriNOVA’s conversational layer detects language automatically and handles both voice and text inputs.  
Queries are translated internally for reasoning and then returned in the farmer’s own language.  
Multi-intent messages are split and routed to respective tools. The chatbot maintains a human tone — concise, empathetic, and safety-focused — while citing relevant sources.

---

### 3. Activity Tracking
Farm activities such as sowing, irrigation, fertilizer application, and pest sightings are logged in simple terms.  
These logs power contextual nudges (e.g., “Irrigation due tomorrow”) and predictive insights (e.g., higher pest risk after rain).  
Feedback loops allow farmers to mark helpful or incorrect advice, feeding a “correction memory” that continually improves future recommendations.

---

### 4. Personalized Advisory
Each recommendation is built from three logical layers:
- Evidence: Retrieved knowledge (RAG snippets, government notices, soil/weather context).  
- Tools: Specialized engines for crop advisory, fertilizer planning, and pest diagnosis.  
- Constraints: Weather windows, regulatory bans, PPE & safety compliance.  

The system generates step-by-step guidance — prioritizing non-chemical actions, followed by precise inputs (dose, timing, and precautions).  
If confidence is low, the assistant defaults to conservative measures or requests additional inputs (clearer image, soil data).

---

### 5. Reminders & Alerts
Based on profile, activity, and forecast data, the system issues timely, context-aware alerts, such as:
- “Rain expected in 18–24 hrs — avoid spraying today.”  
- “Urea split-dose window opens Friday.”  
- “PMFBY subsidy deadline in 3 days.”  
- “Mandi modal price rising — consider selling next week.”  

These reminders are tailored in simple local language to maximize farmer comprehension.

---

### 6. Knowledge Engine
AgriNOVA’s core RAG layer retrieves concise, validated passages from crop guides, extension documents, pest management notes, and regional calendars.  
It integrates government advisories to ensure legal compliance and market/web updates to stay current.  
This ensures every recommendation remains accurate, practical, and region-specific.

---

### 7. Wellbeing & Safety Layer (Sentiment Analysis & Escalation)

AgriNOVA uses **sentiment analysis** on voice and text inputs to detect stress or distress indicators in farmer interactions.  
A calculated **Red Score** quantifies emotional risk based on language tone, frequency of negative terms, and contextual triggers like crop loss or debt.  
When the score exceeds a set threshold, the system automatically **switches to empathetic response mode**, pauses hazardous recommendations, and **alerts a designated reporting authority** .  
Sensitive data remains anonymized, ensuring farmer safety, privacy, and emotional wellbeing.


---

## Wellbeing & Safety Layer (Sentiment + Red Score + Escalation)

- Purpose: Detect farmer distress and provide supportive, safe responses (non-clinical).  
- Signal Inputs: Message sentiment, red-flag keywords, recent stressors (crop loss, debt, extreme weather), and behavior patterns.  
- Red Score Bands:  
  - Green (0–2): Normal tone, regular guidance.  
  - Amber (3–5): Empathetic tone, gentle steps, suggest SHG/KVK helplines.  
  - Red (6+): Share official helplines (Kiran 1800-599-0019, 112), pause risky advice, offer consent-based escalation to reporting authority (FPO/KVK/NGO).  
- Actions: Simplified replies, reassurance-first tone, no pesticide handling in Red zone, optional follow-up checks.  
- Privacy: Explicit consent before any alert; minimal info shared; opt-out available anytime.  
- Impact: Preserves dignity, prevents harm, and builds trust by treating mental wellbeing as part of farm safety.

---

## How an Answer Is Produced (End-to-End Logic)

1. Ingest: Receive text/voice (any language) and optional image.  
2. Language & Intent: Detect → translate → split into sub-queries.  
3. Route:  
   - Weather → Forecast Tool  
   - Image → Vision → Pest/Disease Detector  
   - Fertilizer/Crop → Advisory Tool (soil + history)  
   - Schemes → Govt Advisories Tool  
   - References → RAG Knowledge Base  
   - Market → Web & Price Tools  
4. Cross-Check: Align results, remove conflicts, apply weather/legal constraints.  
5. Personalize & Safeguard: Adapt tone, include PPE and regulation notes.  
6. Synthesize & Translate: Merge outputs into one coherent plan (today → later), translate back to user language.  
7. Record & Learn: Log activity, outcomes, and corrections for continuous improvement.

---

## What the Farmer Receives

- One-line main answer (diagnosis or action).  
- Short reasoning & confidence score (for image-based cases).  
- Action checklist (immediate & later tasks).  
- Timing and safety windows (weather, PPE, bans).  
- Optional extras (scheme or market info).  
- Compact citations (so the farmer can verify).

---

## Expected Impact

- Empowers farmers with personalized, on-demand support.  
- Enhances productivity & sustainability through context- and timing-aware actions.  
- Bridges the knowledge gap by combining AI intelligence with local agronomic expertise.  

---

## Example (Quick Walk-through)

Farmer uploads a tomato leaf photo and asks:  
“Is this blight? What to spray? Will it rain tomorrow?”

- Vision → Detects Early Blight (medium confidence).  
- Weather → Heavy rain predicted tomorrow.  
- Advisory → “Prune affected leaves today, delay systemic sprays, use rain-fast fungicide after rain.”  
- Govt Layer → No local bans; references IPM guidelines.  
- RAG → Adds cultural control measures.  
- Final Reply: Concise, step-by-step plan (with doses & safety), translated back to Malayalam.

---

## Reliability & Safety Principles

- Triangulation: Vision + KB + web verification for accuracy.  
- Weather-aware: Avoid actions that may fail or harm crops.  
- Legal-first: Government advisories filter unsafe/banned inputs.  
- Human-centered: Prioritize safety and empathy when confidence is low.

---

AgriNOVA — bridging intelligence and empathy for every farmer, everywhere.
