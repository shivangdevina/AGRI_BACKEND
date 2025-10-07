
<div align="center">
  ![WhatsApp Image 2025-10-04 at 00 36 09_291b11eb](https://github.com/user-attachments/assets/35210a2e-902f-4ddd-9d23-1d574d4b0266)

  
  <h1>AgriNOVA: AI-Powered Personal Farming Assistant</h1>
</div>

---
---
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

AgriNOVA builds a dynamic **farmer profile** during the first interaction, capturing key parameters such as location, land size, crop type, soil condition (from user input or API), irrigation method, and personal preferences.  
This profile forms the foundation for **localized and adaptive recommendations**, enabling the system to tailor advice to the farmer’s agro-climatic context and reduce repetitive questioning.  

Each interaction—like sowing logs, fertilizer use, or pest reports—is stored in the **activity history**, allowing the backend to recognize seasonal patterns, crop cycles, and previous outcomes.  
This continuous learning loop refines future recommendations, making them **progressively personalized** and context-aware. A built-in **feedback improvement system** captures user ratings or corrections, ensuring that AgriNOVA evolves with every farmer interaction to deliver smarter, more reliable, and user-specific guidance over time.


---

### 2. Conversational Interface (Multilingual)

AgriNOVA’s conversational layer automatically detects the farmer’s language and supports both **voice and text** inputs.  
Queries are internally translated for reasoning and then delivered back in the farmer’s native language for maximum clarity.  
A built-in **Text-to-Speech (TTS)** module converts AI-generated responses into natural, human-like audio, allowing farmers with limited literacy to **listen to recommendations** instead of reading them.  
Multi-intent queries are intelligently segmented and routed to the relevant tools, ensuring each part of the question is addressed accurately.  
Throughout, the assistant maintains a **human tone** — concise, empathetic, and safety-focused — while providing traceable, cited information.


---

### 3. Activity Tracking

AgriNOVA maintains a structured **activity tracking module** that records all key farm operations — sowing, irrigation, fertilizer usage, pesticide application, and pest or disease sightings.
Each log entry is timestamped, geotagged (when permission is granted), and linked to the farmer’s profile, enabling correlation with weather forecasts, soil data, and crop growth stages.  

The system uses this data to generate **context-aware alerts** (e.g., “Irrigation due tomorrow”) and **predictive insights** (e.g., elevated pest probability post-rainfall).  
A continuous **feedback learning loop** allows farmers to rate or correct past recommendations; this data feeds into a **correction memory**, ensuring model fine-tuning and **progressive accuracy improvement** in future advisories.


---

### 4. Personalized Advisory

AgriNOVA’s advisory engine generates contextual, data-backed recommendations through three integrated layers, orchestrated by an **agentic AI system** that autonomously selects, sequences, and validates tool outputs for each farmer query.

- **Evidence:** Retrieved from the **vector-based knowledge base (RAG)** and **AgriKG knowledge graph**, combining factual information from agricultural manuals, extension guides, and domain-linked entities such as crop–disease–nutrient relationships. This ensures every suggestion is grounded in verified, interconnected knowledge.  

- **Tools:** Driven by specialized modules for **crop advisory**, **fertilizer planning**, **pest and disease diagnosis**, **government schemes and notices**, **web search**, **mandi price analysis**, and **soil/weather API integration**.   

- **Constraints:** Dynamic filters that incorporate **weather windows**, **government advisories**, **regulatory bans**, and **PPE/safety compliance** to ensure safe and legally valid recommendations.  

The backend synthesizes all these inputs into **step-by-step, actionable guidance** — prioritizing preventive and non-chemical interventions first, followed by precise input details (dose, timing, and precautions).  
When system confidence is low or conflicting outputs are detected, the **agentic AI system** revalidates with alternate tools or defaults to **conservative advice**, optionally prompting the user for additional data (e.g., clearer crop image or updated soil parameters) before finalizing the recommendation.


### 5. Reminders & Alerts

AgriNOVA’s **notification and alert system** integrates multiple data sources to keep farmers informed and proactive.  
Timely alerts are generated based on the farmer’s **profile, activity history, and weather forecasts**, using real-time feeds from **Disaster Management APIs** for extreme weather, flood, or drought warnings.  

Routine reminders for operations such as irrigation, fertilizer application, and pest inspection are delivered through the **AI Insights module**, which analyzes recent activity logs and forecast data to recommend optimal timing.  

All alerts are **multichannel** — farmers receive messages via **Twilio SMS** for offline accessibility and **in-app notifications** within the web interface for connected users.  
These reminders are **localized and language-optimized**, ensuring that critical warnings (e.g., disaster alerts) and regular advisories (e.g., input schedules, scheme deadlines, mandi trends) are clear, actionable, and easily understood by every farmer.


---

### 6. Knowledge Engine

AgriNOVA’s **Knowledge Engine** serves as the cognitive backbone of the advisory ecosystem, combining **retrieval-augmented generation (RAG)** with structured reasoning.  
It draws contextually relevant information from a **vectorized agricultural knowledge base**—comprising crop guides, pest and disease manuals, fertilizer schedules, and regional cropping calendars—while also linking to **semantic sources like the AgriKG knowledge graph** to establish logical relationships between crops, pests, nutrients, and environmental factors.  

The engine continuously synchronizes with **government advisories**, **scientific updates**, and **market intelligence feeds** to maintain regulatory accuracy and temporal relevance.  
Through this hybrid of static domain knowledge and dynamic web intelligence, AgriNOVA ensures every response is **evidence-backed, legally compliant, and geographically localized**, enabling precise, actionable, and trustworthy guidance for farmers.


---

### 7. Wellbeing & Safety Layer (Sentiment Analysis & Escalation)

AgriNOVA uses **sentiment analysis** on voice and text inputs to detect stress or distress indicators in farmer interactions.  
A calculated **Red Score** quantifies emotional risk based on language tone, frequency of negative terms, and contextual triggers like crop loss or debt.  
When the score exceeds a set threshold, the system automatically **switches to empathetic response mode**, pauses hazardous recommendations, and **alerts a designated reporting authority** .  
Sensitive data remains anonymized, ensuring farmer safety, privacy, and emotional wellbeing.


---

<img width="1092" height="406" alt="arch01 drawio" src="https://github.com/user-attachments/assets/d1de4539-b4f0-4d5c-a073-9995041141b8" />



### System Architecture Overview

1. **Pre-processing:** Detects the farmer’s language, performs translation to English, and prepares the query for structured understanding.  
2. **Personal History (Profile + Chat History):** Stores user-specific data including past queries, accepted recommendations, and behavioral patterns for personalized responses.  
3. **History Tracker:** Logs previous interactions and outcomes to enable adaptive learning and contextual continuity.  
4. **Query Expansion:** Decomposes and reformulates user inputs into multiple focused sub-queries for tool routing.  
5. **Original (Language Layer):** Preserves the farmer’s native-language input for translation consistency and user trust.  
6. **Agent AI (LLM + Tool Executors):** The central orchestrator applying **ReAct (Reasoning + Action + Observation)**—deciding which tools to invoke, merging their outputs, and synthesizing a unified response.  
7. **Weather API:** Fetches localized real-time and forecast weather data (rainfall, humidity, wind, temperature) to provide time-sensitive AI insights.  
8. **Soil API:** Retrieves region-specific soil data (NPK levels, pH, moisture) used for crop and fertilizer recommendations.  
9. **Fertilizer and Crop Recommendation:** Generates agronomic Fertilizer and Crop recommendatios based on soil and climate data provided(or fetched through api).    
10. **Gemini API + CNN Model:** Performs disease and pest identification through multimodal image analysis and CNN-based classification.  
11. **Web Search (Tavily/Google):** Retrieves latest online content—schemes, advisories, or research updates—to complement local knowledge.  
12. **Mandi Price (Model/DSS):** Collects and analyzes market price trends to suggest optimal selling periods and locations.  
13. **RAG Search + Knowledge Graph:** Combines vector-based retrieval with **AgriKG** and local crop databases to deliver verified, region-specific insights on practices, calendars, and pest management.  
14. **Red Score & Reporting System:** Uses sentiment analysis to detect distress and generate a wellbeing score; escalates critical cases to local authorities for support while keeping all data anonymized to ensure farmer privacy and safety.
15. **Post Processing:** Translates the AI-generated English response back into the farmer’s original language while preserving technical accuracy.  
16. **User Interface (Web):** Presents the final AI response through a multilingual chat interface with voice, text, and notification support.

---


## How an Answer Is Produced (End-to-End Logic)

1. Ingest: Receive text/voice (any language) and optional image.  
2. Language & Intent: Detect → translate → split into sub-queries.  
3. Route:  
   - Weather → Forecast Tool  
   - Image → Pest/Disease Detector  
   - Fertilizer/Crop → Advisory Tool (soil + history)  
   - Schemes → Govt Advisories Tool  
   - References → RAG Knowledge Base  
   - Market → Web & Price Tools
   - Genral Query → RAG & Web Search Tool
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

## API Endpoints Overview

### 1. Root Endpoint
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **GET** | `/` | Health check for Crop Recommendation API. |

---

### 2. Authentication APIs
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **POST** | `/auth/signup` | Register a new user (name, email, password, role). |
| **POST** | `/auth/login` | Authenticate user and set a refresh token cookie. |
| **GET**  | `/auth/me` | Retrieve current logged-in user profile via token. |
| **GET**  | `/auth/hi` | Fetch all registered users (for admin/debug use). |

---

### 3. Chat System APIs
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **GET**  | `/chats/` | Retrieve all chat sessions for a user (optionally filter by chat type). |
| **POST** | `/chats/createSession` | Create a new chat session for a user. |
| **GET**  | `/chats/{chat_id}` | Retrieve a specific chat session and metadata. |
| **POST** | `/chats/addMessage` | Add a user message, invoke the agentic AI model, and store AI-generated response. |
| **GET**  | `/chats/messages/{session_id}` | Fetch complete message history for a session. |
| **DELETE** | `/chats/{chat_id}` | Delete a specific chat session and its messages. |

---

### 4. Activity Tracker APIs
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **GET**  | `/api/activity/` | Retrieve recorded farm activities (sowing, irrigation, fertilizer use, pest logs) for a user. |
| **POST** | `/api/activity/add` | Add a new farm activity record (linked to user profile). |
| **GET**  | `/api/activity/insights` | Generate AI insights and schedule reminders based on logged activities and forecasts. |

---

### 5. ML Model APIs
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **POST** | `/api_model/pest_detection_and_analyze` | Analyze uploaded crop image for pest or disease detection using CNN + Vision Model. |
| **POST** | `/api_model/crop_recommendations` | Generate AI-based crop recommendations based on soil and weather parameters. |
| **POST** | `/api_model/fertilizer_recommendation` | Provide optimized fertilizer recommendations using ML + agronomic rules engine. |

---

### 6. Red Score & Wellbeing APIs
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **POST** | `/api/redscore/evaluate` | Analyze user messages using sentiment analysis to compute Red Score and wellbeing level. |
| **POST** | `/api/redscore/log` | Store Red Score, emotional indicators, and escalation status into the wellbeing database. |
| **GET**  | `/api/redscore/history/{userid}` | Retrieve a user’s historical Red Score trends for monitoring and support tracking. |
| **POST** | `/api/redscore/escalate` | Escalate critical Red Score cases to assigned local authorities (FPO/KVK/NGO) with user consent. |

---

### Notes:
- **CORS Enabled:** Allows secure frontend-backend communication.  
- **Supabase Database:** Manages user data, chat logs, activities, and wellbeing reports.  
- **Agentic AI Core:** Dynamically selects and orchestrates model tools (RAG, AgriKG, APIs) for personalized recommendations.  
- **Red Score Module:** Ensures farmer emotional wellbeing through sentiment detection, safe response adjustment, and optional escalation.  
- **Data Privacy:** All sensitive or emotional data remains anonymized and securely stored.


---

## Reliability & Safety Principles

- Triangulation: Vision + KB + web verification for accuracy.  
- Weather-aware: Avoid actions that may fail or harm crops.  
- Legal-first: Government advisories filter unsafe/banned inputs.  
- Human-centered: Prioritize safety and empathy when confidence is low.

---

AgriNOVA — bridging intelligence and empathy for every farmer, everywhere.
