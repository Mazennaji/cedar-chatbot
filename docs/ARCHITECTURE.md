# 🏛️ Cedar Chatbot — Architecture

## System Overview

Cedar Chatbot is built as a modular, layered system with clear separation between the NLP engine, API layer, and presentation layer.

## Layers

### 1. NLP Engine (`src/`)

The core intelligence layer, entirely framework-agnostic:

- **`normalizer.py`** — Converts Lebanese Arabizi to Arabic script using phrase-level matching, word-level dictionary lookup, and character-level transliteration with longest-match-first strategy.
- **`language_detector.py`** — Classifies input as English, Arabic MSA, Lebanese Arabic, or Lebanese Arabizi using script analysis, numeral-as-letter detection, and dialect vocabulary matching.
- **`intent_classifier.py`** — Rule-based intent classification across 8 categories with trilingual keyword patterns.
- **`sentiment.py`** — Lexicon-based multilingual sentiment analysis with negation and intensifier handling.
- **`memory.py`** — Session-based conversation memory with sliding window, automatic expiry, and feedback storage.
- **`chatbot.py`** — Orchestrator that chains all components into a single `chat()` pipeline.

### 2. RL Module (`src/rl/`)

- **`reward_model.py`** — BiLSTM + Attention reward model trained on pairwise human preferences.
- **`trainer.py`** — PPO-based policy optimizer for RLHF fine-tuning.

### 3. API Layer (`api/`)

FastAPI-based async REST API:

- **Routes** — Chat, feedback, analytics endpoints.
- **Middleware** — JWT authentication, token-bucket rate limiting, CORS.
- **WebSocket** — Real-time bidirectional chat support.

### 4. Django Dashboard (`dashboard/`)

Admin and analytics dashboard:

- **Core** — Persistent session/message/feedback models with Django Admin.
- **Analytics** — Daily metrics aggregation, language/intent/sentiment distributions.

### 5. Streamlit UI (`ui/`)

Interactive chat interface with RTL Arabic support, metadata display, and inline feedback buttons.

## Data Flow

```
User Input (EN / AR / Arabizi)
    │
    ├─► Language Detector ──► classify language
    ├─► Arabizi Normalizer ──► convert if Arabizi
    ├─► Intent Classifier ──► classify intent
    ├─► Sentiment Analyzer ──► score sentiment
    │
    ▼
Memory Manager ──► retrieve context, store message
    │
    ▼
Transformer Engine ──► generate response
    │
    ▼
Reward Model ──► score response quality
    │
    ▼
Response + Metadata ──► return to client
```

## Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| NLP Model | BlenderBot 400M | Conversational, open-source, fine-tunable |
| API | FastAPI | Async, auto-docs, WebSocket, Pydantic |
| Dashboard | Django | ORM, admin panel, mature ecosystem |
| UI | Streamlit | Rapid prototyping, Python-native |
| RL | PyTorch + Custom PPO | Full control over training loop |
| Auth | Custom JWT | Lightweight, no external deps |