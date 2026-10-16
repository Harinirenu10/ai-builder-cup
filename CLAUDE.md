# Project: measured fit for fashion commerce (Google Cloud AI Builder Cup, Retail & Commerce)
Deadline: prototype submitted by Oct 18. Optimize for a working, demoable vertical slice over completeness.
Thesis: "Measured, not guessed." We measure the body (a 3D twin from 4 photos) and the garment (one flat-lay photo next to an A4 sheet) so shoppers and businesses can trust size and fit answers.

## Hard constraints
- AI: Gemini models only, through the google-genai Python SDK on Gemini Enterprise Agent Platform (formerly Vertex AI): genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location="global"). Model IDs come only from env vars GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_IMAGE_MODEL, VTO_MODEL. Never hardcode a model ID. Never use any gemini-1.x or gemini-2.x model. Dev backends (LLM_BACKEND, default auto): text-only calls try Gemma on Ollama first (OLLAMA_BASE_URL, OLLAMA_MODEL), then fall back to Gemini; image parts are supported on Ollama too (base64, resized); real body photos should still use Vertex. The Gemini Developer API key (GEMINI_API_KEY) is for dev only and must be used sparingly: prefer Ollama, keep Gemini calls few (cap GEMINI_API_MAX_CALLS, default 30), use recorded fixtures in tests, never commit or paste the key into any file, command or log; it lives in the environment or the gitignored .env. The submitted demo uses Gemini on Vertex.
- Every Gemini call goes through packages/fitcore/fitcore/llm.py (structured JSON output via pydantic schemas, retries, latency logging).
- Hosting: Cloud Run for services (min instances 0), Firebase Hosting for the web app, Firestore + Cloud Storage for data, Cloud Tasks for async jobs. Ask before adding any other cloud service.
- Perception: MediaPipe Pose Landmarker. Body model: SMPL via the smplx package; model files are license-gated and placed manually in services/geometry/models/smpl/.
- Agents: Google ADK (google-adk) for the planner agent.
- Privacy: raw body photos are deleted after processing unless the user opts in to keep the front photo. Measurements and shape parameters are readable only by their owner. Business-facing endpoints return verdicts only, never measurements. Users confirm they are 18+ before body capture.
- Language rules for any Gemini text shown to users: describe the garment, never judge the body. No words about weight, flaws or flattering.
- Never scrape behind logins or bypass captchas. Label simulated data as simulated in the UI.

## Repo layout
- packages/fitcore: pure-Python shared library (schemas, llm wrapper, units, fit engine, garment photo measurement, reviews, wash, tailor card), installed into every Python service.
- services/api: FastAPI. Customer and business endpoints, ADK planner agent, orchestration, Cloud Tasks enqueueing.
- services/geometry: FastAPI. MediaPipe, SMPL fitting, critic loop, poses, meshes. CPU only.
- services/reader: FastAPI + Playwright. Product page reading.
- web: React + Vite + TypeScript + Tailwind. Areas: /app (customer PWA), /biz (business console), /store (fictional mock storefront "Demo Threads Co."). 3D via @google/model-viewer.
- seed, eval, samples, infra.

## Conventions
- Python 3.11, type hints, pydantic v2 for all API I/O and Gemini outputs. Centimeters internally everywhere.
- MeasurementZone enum: chest, waist, hip, shoulder_width, sleeve_length, torso_length, garment_length, inseam, thigh, neck, back_rise, upper_back.
- Firestore collections: users, twins, jobs (+ rounds), products, fits, garments, wardrobe (per user), events, gifts, merchants, consents, orders, returns, charts, sims.
- Each module ships a small CLI that runs on files in samples/ without the cloud.
- Tests: pytest, offline by default; Gemini calls in tests use recorded fixtures in tests/fixtures/llm/.
- Done means: works locally, tests pass, deploys with make deploy-<service>, one README section.
- Deterministic code for math and geometry; Gemini for perception, reasoning and language.
