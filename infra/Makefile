# Infrastructure and local development Makefile for Measured Fit
SHELL := /bin/bash

# Configuration defaults
REGION ?= us-central1
PROJECT ?= $(GOOGLE_CLOUD_PROJECT)
API_SERVICE ?= fit-api
GEOMETRY_SERVICE ?= fit-geometry
READER_SERVICE ?= fit-reader
TASKS_QUEUE ?= fit-jobs-queue

# Cloud Run base flags (per specification). infra/deploy.sh adds memory, auth and env-file flags per service
# (geometry: GEOMETRY_FLAGS below = --cpu=4 --memory=8Gi --timeout=900) and --no-traffic once a revision is pinned.
BASE_DEPLOY_FLAGS := --region=$(REGION) --min-instances=0 --max-instances=3 --cpu-boost
GEOMETRY_FLAGS := $(BASE_DEPLOY_FLAGS) --cpu=4 --memory=8Gi --timeout=900

# Dry-run handling: when DRY_RUN=1, print command without executing
ifeq ($(DRY_RUN),1)
EXEC_PREFIX := echo "[DRY_RUN=1]"
else
EXEC_PREFIX :=
endif

.PHONY: setup seed smoke all dev test deploy-api deploy-geometry deploy-reader deploy-web create-queue help

setup:
	bash infra/setup.sh

seed:
	python seed/seed.py $(SEED_ARGS)

smoke:
	python infra/smoke_test.py $(BASE_URL)

help:
	@echo "Available targets:"
	@echo "  dev              Start Firebase emulators and three FastAPI services (uvicorn --reload)"
	@echo "  test             Run offline unit test suite across schemas, units, and llm"
	@echo "  deploy-api       Deploy API service to Cloud Run (supports DRY_RUN=1)"
	@echo "  deploy-geometry  Deploy Geometry service to Cloud Run with 4 CPU & 8Gi (supports DRY_RUN=1)"
	@echo "  deploy-reader    Deploy Reader service to Cloud Run (supports DRY_RUN=1)"
	@echo "  deploy-web       Deploy web app to Firebase Hosting (supports DRY_RUN=1)"
	@echo "  create-queue     Create Cloud Tasks queue in GCP (supports DRY_RUN=1)"

dev:
	@echo "Starting Firebase emulators and local FastAPI microservices..."
	@echo "  -> API service on http://localhost:8000"
	@echo "  -> Geometry service on http://localhost:8001"
	@echo "  -> Reader service on http://localhost:8002"
	@echo "  -> Firebase emulators & Web dev server"
	firebase emulators:start & \
	uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload & \
	uvicorn services.geometry.main:app --host 0.0.0.0 --port 8001 --reload & \
	uvicorn services.reader.main:app --host 0.0.0.0 --port 8002 --reload

test:
	pytest packages/fitcore/tests -v
	python -m pytest services/api/tests -v
	python -m pytest services/reader/tests -v

deploy-api:
	DRY_RUN=$(DRY_RUN) BASE_DEPLOY_FLAGS="$(BASE_DEPLOY_FLAGS)" bash infra/deploy.sh api

deploy-geometry:
	DRY_RUN=$(DRY_RUN) BASE_DEPLOY_FLAGS="$(BASE_DEPLOY_FLAGS)" bash infra/deploy.sh geometry

deploy-reader:
	DRY_RUN=$(DRY_RUN) BASE_DEPLOY_FLAGS="$(BASE_DEPLOY_FLAGS)" bash infra/deploy.sh reader

deploy-web:
	@cd web && npm run build
	$(EXEC_PREFIX) firebase deploy --only hosting $${PROJECT:+--project=$$PROJECT}

create-queue:
	$(EXEC_PREFIX) gcloud tasks queues create $(TASKS_QUEUE) \
		--location=$(REGION) \
		$${PROJECT:+--project=$$PROJECT}
