from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.config import APP_DB_PATH, DATA_ROOT, PROJECTS_ROOT, TMP_UPLOAD_ROOT
from src.app.db import MetadataStore
from src.app.services.cv_pipeline import CvService
from src.app.services.pipeline import PipelineService
from src.app.web.routes import api, router
from src.app.worker import JobWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    store = MetadataStore(APP_DB_PATH)
    store.initialize()
    pipeline = PipelineService(store)
    cv_service = CvService(store)
    worker = JobWorker(store, pipeline, cv_service)
    worker.start()

    app.state.store = store
    app.state.pipeline = pipeline
    app.state.cv_service = cv_service
    app.state.worker = worker
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(
    title="TrollFish Local Workbench",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(api)
