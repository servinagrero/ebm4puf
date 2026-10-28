#!/usr/bin/env bash

ARTIFACT=./data/puf_artifact.h5
DATA_STUDY_DIR=./data/${STUDY_NAME}
FIGURE_STUDY_DIR=./figures/${STUDY_NAME}
RBM_STUDY_NAME=rbm
RBM_DATA_STUDY_DIR=./data/${RBM_STUDY_NAME}
RBM_FIGURE_STUDY_DIR=./figures/${RBM_STUDY_NAME}

# uv run python scripts/run_artifact_rbm_studies.py \
#     --artifact "${ARTIFACT}" \
#     --study-name "${RBM_STUDY_NAME}" \
#     --output-dir "${RBM_DATA_STUDY_DIR}"

# uv run python scripts/plot_rbm_studies.py \
#     --study-name "${RBM_STUDY_NAME}" \
#     --study-dir "${RBM_DATA_STUDY_DIR}" \
#     --figures-dir "${RBM_FIGURE_STUDY_DIR}"
