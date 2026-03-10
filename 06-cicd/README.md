# 06 – CI/CD: Automated Training & Deployment with GitHub Actions

This module uses a modern CI/CD pipeline that automates the training, packaging, and testing of our application. The main workflow (`ci-cd.yml`) calls a reusable training job, builds a self-contained Docker image with the model "baked in," and pushes it to the **GitHub Container Registry (GHCR)**.

Render is then used to pull this pre-built, validated image and run it as a live web service.

**Alignment with 05-monitoring:** Same model (PU_DO + trip_distance, XGBRegressor), same API (`/predict`, `/health`). You can point `simulate.py` at a deployed Render URL to log predictions for drift monitoring.

---

## 🔁 Workflow Concept

The process is a clean, automated flow for building and testing the application, followed by a manual deployment step on Render.

```
Git Push
    │
    ▼
┌──────────────────────────────────────────┐
│ CI/CD Pipeline (ci-cd.yml)               │
│                                          │
│  1. Calls train.yml → creates artifact   │
│  2. Lints & tests the code               │
│  3. Builds & tests Docker image          │
│  4. Pushes image to GHCR                 │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ GitHub Container Registry (GHCR)         │
│                                          │
│  Stores the final, versioned image:      │
│  ghcr.io/<user>/<repo>:latest            │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼ (Manual Deploy on Render)
┌──────────────────────────────────────────┐
│ Render.com                               │
│                                          │
│  Pulls the image from GHCR and runs it   │
│  as a live web service.                  │
│                                          │
└──────────────────────────────────────────┘
```
---

## 🗂️ Key Files

| File | Role |
| :--- | :--- |
| `train.py` | Trains the model and saves a production-ready copy to `models/model/`. |
| `app.py` | FastAPI service that loads the local `models/model/` at startup. |
| `test_api.py` | Automated tests run against the live Docker container to ensure quality. |
| `.github/workflows/ci-cd.yml` | **The main orchestrator**: calls training, lints, builds, tests, and pushes the image. |
| `.github/workflows/train.yml` | A **reusable component** dedicated solely to running `train.py`. |
| `Dockerfile` | A blueprint to package the code and the trained model into one container. |

**Note on `requirements.txt`:** There are two `requirements.txt` files. The one in the root directory is for your local development environment (`.venv`). The one inside `06-cicd/` is specifically for the `Dockerfile`, ensuring the container has only the dependencies it needs for production.

---


## 🚀 First-Time Deployment Guide

Follow these steps in order to deploy the project to your own Render account.

### 1️⃣ Validate Your Code Locally (Pre-Flight Check)

Before pushing your code, always run the linter locally on the application code. This catches style errors early and prevents the CI/CD pipeline from failing.

From the root directory of the repository, run:
```bash
flake8 06-cicd
```
If this command shows no output, you are good to go!

### 2️⃣ Commit and Push Your Code

Commit and push all the latest files from this module to your GitHub repository's `main` branch.

```bash
git add .
git commit -m "feat: Finalize CI/CD pipeline for module 06"
git push origin main
```
This push will automatically trigger the CI/CD pipeline.

### 3️⃣ Wait for the Pipeline to Succeed

Go to your repository's **Actions** tab and wait for the **CI/CD Pipeline** to complete. This first run builds your Docker image and pushes it to GHCR, but it will be **private** by default.

### 4️⃣ Make the Docker Image Public (One-Time Action)

You must make the image package public so Render can access it.

1.  On your repository's main page, go to the **Packages** section on the right sidebar.
2.  Click on your image name (e.g., `ie-mlops-course`).
3.  On the image's page, go to **Package settings**.
4.  Scroll to the "Danger Zone" and change the visibility to **Public**.

### 5️⃣ Create the Render Service

1.  On the Render Dashboard, click **New → Web Service**.
2.  Choose **"Deploy an existing image from a registry"**.
3.  For the Image URL, enter `ghcr.io/<your-github-user>/<your-repo-name>` (all lowercase).
4.  Give the service a name, select the **Free** instance type, and click **Create Web Service**.

### 6️⃣ Verify Your Deployment

Once Render is live (≈1–2 min), check the health endpoint. You can find your service's URL on the Render dashboard.

```bash
curl https://<your-service-name>.onrender.com/health
```
The response should show `"status":"ok"` and `"model_loaded":true`.

---

## 🔄 How to Redeploy with Changes

1.  Make your code changes (`train.py`, `app.py`, etc.).
2.  Run `flake8 06-cicd` locally to check for issues.
3.  `git commit` and `git push` them to `main`.
4.  Wait for the **CI/CD Pipeline** to complete successfully on GitHub.
5.  Go to your Render dashboard, find your service, and click **Manual Deploy → Deploy latest commit**.

