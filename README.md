# 🚀 Corporate CI/CD Pipeline for ML Flask App

A production-grade CI/CD pipeline built using **Jenkins, Docker, Kubernetes (EKS), and Argo CD (GitOps)** to deploy a **Flask-based Sentiment Analysis application** powered by Machine Learning.

---

## 📌 Project Overview

This project demonstrates an end-to-end DevOps pipeline:

* Build → Test → Containerize → Push → Deploy → Monitor
* Fully automated using **CI/CD + GitOps principles**
* Designed as a **final-year engineering project**

The deployed app performs **sentiment analysis** on user input text and returns:

* ✅ Positive
* ❌ Negative
* 📊 Confidence score

---

## 🛠️ Tech Stack

* **Backend**: Python, Flask
* **ML**: XGBoost, Scikit-learn, NLTK
* **CI**: Jenkins (Multibranch Pipeline)
* **CD**: Argo CD (GitOps)
* **Containerization**: Docker
* **Orchestration**: Kubernetes (AWS EKS)
* **Monitoring**: Prometheus + Grafana

---

## ⚙️ CI/CD Workflow

1. Code pushed to GitHub (`main` branch)
2. Jenkins pipeline triggers:

   * Builds Docker image
   * Pushes to Docker Hub
   * Updates Kubernetes manifest
3. Argo CD detects changes (GitOps)
4. Automatically deploys to EKS
5. Kubernetes performs rolling update
6. Application exposed via LoadBalancer

---

## 📂 Project Structure (Essential Files Only)

```
Corporate-CI-CD-pipeline/
│
├── app.py
├── config.py
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
├── .dockerignore
│
├── models/
│   ├── countVectorizer.pkl
│   └── model_xgb.pkl
│
├── templates/
│   ├── home.html
│   ├── result.html
│   └── base.html
│
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── kubernetes/
│   ├── deployment.yml
│   └── service.yml
│
└── argocd/
    └── application.yml
```

---

## 🐳 Docker

Build and run locally:

```bash
docker build -t sentiment-app .
docker run -p 5000:5000 sentiment-app
```

---

## ☸️ Kubernetes Deployment

Apply manifests:

```bash
kubectl apply -f kubernetes/
```

---

## 🔁 Jenkins Pipeline Stages

* Checkout Code
* Build Docker Image
* Push to Docker Hub
* Update Kubernetes Manifest

---

## 🔐 Required Credentials

### Jenkins:

* `dockerhub-creds` → Docker Hub Access Token
* `github-creds` → GitHub Personal Access Token

---

## 📊 Monitoring

* Prometheus → Metrics collection
* Grafana → Visualization dashboards

---

## 💡 Key Features

* GitOps-based deployment (Argo CD)
* Fully automated CI/CD pipeline
* Production-ready Docker image (multi-stage build)
* Kubernetes health checks (liveness + readiness)
* Scalable architecture

---

## ⚠️ Important Notes

* Ensure `models/*.pkl` files are present before running
* Use **Java 21 for Jenkins**
* Use **Docker Hub token (not password)**
* Always clean up AWS resources after testing (to avoid cost)

---

## 🧠 Learning Outcomes

* Real-world CI/CD pipeline implementation
* Kubernetes deployment strategies
* GitOps vs traditional deployment
* Containerization best practices

---

## 📎 Reference

Project inspired by a CI/CD tutorial and adapted for ML deployment. 

---

## 👨‍💻 Author

**Raj Rishi Samanta**
Final Year Engineering Project

---

⭐ If you found this useful, consider giving it a star!
