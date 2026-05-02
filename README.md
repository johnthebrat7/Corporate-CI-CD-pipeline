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

---

## Demo
<img width="940" height="468" alt="image" src="https://github.com/user-attachments/assets/274840c7-020e-4f9c-a1b7-aa649c5f1b8b" />

<img width="940" height="476" alt="image" src="https://github.com/user-attachments/assets/5c9738be-59db-441c-9b89-e17b21e7adef" />

<img width="940" height="472" alt="image" src="https://github.com/user-attachments/assets/19b372b1-a9be-4989-94b5-04225d76e93e" />

<img width="940" height="464" alt="image" src="https://github.com/user-attachments/assets/f74acf51-a40a-4802-8adc-7ee42b0df6e6" />

<img width="940" height="484" alt="image" src="https://github.com/user-attachments/assets/1079e137-ab14-4c9f-9cc4-d3a9ededa050" />

<img width="940" height="473" alt="image" src="https://github.com/user-attachments/assets/e01e1e6a-165b-428d-82cd-2694e37f755e" />

## Demo Video


https://drive.google.com/file/d/1ZCIQWqC0ArT1RSTtNOzafZmhJm35sA3L/view?usp=drive_link
