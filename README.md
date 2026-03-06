🚀 Production-Grade Sentiment Analysis CI/CD Pipeline
This repository demonstrates an end-to-end, production-ready DevSecOps pipeline for a Sentiment Analysis Web Application. The project leverages a Multibranch Pipeline strategy for Continuous Integration and a GitOps approach for Continuous Deployment, ensuring high availability, scalability, and security.

🏗️ Architecture Overview
The pipeline is split into two core components:

Continuous Integration (CI): Managed by Jenkins (Multibranch) to handle automated testing, building, and pushing of Docker images.

Continuous Deployment (CD): Managed by ArgoCD (GitOps) to synchronize the Kubernetes state with the manifest repository.

🛠️ Technology Stack


🔄 The Workflow
1. Developer Workflow (Branching Strategy)

Feature Branches: Developers create feature-specific branches (e.g., feature/improve-sentiment-accuracy) for code updates.
Pull Requests (PR): Once development is complete, a PR is raised to the main branch.
Main Branch: Acts as the stable production-ready code base.

3. Jenkins CI Pipeline (The Engine)

Jenkins automatically detects changes across all branches and performs the following:
Checkout: Pulls the latest code for the Sentiment Analysis app.
Build: Creates a Docker image containing the app and its ML dependencies.
Push: Tags the image with a unique build ID and pushes it to Docker Hub.
Manifest Update: Programmatically updates the deployment.yaml image tag in the repository to trigger the CD process.

3. ArgoCD GitOps (The Controller)

Pull-Based Deployment: ArgoCD monitors the k8s/ folder in this repository.
Sync: When Jenkins updates the image tag, ArgoCD detects "Out of Sync" and automatically pulls the new manifest to the Amazon EKS cluster. 

4. Monitoring & Observability

Prometheus: Scrapes real-time metrics from the Sentiment Analysis pods.
Grafana: Provides dashboards for monitoring CPU/Memory usage and request traffic. 
