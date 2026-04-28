pipeline {
    agent any
    options {
        disableConcurrentBuilds()
    }
    environment {
        IMAGE_NAME = "rajrishidockerhub/sentiment-app"
        GIT_USER   = "johnthebrat7"
        GIT_EMAIL  = "rajrishisamanta2@gmail.com"
        GIT_REPO   = "johnthebrat7/Corporate-CI-CD-pipeline"
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build and Push Image') {
            when { branch 'main' }
            steps {
                script {
                    env.IMAGE_TAG = "build-${BUILD_NUMBER}"
                    withCredentials([usernamePassword(
                        credentialsId: 'dockerhub-cred',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh """
                        docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                        echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin
                        docker push ${IMAGE_NAME}:${IMAGE_TAG}
                        """
                    }
                }
            }
        }
        stage('Update K8s Manifest') {
            when { branch 'main' }
            steps {
                script {
                    withCredentials([usernamePassword(
                        credentialsId: 'github-cred',
                        usernameVariable: 'GIT_USERNAME',
                        passwordVariable: 'GIT_TOKEN'
                    )]) {
                        sh """
                        set -e
                        git config user.name "${GIT_USER}"
                        git config user.email "${GIT_EMAIL}"
                        git fetch origin
                        git checkout main
                        git reset --hard origin/main
                        sed -i "s|image:.*|image: ${IMAGE_NAME}:${IMAGE_TAG}|" kubernetes/deployment.yml
                        git add kubernetes/deployment.yml
                        git diff --cached --quiet || git commit -m "ci: update image to ${IMAGE_TAG} [skip ci]"
                        git push https://\$GIT_USERNAME:\$GIT_TOKEN@github.com/${GIT_REPO}.git main
                        """
                    }
                }
            }
        }
    }
    post {
        always {
            sh 'docker image prune -f || true'
        }
    }
}