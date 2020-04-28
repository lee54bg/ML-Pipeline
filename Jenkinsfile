pipeline {
    agent { label 'ubuntu-slave1' }
    stages {
        stage('Environment Setup') {
            steps {
                sh 'echo "Setting up the environment"'
                sh 'python3 -m venv main'
                sh 'cd main'
                sh 'source bin/activate'
                sh 'pip install --upgrade pip'
                sh 'pip install sklearn'
                sh 'pip install pandas'
            }
        }
        stage('Build') {
            steps {
                // sh 'echo "World"'
                sh 'python3 test.py'
            }
        }
        stage('Test') {
            steps {
                sh 'echo "Starting the test phase..."'
            }
        }
        stage('Cleanup') {
            steps {
                sh 'echo "Starting the cleanup phase..."'
                sh 'python3 removefiles.py'
                // sh '''
                //     echo "This should not go through"
                //     ls -lah
                // '''
            }
        }
    }
}
