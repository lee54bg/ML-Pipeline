pipeline {
    // This will be used for the Jenkins Slave
    agent { label 'jenkinsworker1' }
    stages {
        stage('Environment Setup') {
            steps {
                sh 'echo "Setting up the environment"'
                sh 'python3 -m venv main'
                sh 'cd main \
                    && source bin/activate \
                    && pip install --upgrade pip \
                    && pip install wheel \
                    && pip install sklearn \
                    && pip install pandas \
                    && pip install numpy \
                    && pip install matplotlib \
                '
            }
        }
        stage('Build') {
            steps {
                sh 'echo "Preprocessing and training in progress..."'
                sh 'main/bin/python3 main/random_forest_classification.py'
            }
        }
        stage('Test') {
            steps {
                sh 'echo "Starting the test phase..."'
                sh 'main/bin/python3 main/ml_test.py'
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
        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'foo.pdf', fingerprint: true, onlyIfSuccessful: true
            }
        }
        stage('Deploy') {
            steps {
                sh 'echo "Deploying the model..."'
            }
        }
    } // End of stages
    // Send email after everything is finished
    post {
        always {
            emailext body: 'A Test EMail', recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'RequesterRecipientProvider']], subject: 'Test'
        }
    }
}
