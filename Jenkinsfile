pipeline {
    agent { label 'ubuntu-slave1' }
    stages {
        stage('Build') {
            steps {
                // sh 'echo "World"'
                sh 'python3 test.py'
                sh '''
                    echo "This should not go through"
                    ls -lah
                '''
            }
        }
        stage('Test') {
            steps {
                sh 'echo "Starting the test phase..."'
                sh 'Executing script'
                // sh '''
                //     echo "This should not go through"
                //     ls -lah
                // '''
            }
        }

    }
}
