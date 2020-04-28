pipeline {
    agent { label 'ubuntu-slave1' }
    stages {
        stage('Build') {
            steps {
                // sh 'echo "World"'
                sh 'python test.py'
                sh '''
                    echo "Multiline shell steps works too"
                    ls -lah
                '''
            }
        }
    }
}
