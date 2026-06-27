pipeline {
    agent { label params.NODE_LABEL }

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30'))
    }

    triggers {
        // Run once per day. Adjust as needed.
        cron('H 2 * * *')
    }

    parameters {
        string(name: 'NODE_LABEL', defaultValue: 'mlp-validate-icx24-ubuntu', description: 'Jenkins node label')
        text(name: 'EMAIL_TO', defaultValue: 'xiangdong.zeng@intel.com', description: 'Recipients, comma-separated')
        string(name: 'EMAIL_FROM', defaultValue: 'dgpu_validation@intel.com', description: 'Sender email address')
        password(name: 'SMTP_PASSWORD', defaultValue: '', description: 'SMTP password')
        password(name: 'BUILDKITE_TOKEN', defaultValue: '', description: 'Buildkite API token (bkua_...)')
        string(name: 'SMTP_HOST', defaultValue: 'smtp.intel.com', description: 'SMTP server host')
        string(name: 'SMTP_PORT', defaultValue: '587', description: 'SMTP server port, e.g. 587 for STARTTLS')
        string(name: 'SMTP_USERNAME', defaultValue: '', description: 'SMTP username (if empty, EMAIL_FROM is used)')
    }

    environment {
        SMTP_FROM = "${params.EMAIL_FROM}"
        SMTP_HOST = "${params.SMTP_HOST}"
        SMTP_PORT = "${params.SMTP_PORT ?: '587'}"
        SMTP_USERNAME = "${params.SMTP_USERNAME ?: params.EMAIL_FROM}"
        SMTP_PASSWORD = "${params.SMTP_PASSWORD}"
        BUILDKITE_TOKEN = "${params.BUILDKITE_TOKEN}"
    }

    stages {
        stage('Checkout') {
            steps {
                deleteDir()
                checkout scm
            }
        }

        stage('Validate Inputs') {
            steps {
                script {
                    if (!env.BUILDKITE_TOKEN?.trim()) {
                        error('BUILDKITE_TOKEN is empty.')
                    }
                    if (!params.EMAIL_TO?.trim()) {
                        error('EMAIL_TO is empty.')
                    }
                    if (!env.SMTP_FROM?.trim()) {
                        error('EMAIL_FROM is empty.')
                    }
                    if (!env.SMTP_HOST?.trim()) {
                        error('SMTP_HOST is empty.')
                    }
                }
            }
        }

        stage('Run Analyzer And Send Mail') {
            steps {
                sh '''
                    set -eux
                    mkdir -p "${WORKSPACE}/output"

                    python3 scripts/llmbench/buildkite_intel_ci_analyzer.py \
                      --days 1 \
                      --output-dir "${WORKSPACE}/output/buildkite_intel_ci_${BUILD_NUMBER}" \
                      --email-to "${EMAIL_TO}" \
                      --email-from "${SMTP_FROM}" \
                      --smtp-host "${SMTP_HOST}" \
                      --smtp-port "${SMTP_PORT}" \
                      --smtp-starttls
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'output/buildkite_intel_ci_*/summary.*', allowEmptyArchive: true
            archiveArtifacts artifacts: 'output/buildkite_intel_ci_*/failed_jobs.*', allowEmptyArchive: true
        }
        failure {
            // Optional fallback notification when pipeline fails before analyzer email is sent.
            emailext(
                subject: "[FAILED] Buildkite Intel CI Analyzer Job - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                to: params.EMAIL_TO,
                body: "Jenkins job failed. Please check ${env.BUILD_URL}",
                mimeType: 'text/plain'
            )
        }
    }
}
