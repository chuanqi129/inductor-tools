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
        booleanParam(name: 'SMTP_STARTTLS', defaultValue: true, description: 'Enable STARTTLS (usually with port 587)')
        booleanParam(name: 'SMTP_SSL', defaultValue: false, description: 'Enable SMTP over SSL (usually with port 465)')
    }

    environment {
        SMTP_FROM = "${params.EMAIL_FROM}"
        SMTP_HOST = "${params.SMTP_HOST}"
        SMTP_PORT = "${params.SMTP_PORT ?: '587'}"
        SMTP_USERNAME = "${params.SMTP_USERNAME}"
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
                    def missing = []
                    if (!env.BUILDKITE_TOKEN?.trim()) {
                        missing << 'BUILDKITE_TOKEN'
                    }
                    if (!params.EMAIL_TO?.trim()) {
                        missing << 'EMAIL_TO'
                    }
                    if (!params.EMAIL_FROM?.trim()) {
                        missing << 'EMAIL_FROM'
                    }

                    if (missing) {
                        echo "Input validation failed. Missing/empty params: ${missing.join(', ')}"
                        error("Required parameters are missing: ${missing.join(', ')}")
                    }
                }
            }
        }

        stage('Run Analyzer And Send Mail') {
            steps {
                script {
                    sh """
                        set -eux
                        mkdir -p \"${WORKSPACE}/output\"

                        python3 scripts/llmbench/buildkite_intel_ci_analyzer.py \\
                          --days 1 \\
                          --output-dir \"${WORKSPACE}/output/buildkite_intel_ci_${BUILD_NUMBER}\"

                        cp \"${WORKSPACE}/output/buildkite_intel_ci_${BUILD_NUMBER}/summary.html\" \"${WORKSPACE}/output/latest_summary.html\"
                    """
                }
            }
        }
    }

    post {
        success {
                        script {
                                String reportPath = 'output/latest_summary.html'
                                String reportHtml = fileExists(reportPath) ? readFile(file: reportPath) : '<html><body><p>Report file not found.</p></body></html>'
                                String embeddedReportHtml = reportHtml
                                        .replaceAll('(?is)<style[^>]*>.*?</style>', '')
                                        .replaceAll('(?is)</?(html|head|body)[^>]*>', '')
                                String buildUrl = env.BUILD_URL ?: ''
                                String artifactUrl = buildUrl ? "${buildUrl}artifact/output/latest_summary.html" : ''

                                String mailHtml = """
                                <html>
                                    <head>
                                        <style>
                                            .email-report { font-family: Arial, Helvetica, sans-serif; color: #1f2937; }
                                            .email-report p { margin: 0 0 8px 0; line-height: 1.35; }
                                            .email-report h1, .email-report h2, .email-report h3 { margin: 0 0 8px 0 !important; line-height: 1.2 !important; }
                                            .email-report hr { margin: 10px 0; border: 0; border-top: 1px solid #d1d5db; }
                                            .email-report .page { max-width: 1200px !important; padding: 8px !important; }
                                            .email-report .hero, .email-report .section { padding: 12px !important; margin-top: 10px !important; border-radius: 8px !important; }
                                            .email-report .metrics, .email-report .grid { gap: 8px !important; margin-top: 8px !important; }
                                            .email-report .metric-card { padding: 10px !important; }
                                            .email-report table { width: 100%; border-collapse: collapse !important; table-layout: fixed; margin: 6px 0 !important; font-size: 13px !important; }
                                            .email-report th, .email-report td { padding: 6px 8px !important; border: 1px solid #d1d5db !important; vertical-align: top; line-height: 1.25 !important; word-break: break-word; }
                                            .email-report th { background: #f3f4f6; }
                                            .email-report .table-wrap { overflow: visible !important; }
                                        </style>
                                    </head>
                                    <body>
                                        <div class=\"email-report\">
                                            <p>Buildkite Intel CI daily report is ready.</p>
                                            <hr/>
                                            ${embeddedReportHtml}
                                            <hr/>
                                            <p>
                                                Jenkins Build: <a href=\"${buildUrl}\">${buildUrl}</a><br/>
                                                Archived Report: <a href=\"${artifactUrl}\">${artifactUrl}</a>
                                            </p>
                                        </div>
                                    </body>
                                </html>
                                """.stripIndent()

                                emailext(
                                        subject: "[DAILY] Buildkite Intel CI Report - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                                        to: params.EMAIL_TO,
                                        from: params.EMAIL_FROM,
                                        mimeType: 'text/html; charset=UTF-8',
                                    body: mailHtml,
                                    attachmentsPattern: 'output/latest_summary.html'
                                )
                        }
        }
        always {
            archiveArtifacts artifacts: 'output/buildkite_intel_ci_*/summary.*', allowEmptyArchive: true
            archiveArtifacts artifacts: 'output/buildkite_intel_ci_*/failed_jobs.*', allowEmptyArchive: true
            archiveArtifacts artifacts: 'output/latest_summary.html', allowEmptyArchive: true
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
