@NonCPS
String shellQuote(Object value) {
    return "'${String.valueOf(value).replace("'", "'\"'\"'")}'"
}

@NonCPS
String unwrapIntelCommand(String command) {
    def matcher = command =~ /(?s)^\s*bash\s+\.buildkite\/scripts\/hardware_ci\/run-intel-test\.sh\s+'(.*)'\s*$/
    return matcher.matches() ? matcher.group(1) : command
}

@NonCPS
String revisionFromImage(String dockerImage, String fallbackRevision) {
    def matcher = dockerImage =~ /:([0-9a-fA-F]{40})-xpu$/
    return matcher.find() ? matcher.group(1) : fallbackRevision
}

@NonCPS
List<Map<String, String>> serializableFileDetails(Object files) {
    return files.collect { file ->
        [name: String.valueOf(file.name), path: String.valueOf(file.path)]
    }
}

@NonCPS
int retryAttempts(Map stepConfig) {
    def automatic = stepConfig.retry instanceof Map ? stepConfig.retry.automatic : null
    if (!(automatic instanceof List)) {
        return 1
    }
    int retryLimit = 0
    for (def rule : automatic) {
        int ruleLimit = rule instanceof Map ? (rule.limit ?: 0) as int : 0
        if (ruleLimit > retryLimit) {
            retryLimit = ruleLimit
        }
    }
    return retryLimit + 1
}

@NonCPS
List<Map> collectIntelTests(List<Map> yamlDocuments) {
    List<Map> tests = []
    yamlDocuments.each { document ->
        String yamlName = document.yamlName
        Map config = document.config
        (config.steps ?: []).eachWithIndex { rawStep, stepIndex ->
            if (!(rawStep instanceof Map) || !rawStep.commands || rawStep.device != 'intel_gpu') {
                return
            }

            List<String> commands = rawStep.commands instanceof List ? rawStep.commands : [rawStep.commands]
            String testCommand = commands.collect { unwrapIntelCommand(String.valueOf(it)) }.join(' && ')
            tests << [
                yamlName: yamlName,
                label: String.valueOf(rawStep.label ?: "step-${stepIndex + 1}"),
                command: testCommand,
                env: rawStep.env instanceof Map ? rawStep.env : [:],
                timeoutMinutes: (rawStep.timeout_in_minutes ?: 60) as int,
                retryAttempts: retryAttempts(rawStep)
            ]
        }
    }
    return tests
}

node(params.TEST_NODE) {
    String vllmRepo = 'https://github.com/vllm-project/vllm.git'
    String vllmBranch = 'main'
    List<Map> intelTests = []
    String activeContainer = ''

    try {
        stage('Prepare') {
            deleteDir()

            if (!params.DOCKER_IMAGE?.trim()) {
                error('DOCKER_IMAGE must not be empty')
            }
            if (!params.HF_HOME?.trim() || !params.HF_HOME.startsWith('/')) {
                error('HF_HOME must be an absolute path on the TEST_NODE host')
            }

            String vllmRevision = revisionFromImage(params.DOCKER_IMAGE.trim(), vllmBranch)
            echo "Using vLLM revision ${vllmRevision}"

            sh """#!/usr/bin/env bash
                set -euxo pipefail
                mkdir -p ${shellQuote(params.HF_HOME)} logs
                git clone ${shellQuote(vllmRepo)} vllm
                git -C vllm checkout ${shellQuote(vllmRevision)}
                docker pull ${shellQuote(params.DOCKER_IMAGE)}
            """

            List<Map<String, String>> yamlFiles = serializableFileDetails(
                findFiles(glob: 'vllm/.buildkite/intel_jobs/*.yaml')
            )
            writeFile(
                file: 'logs/discovery.log',
                text: "Found ${yamlFiles.size()} Intel CI YAML files\n"
            )
            echo "Found ${yamlFiles.size()} Intel CI YAML files"

            List<Map> yamlDocuments = []
            for (def yamlFile : yamlFiles) {
                echo "Reading ${yamlFile.path}"
                yamlDocuments << [
                    yamlName: yamlFile.name,
                    config: readYaml(file: yamlFile.path) as Map
                ]
            }
            intelTests = collectIntelTests(yamlDocuments)
            if (!intelTests) {
                error('No Intel GPU tests were found in .buildkite/intel_jobs/*.yaml')
            }
            echo "Discovered ${intelTests.size()} Intel CI test steps from ${yamlDocuments.size()} YAML files; YAML parallelism is ignored"
        }

        intelTests.eachWithIndex { test, testIndex ->
            String stageName = "${testIndex + 1}. ${test.label}".take(120)
            String logName = String.format('%03d-%s.log', testIndex + 1, test.yamlName.replace('.yaml', ''))

            stage(stageName) {
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE', catchInterruptions: false) {
                    timeout(time: test.timeoutMinutes, unit: 'MINUTES') {
                        retry(test.retryAttempts) {
                            Map<String, String> containerEnv = [
                                HF_HOME: '/root/.cache/huggingface',
                                HUGGINGFACE_HUB_CACHE: '/root/.cache/huggingface/hub',
                                HF_HUB_VERBOSITY: 'info',
                                PYTHONUNBUFFERED: '1',
                                VLLM_TEST_DEVICE: 'xpu',
                                VLLM_DISABLE_COMPILE_CACHE: '1'
                            ]
                            test.env.each { key, value ->
                                containerEnv[String.valueOf(key)] = String.valueOf(value)
                            }
                            if (env.http_proxy) containerEnv.http_proxy = env.http_proxy
                            if (env.https_proxy) containerEnv.https_proxy = env.https_proxy
                            if (env.HTTP_PROXY) containerEnv.HTTP_PROXY = env.HTTP_PROXY
                            if (env.HTTPS_PROXY) containerEnv.HTTPS_PROXY = env.HTTPS_PROXY
                            if (env.no_proxy) containerEnv.no_proxy = env.no_proxy
                            if (env.NO_PROXY) containerEnv.NO_PROXY = env.NO_PROXY

                            String envArgs = containerEnv.collect { key, value ->
                                "--env ${shellQuote("${key}=${value}")}"
                            }.join(' ')
                            String containerName = "vllm-intel-${env.BUILD_NUMBER}-${testIndex}"
                            activeContainer = containerName
                            String commandFile = "logs/${logName}.command.sh"
                            String preflight = testIndex == 0 ? '''
echo "Visible DRM devices:"
ls -la /dev/dri /dev/dri/by-path || true
echo "XPU runtime status:"
timeout 30 xpu-smi discovery || true
python3 -c 'import torch; print("torch.xpu available:", torch.xpu.is_available()); print("torch.xpu device count:", torch.xpu.device_count())'
echo "Hugging Face cache: ${HF_HOME}"
du -sh "${HF_HOME}" 2>/dev/null || true
''' : ''

                            writeFile(
                                file: "logs/${logName}.metadata.txt",
                                text: """yaml=${test.yamlName}
label=${test.label}
timeout_minutes=${test.timeoutMinutes}
retry_attempts=${test.retryAttempts}
parallelism=ignored
"""
                            )
                            writeFile(
                                file: commandFile,
                                text: "#!/usr/bin/env bash\nset -e\necho \"Starting Intel CI test\"\necho \"Working directory: \$(pwd)\"\n${preflight}${test.command}\n"
                            )
                            sh """#!/usr/bin/env bash
                                set -o pipefail
                                docker rm -f ${shellQuote(containerName)} >/dev/null 2>&1 || true
                                docker run --rm \\
                                    --name ${shellQuote(containerName)} \\
                                    --entrypoint /bin/bash \\
                                    --device /dev/dri:/dev/dri \\
                                    --privileged \\
                                    --network host \\
                                    --ipc host \\
                                    -v /dev/dri/by-path:/dev/dri/by-path \\
                                    -v ${shellQuote(params.HF_HOME)}:/root/.cache/huggingface \\
                                    -v ${shellQuote("${env.WORKSPACE}/${commandFile}:/tmp/vllm-jenkins-test.sh:ro")} \\
                                    --env HF_TOKEN \\
                                    ${envArgs} \\
                                    ${shellQuote(params.DOCKER_IMAGE)} \\
                                    -e /tmp/vllm-jenkins-test.sh \\
                                    2>&1 | tee -a ${shellQuote("logs/${logName}")}
                            """
                            activeContainer = ''
                        }
                    }
                }
            }
        }
    } finally {
        if (activeContainer) {
            sh "docker rm -f ${shellQuote(activeContainer)} >/dev/null 2>&1 || true"
        }
        stage('Archive Logs') {
            archiveArtifacts artifacts: 'logs/**', allowEmptyArchive: true, fingerprint: true
        }
    }
}