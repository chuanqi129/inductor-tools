import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cpu", help="Devices to use")
    parser.add_argument("--dtypes", default="float32", help="Data types")
    parser.add_argument("--suites", default="torchbench", help="Suites to run")
    parser.add_argument("--compilers", default="eager,inductor", help="Compilers")
    parser.add_argument("--flag-compilers", default="", help="Flag compilers")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--keep-output-dir", action="store_true", help="Keep output directory")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--print-run-commands", action="store_true", help="Print run commands")
    action_group.add_argument("--visualize-logs", action="store_true", help="Visualize logs")
    action_group.add_argument("--run", action="store_true", help="Run the benchmarks")
    parser.add_argument("--log-operator-inputs", action="store_true", help="Log operator inputs")
    parser.add_argument("--include-slowdowns", action="store_true", help="Include slowdowns")
    parser.add_argument("--extra-args", default="", help="Extra arguments")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--inference", action="store_true", help="Run inference")
    mode_group.add_argument("--training", action="store_true", help="Run training")
    parser.add_argument("--base-sha", default=None, help="Base SHA")
    parser.add_argument("--total-partitions", type=int, default=1, help="Total partitions")
    parser.add_argument("--partition-id", type=int, default=0, help="Partition ID")
    parser.add_argument("--update-dashboard", action="store_true", help="Update dashboard")
    parser.add_argument("--no-graphs", action="store_true", help="Do not generate graphs")
    parser.add_argument("--no-update-archive", action="store_true", help="Do not update archive")
    parser.add_argument("--no-gh-comment", action="store_true", help="Do not comment on GitHub")
    parser.add_argument("--no-detect-regressions", action="store_true", help="Do not detect regressions")
    parser.add_argument("--update-dashboard-test", action="store_true", help="Update dashboard test")
    parser.add_argument("--dashboard-image-uploader", default=None, help="Dashboard image uploader")
    parser.add_argument("--dashboard-archive-path", default=None, help="Dashboard archive path")
    parser.add_argument("--archive-name", default=None, help="Archive name")
    parser.add_argument("--dashboard-gh-cli-path", default=None, help="Dashboard gh CLI path")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    parser.add_argument("--enable-cpu-launcher", action="store_true", help="Enable CPU launcher")
    parser.add_argument("--cpu-launcher-args", default=None, help="CPU launcher args")
    parser.add_argument("--no-cold-start-latency", action="store_true", help="Skip cold start latency")
    parser.add_argument(
        "--inductor-compile-mode",
        type=str,
        default="default",
        help="Inductor compile mode",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Run in channels-last memory format",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
