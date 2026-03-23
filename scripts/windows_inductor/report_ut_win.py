# report_ut_win.py
import os
import re
import sys
from html import escape


def parse_log_file(filepath):
    """
    Parses a single log file to extract failed test case information.

    Args:
        filepath (str): The path to the log file.

    Returns:
        list: A list of dictionaries containing failed test information.
              Each dictionary has keys: 'file', 'class', 'test', 'time', 'log_file', 'wrapper'.
    """
    failed_tests = []
    print(f"Parsing log file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return []
    except Exception as e:
        print(f"Error: Problem reading file {filepath}: {e}")
        return []

    # Regular expression to match FAILED lines
    # Pattern: FAILED [time] file_path::class_name::test_name
    # Uses non-greedy matching (.*?) to handle special characters in paths, class names, and test names
    pattern = r"FAILED \[([0-9.]+)s\] (.*?\.py)::([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)"
    matches = re.findall(pattern, content)

    for match in matches:
        time_taken, file_path, class_name, test_name = match
        log_basename = os.path.basename(filepath)
        wrapper_type = "cpp wrapper" if log_basename.startswith("cpp") else "default wrapper"
        failed_tests.append(
            {
                "file": file_path,
                "class": class_name,
                "test": test_name,
                "time": float(time_taken),
                "log_file": log_basename,  # Record which log file it came from
                "wrapper": wrapper_type,
            }
        )

    print(f"  -> Found {len(failed_tests)} failed test cases in {filepath}.")
    return failed_tests


def generate_html_report(
    all_failed_tests,
    nightly_version,
    build_url="",
    output_filename="ut_test_failure_report.html",
):
    """
    Generates an HTML report based on the list of failed tests.

    Args:
        all_failed_tests (list): The list of all failed tests extracted from log files.
        output_filename (str): The name of the output HTML file.
    """
    print(f"\nGenerating HTML report: {output_filename}")

    # Sort by time taken in descending order (optional)
    # all_failed_tests.sort(key=lambda x: x['time'], reverse=True)

    build_url_section = ""
    if build_url:
        safe_url = escape(build_url)
        build_url_section = (
            f'<p><strong>Jenkins Build:</strong> <a href="{safe_url}">{safe_url}</a></p>'
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Inductor UT Test Failure Report on Windows</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #d9534f; /* Bootstrap danger color */
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .test-name {{
            font-family: monospace;
        }}
        .log-file {{
            font-style: italic;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>Inductor UT Test Failure Report on Windows</h1>
    <p><strong>PyTorch Nightly Wheel:</strong> {nightly_version}</p>
    {build_url_section}
    <p><strong>Total Failures:</strong> {len(all_failed_tests)}</p>
    <table>
        <thead>
            <tr>
                <th>Index</th>
                <th>Wrapper</th>
                <th>Test File</th>
                <th>Test Method</th>
            </tr>
        </thead>
        <tbody>
"""

    for i, test in enumerate(all_failed_tests, start=1):
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{test['wrapper']}</td>
                <td>{test['file']}</td>
                <td class="test-name">{test['test']}</td>
            </tr>"""

    html_content += """
        </tbody>
    </table>
</body>
</html>"""

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report successfully generated: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error: Problem writing HTML file: {e}")


def main(log_dir):
    if not os.path.isdir(log_dir):
        print(f"Error: The specified path '{log_dir}' is not a valid directory.")
        sys.exit(1)

    # the name of the subdir under log_dir is the PyTorch nightly wheel version
    subdirs = [
        d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))
    ]
    if not subdirs:
        print(f"Error: No nightly version directories found in '{log_dir}'.")
        sys.exit(1)
    nightly_version = subdirs[0]
    build_url = os.getenv("BUILD_URL", "")

    log_dir = os.path.join(log_dir, nightly_version)
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not log_files:
        print(f"Warning: No .log files found in directory '{log_dir}'.")
        print("Generating an empty report...")
        generate_html_report([], nightly_version, build_url)
        return

    print(f"Found {len(log_files)} .log file(s) in '{log_dir}': {log_files}")

    all_failed_tests = []
    for log_filename in log_files:
        log_path = os.path.join(log_dir, log_filename)
        tests_from_file = parse_log_file(log_path)
        all_failed_tests.extend(tests_from_file)

    if not all_failed_tests:
        print("\nNo failed test cases were found.")
        print("Generating an empty report...")
        generate_html_report([], nightly_version, build_url)
    else:
        print(
            f"\nIn total, parsed {len(all_failed_tests)} failed test cases from {len(log_files)} log files."
        )
        generate_html_report(all_failed_tests, nightly_version, build_url)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python report_ut_win.py <log_dir>")
        sys.exit(1)

    ut_log_dir = sys.argv[1]
    main(ut_log_dir)
