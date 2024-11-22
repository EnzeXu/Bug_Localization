# from ..dataset import BugReportRepo, BugReportCommit
from ..dataset import http_get, BugReportCommit
import re
import requests

# br_commit = BugReportCommit.from_url("https://github.com/actiontech/dble/pull/3645/commits/c0d5b5f2e5fbab9fd9fd525efa0e338105113400")
br_commit = BugReportCommit.from_url("https://github.com/androidx/media/commit/3396737d06cb6d321d3c797e8db5e2230e44bae4")
br_commit.get_snippets()
print(br_commit)
# target_snippets, neighbor_snippets = br_commit.get_code_snippet_list()
# for i in range(len(target_snippets)):
#     print(f"[target snippet {i + 1}/{len(target_snippets)}]")
#     print(target_snippets[i])

import re

# # Compile the hunk pattern with an additional capture group for the part after @@
# hunk_pattern = re.compile(r'^@@ -(\d+),\d+ \+(\d+),\d+ @@(.*)')
#
# # Example input string
# hunk_line = "@@ -787,16 +787,16 @@ public void reviseLowerCase() {"
#
# # Apply the regex pattern
# match = hunk_pattern.match(hunk_line)
#
# if match:
#     print(match.group(1))
#     print(match.group(2))
#     print(match.group(3))
#     rest_of_line = match.group(3).strip()
#     print(f"Rest of the line: '{rest_of_line}'")


# br_commit = BugReportCommit.from_url("https://github.com/actiontech/dble/pull/3645/commits/c0d5b5f2e5fbab9fd9fd525efa0e338105113400")
# print(br_commit.process_active_lines(
#     [" "] * 3 + ["-"] * 2 + ["+"] * 2 + [" "] * 3 + ["-"] * 5 + ["+"] * 5 + [" "] * 3,
#     787
# ))


# # Function to fetch the diff of a specific commit
# def fetch_diff(commit_url):
#     diff_url = commit_url + '.diff'
#     status, response = http_get(diff_url)
#     return response.text
#     # response = requests.get(diff_url)
#     # if response.status_code == 200:
#     #     return response.text
#     # else:
#     #     raise Exception(f"Failed to fetch diff: {response.status_code}")
#
#
# # Function to parse changed lines from diff
# def parse_diff(diff_text):
#     changed_lines = {}
#     file_pattern = re.compile(r'^diff --git a/(.*?) b/.*')
#     hunk_pattern = re.compile(r'^@@ -(\d+),\d+ \+(\d+),\d+ @@')
#     current_file = None
#
#     for line in diff_text.splitlines():
#         file_match = file_pattern.match(line)
#         if file_match:
#             current_file = file_match.group(1)
#             changed_lines[current_file] = []
#
#         hunk_match = hunk_pattern.match(line)
#         if hunk_match and current_file:
#             start_line = int(hunk_match.group(2))
#             changed_lines[current_file].append(start_line)
#
#     return changed_lines
#
#
# # Function to fetch the full Java file at a specific commit
# def fetch_file_at_commit(repo, file_path, commit_hash):
#     file_url = f"https://raw.githubusercontent.com/{repo}/{commit_hash}/{file_path}"
#     status, response = http_get(file_url)
#     return response.text
#     # response = requests.get(file_url)
#     # if response.status_code == 200:
#     #     return response.text
#     # else:
#     #     raise Exception(f"Failed to fetch file: {response.status_code}")
#
#
# # Function to extract the function surrounding the changed line
# def extract_function(java_code, changed_lines):
#     print(f"changed_lines: {changed_lines}")
#     function_snippets = []
#     lines = java_code.splitlines()
#     function_pattern = re.compile(r'^\s*(public|protected|private).*?\{')
#
#     for line_number in changed_lines:
#         for i in range(line_number + 1, -1, -1):
#             if i >= len(lines):
#                 continue
#             print(f"i = {i} line: '{lines[i]}'")
#             if function_pattern.match(lines[i]):
#                 function_snippet = []
#                 for j in range(i, len(lines)):
#                     function_snippet.append(lines[j])
#                     if lines[j].strip() == "}":
#                         break
#                 function_snippets.append("\n".join(function_snippet))
#                 break
#     return function_snippets
#
#
#
#
# if __name__ == "__main__":
#     # Example usage
#     commit_url = "https://github.com/actiontech/dble/pull/3645/commits/c0d5b5f2e5fbab9fd9fd525efa0e338105113400"
#     br_commit = BugReportCommit.from_url(commit_url)
#     repo = br_commit.repo
#     commit_hash = br_commit.commit_hash
#     # commit_hash = "fe667560d0f67b43a63f82adc15ec205beb49cad"
#
#     diff_text = fetch_diff(br_commit.raw_url)
#     # print(diff_text)
#     changed_lines = parse_diff(diff_text)
#
#     for file_path, lines in changed_lines.items():
#         print(f"file_path: {file_path} lines: {lines}")
#         if file_path[-5:] != ".java":
#             continue
#         java_code = fetch_file_at_commit(repo, file_path, commit_hash)
#         # print(java_code)
#         functions = extract_function(java_code, lines)
#         for function in functions:
#             print(f"Function in {file_path} involving changed lines:")
#             print(f"'{function}'")
#             print("\n" + "=" * 80 + "\n")
#
#
#
#     # br_repo = BugReportRepo("actiontech/dble")
#     # print(br_repo)
#     # br_commit = BugReportCommit.from_url(
#     #     "https://api.github.com/repos/google/ExoPlayer/commits/f54284080231bb5a65487a3709506569ffb609ca")
#     # print(br_commit)

