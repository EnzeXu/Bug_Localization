import os
import re
import pickle

from ._utils import http_get_multiple_page, http_get
from .bug_report_base import BugReportBase


class BugReportCommit(BugReportBase):
    def __init__(self, repo, commit_hash: str, api_url=None, raw_url=None):
        super().__init__(api_url, raw_url)

        self.repo = repo.strip("/")
        self.commit_hash = commit_hash
        self.available = 1
        self.target_snippets = None
        self.neighbor_snippets = None

    def get_snippets(self):
        self.target_snippets, self.neighbor_snippets = self.get_code_snippet_list()
        if len(self.target_snippets) == 0:
            self.available = 0
        # self.issue_list = self._get_issue_list()

    @classmethod
    def from_url(cls, url: str):
        # url = url.strip("")
        # assert url[:5] == "https" and url.count("/") == 7
        # terms = url.split("/")
        # repo = f"{terms[4]}/{terms[5]}"
        # commit_id = terms[7]
        repo, api_url, raw_url, last_id = BugReportBase.get_repo_urls_last_id(url)
        return cls(repo, last_id, api_url, raw_url)

    def get_code_snippet_list(self):
        # commit_url = "https://github.com/androidx/media/commit/2ac8247cf4a60ac86a516a8c508d2bcbb1202b33"
        # br_commit = BugReportCommit.from_url(commit_url)
        # repo = br_commit.repo
        # commit_hash = br_commit.commit_hash
        # commit_hash = "fe667560d0f67b43a63f82adc15ec205beb49cad"
        diff_url = self.raw_url + '.diff'
        status, response = http_get(diff_url, silence=True)
        diff_text = response.text
        # print(diff_text)
        changed_lines = self.parse_diff(diff_text)

        target_functions_all = []
        neighbor_functions_all = []

        for file_path, lines in changed_lines.items():
            # if file_path != "src/main/java/com/actiontech/dble/config/ServerConfig.java":
            #     continue
            # print(f"file_path: {file_path}")
            if file_path[-5:] != ".java":
                continue
            file_url = f"https://raw.githubusercontent.com/{self.repo}/{self.commit_hash}/{file_path}"
            status, response = http_get(file_url, silence=True)
            java_code = response.text
            # print(java_code)
            function_snippets, neighbor_function_snippets = self.extract_function(java_code, lines, file_path)
            target_functions_all.extend(function_snippets)
            neighbor_functions_all.extend(neighbor_function_snippets)
            # for function in functions:
            #     print(f"Function in {file_path} involving changed lines:")
            #     print(f"'{function}'")
            #     print("\n" + "=" * 80 + "\n")
        return target_functions_all, neighbor_functions_all

    @staticmethod
    def parse_diff(diff_text):
        changed_lines_dic = {}
        file_pattern = re.compile(r'^diff --git a/(.*?) b/.*')
        hunk_pattern = re.compile(r'^@@ -(\d+),\d+ \+(\d+),\d+ @@(.*)')
        # update_pattern = re.compile(r'^[+] .*')
        current_file = None

        # update_dic = dict()
        context_start_list = []
        start_plus_index = None

        for i_line, line in enumerate(diff_text.splitlines()):
            if line[:3] in ["ind", "---", "+++"]:
                continue
            file_match = file_pattern.match(line)
            if file_match:
                if len(context_start_list) > 0:
                    target_lines = BugReportCommit.process_active_lines(context_start_list, start_plus_index)
                    if current_file not in changed_lines_dic:
                        changed_lines_dic[current_file] = target_lines
                        # print(f"match: '{current_file}' at '{target_lines}'")
                    else:
                        changed_lines_dic[current_file].extend(target_lines)
                    # print(f"match: '{current_file}' at '{target_lines}'")
                    # if current_file == "src/main/java/com/actiontech/dble/config/ServerConfig.java":
                    # print(f"match: '{current_file}' at '{target_lines}'")
                context_start_list = []
                current_file = file_match.group(1)
                continue
                # changed_lines[current_file] = []

            hunk_match = hunk_pattern.match(line)
            if hunk_match and current_file:
                if len(context_start_list) > 0:
                    target_lines = BugReportCommit.process_active_lines(context_start_list, start_plus_index)
                    if current_file not in changed_lines_dic:
                        changed_lines_dic[current_file] = target_lines
                        # print(f"match: '{current_file}' at '{target_lines}'")
                    else:
                        changed_lines_dic[current_file].extend(target_lines)
                    # print(f"match: '{current_file}' at '{target_lines}'")
                    # if current_file == "src/main/java/com/actiontech/dble/config/ServerConfig.java":
                    # print(f"match: '{current_file}' at '{target_lines}'")
                context_start_list = []
                start_plus_index = int(hunk_match.group(2))
                continue
                # match_class = str(hunk_match.group(3).strip())
                # if len(match_class) > 0:
                #     if current_file not in changed_lines:
                #         changed_lines[current_file] = [match_class]
                #         print(f"match: '{current_file}' at '{match_class}'")
                #     elif match_class not in changed_lines[current_file]:
                #         changed_lines[current_file].append(match_class)
                #     # if current_file == "src/main/java/com/actiontech/dble/config/ServerConfig.java":
                #         print(f"match: '{current_file}' at '{match_class}'")
            context_start_list.append(line[0])

            # update_match = update_pattern.match(line)
            # if update_match and current_file and len(line) >= 20:
            #     update_line = line[1:]
            #     if current_file not in update_dic:
            #         update_dic[current_file] = [update_line]
            #     else:
            #         update_dic[current_file].append(update_line)
            #     # if current_file == "src/main/java/com/actiontech/dble/config/ServerConfig.java":
            #     print(f"match: '{current_file}' at '{update_line}'")

        return changed_lines_dic

    @staticmethod
    def extract_function(java_code, changed_lines, file_path=None):
        # print(f"changed_lines: {changed_lines}")
        function_snippets = []
        lines = java_code.splitlines()
        function_pattern = re.compile(r'^\s*(public|protected|private).*?\{')
        # pure_lines = [item.strip() for item in lines]
        # start_lines = [pure_lines.index(item) for item in changed_lines]

        neighbor_lines = []

        for line_number in changed_lines:
            for i_start in range(line_number, -1, -1):
                if i_start >= len(lines):
                    continue
                # print(f"i = {i_start} line: '{lines[i_start]}'")
                if function_pattern.match(lines[i_start]) and " class " not in lines[i_start]:
                    n_prefix_space = len(lines[i_start]) - len(lines[i_start].lstrip())
                    function_snippet = []
                    i_end = len(lines) - 1
                    for j in range(i_start, len(lines)):
                        function_snippet.append(lines[j])
                        if lines[j].rstrip() == f"{' ' * n_prefix_space}}}":
                            i_end = j
                            break
                    new_code_snippet = "\n".join(function_snippet)
                    if i_start <= line_number <= i_end and new_code_snippet not in function_snippets:
                        function_snippets.append(new_code_snippet)
                        if i_start - 1 > 0:
                            neighbor_lines.append(i_start - 1)
                        if i_start - 11 > 0:
                            neighbor_lines.append(i_start - 11)
                        if i_start - 21 > 0:
                            neighbor_lines.append(i_start - 21)
                        if i_end + 3 < len(lines):
                            neighbor_lines.append(i_end + 3)
                        if i_end + 13 < len(lines):
                            neighbor_lines.append(i_end + 13)
                        if i_end + 23 < len(lines):
                            neighbor_lines.append(i_end + 23)
                    break

        neighbor_lines = sorted(list(set(neighbor_lines)))
        neighbor_function_snippets = []
        for line_number in neighbor_lines:
            for i_start in range(line_number, -1, -1):
                if i_start >= len(lines):
                    continue
                # print(f"i = {i_start} line: '{lines[i_start]}'")
                if function_pattern.match(lines[i_start]) and " class " not in lines[i_start]:
                    n_prefix_space = len(lines[i_start]) - len(lines[i_start].lstrip())
                    function_snippet = []
                    i_end = len(lines) - 1
                    for j in range(i_start, len(lines)):
                        function_snippet.append(lines[j])
                        if lines[j].rstrip() == f"{' ' * n_prefix_space}}}":
                            i_end = j
                            break
                    new_code_snippet = "\n".join(function_snippet)
                    if i_start <= line_number <= i_end and new_code_snippet not in function_snippets and new_code_snippet not in neighbor_function_snippets:
                        neighbor_function_snippets.append(new_code_snippet)

                    break
        # print(f"file path: {file_path}, target length: {len(function_snippets)}, neighbor length: {len(neighbor_function_snippets)}")
        return function_snippets, neighbor_function_snippets

    @staticmethod
    def process_active_lines(start_list, plus_start):
        res = []
        plus_index = plus_start
        plain_index = plus_start
        for i, item in enumerate(start_list):
            assert item in [" ", "-", "+"], f"item '{item}' is not valid"
            if item == "+":
                plus_index += 1
                res.append(plus_index)
                # print(f"+ at line {plus_index}")
            elif item == "-":
                if plain_index not in res:
                    res.append(plain_index)
                    # print(f"- at line {plain_index}")
                else:
                    # print(f"- at line {plain_index}. skipped")
                    pass
            else:
                plus_index += 1
                plain_index = plus_index
                # print(f"plain at line {plus_index}")
        return res
