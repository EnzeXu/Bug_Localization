import os
import pickle
from tqdm import tqdm

from ._utils import http_get_multiple_page
from .bug_report_base import BugReportBase
from .bug_report_issue import BugReportIssue

class BugReportRepo(BugReportBase):
    def __init__(self, repo, silence=True):
        super().__init__(silence=silence)

        self.repo = repo.strip("/")
        self.api_url = f"https://api.github.com/repos/{self.repo}"
        self.raw_url = f"https://github.com/{self.repo}"
        self.issue_list = self.get_issue_list()
        self.issue_num = len(self.issue_list)
        self.issue_id_list = [int(item["number"]) for item in self.issue_list]
        self.issue_url_list = [item["url"] for item in self.issue_list]
        self.issue_class_list = None
        self.generate()

    def get_issue_list(self):
        issue_list_url = f"https://api.github.com/repos/{self.repo}/issues"
        issue_list_params = {
            "state": "closed",
            "per_page": 100,
        }
        data_cache_save_folder = os.path.join("data", "cache")
        if not os.path.exists(data_cache_save_folder):
            os.makedirs(data_cache_save_folder)
        data_cache_save_path = os.path.join(data_cache_save_folder, "repo_issue_dic.pkl")
        if os.path.exists(data_cache_save_path):
            # self.print(f"existing!")
            with open(data_cache_save_path, "rb") as f:
                repo_issue_dic = pickle.load(f)
        else:
            repo_issue_dic = dict()
        # self.print(f"{self.repo in repo_issue_dic}, {self.repo}, {repo_issue_dic.keys()}")
        if self.repo in repo_issue_dic:
            self.print(f"[BugReportRepo] Repo {self.repo} in repo_issue_dic exists. Skipped collecting.")
            issue_list = repo_issue_dic[self.repo]
            if len(issue_list) == 0:
                self.available = 0
        else:
            status, res = http_get_multiple_page(issue_list_url, issue_list_params)
            if status == 0:
                self.available = 0
                issue_list = []
            else:
                issue_list = list(res)
                self.print(f"[BugReportRepo] length of issue list: {len(issue_list)}")
            repo_issue_dic[self.repo] = issue_list
            with open(data_cache_save_path, "wb") as f:
                self.print(f"[BugReportRepo] saved {self.repo} issue list to: {data_cache_save_path}")
                pickle.dump(repo_issue_dic, f)

        self.print(f"[BugReportRepo] Number of issues collected: {len(repo_issue_dic)}")
        self.print(f"[BugReportRepo] Their names:\n{[key for key in repo_issue_dic]}")
        self.print(f"[BugReportRepo] Their issue lengths:\n{[len(value) for key, value in repo_issue_dic.items()]}")
        return issue_list

    def generate(self):
        data_cache_save_folder = os.path.join("data", "cache")
        data_cache_issue_save_folder = os.path.join("data", "cache_issue", self.repo.replace("/", "@"))
        if not os.path.exists(data_cache_save_folder):
            os.makedirs(data_cache_save_folder)
        data_cache_save_path = os.path.join(data_cache_save_folder, "repo_pair_dic.pkl")
        if os.path.exists(data_cache_save_path):
            with open(data_cache_save_path, "rb") as f:
                repo_pair_dic = pickle.load(f)
        else:
            repo_pair_dic = dict()

        if not os.path.exists(data_cache_issue_save_folder):
            os.makedirs(data_cache_issue_save_folder)

        if self.repo in repo_pair_dic:
            self.print(f"[BugReportRepo Pairing] Repo {self.repo} in repo_pair_dic exists ({len(repo_pair_dic[self.repo])} pairs). Skipped collecting.")
        else:
            data_bug_report_save_folder = os.path.join("data", "bug_report")
            if not os.path.exists(data_bug_report_save_folder):
                os.makedirs(data_bug_report_save_folder)
            data_bug_report_save_path = os.path.join(data_bug_report_save_folder, f"{self.repo.replace('/', '@')}_issue_class_list.pkl")
            if os.path.exists(data_bug_report_save_path):
                with open(data_bug_report_save_path, "rb") as f:
                    self.issue_class_list = pickle.load(f)
            else:
                self.issue_class_list = []
                progress_bar = tqdm(total=len(self.issue_url_list))
                for one_issue_url, one_issue_id in zip(self.issue_url_list, self.issue_id_list):
                    issue_cache_path = f"{data_cache_issue_save_folder}/{one_issue_id:06d}.pkl"
                    if os.path.exists(issue_cache_path):
                        with open(issue_cache_path, "rb") as f:
                            one_issue_class = pickle.load(f)
                    else:
                        one_issue_class = BugReportIssue.from_url(one_issue_url, silence=self.silence)
                        with open(issue_cache_path, "wb") as f:
                            pickle.dump(one_issue_class, f)
                    if one_issue_class.available:
                        self.issue_class_list.append(one_issue_class)
                    self.print(f"########################## repo = {self.repo}, issue id = {one_issue_id}, available = {one_issue_class.available}")
                    progress_bar.update(1)
                progress_bar.close()
                with open(data_bug_report_save_path, "wb") as f:
                    pickle.dump(self.issue_class_list, f)
            self.print(f"[BugReportRepo Pairing] Repo {self.repo} class list generated ({len(self.issue_class_list)} issues).")

            pair_list = []

            positive_count = 0
            negative_count = 0

            for one_br_issue in tqdm(self.issue_class_list):
                if not one_br_issue.available:
                    continue
                for one_pull_request in one_br_issue.pull_request_list:
                    if not one_pull_request.available:
                        continue
                    for one_commit in one_pull_request.commit_list:
                        if not one_commit.available:
                            continue
                        for one_snippet in one_commit.target_snippets:
                            pair_list.append([one_br_issue.body, one_snippet, 1.0])
                            positive_count += 1
                        for one_snippet in one_commit.neighbor_snippets:
                            pair_list.append([one_br_issue.body, one_snippet, 0.0])
                            negative_count += 1
            self.print(f"repo: {self.repo}")
            self.print(f"positive_count: {positive_count}")
            self.print(f"negative_count: {negative_count}")
            repo_pair_dic[self.repo] = pair_list
            with open(data_cache_save_path, "wb") as f:
                pickle.dump(repo_pair_dic, f)

if __name__ == "__main__":
    br_repo = BugReportRepo("actiontech/dble")
    print(br_repo)
    pass
