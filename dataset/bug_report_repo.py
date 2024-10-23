import os
import pickle

from ._utils import http_get_multiple_page
from .bug_report_base import BugReportBase


class BugReportRepo(BugReportBase):
    def __init__(self, repo):
        super().__init__()

        self.repo = repo.strip("/")
        self.available = 1
        self.issue_list = self._get_issue_list()
        self.issue_num = len(self.issue_list)
        self.issue_id_list = [int(item["number"]) if "number" in item else -1 for item in self.issue_list]

    def _get_issue_list(self):
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
            with open(data_cache_save_path, "rb") as f:
                repo_issue_dic = pickle.load(f)
        else:
            repo_issue_dic = dict()

        if self.repo in repo_issue_dic:
            print(f"[BugReportRepo] Repo {self.repo} exists. Skipped collecting.")
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
                print(f"[BugReportRepo] length of issue list: {len(issue_list)}")
            repo_issue_dic[self.repo] = issue_list
            with open(data_cache_save_path, "wb") as f:
                pickle.dump(repo_issue_dic, f)

        print(f"[BugReportRepo] Number of repos collected: {len(repo_issue_dic)}")
        print(f"[BugReportRepo] Their names:\n{[key for key in repo_issue_dic]}")
        print(f"[BugReportRepo] Their issue lengths:\n{[len(value) for key, value in repo_issue_dic.items()]}")
        return issue_list





if __name__ == "__main__":
    br_repo = BugReportRepo("actiontech/dble")
    print(br_repo)
    pass
