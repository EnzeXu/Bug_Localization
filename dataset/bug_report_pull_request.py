import os
import re
import pickle
from tqdm import tqdm

from ._utils import http_get_multiple_page, http_get
from .bug_report_base import BugReportBase
from .bug_report_commit import BugReportCommit


class BugReportPullRequest(BugReportBase):
    def __init__(self, repo, pull_number: str | int, api_url=None, raw_url=None, silence=True):
        super().__init__(api_url, raw_url, silence)

        self.repo = repo.strip("/")
        self.pull_number = str(pull_number)
        self.commit_list = None
        self.num_target_snippets = None
        self.num_neighbor_snippets = None
        self.title = None
        self.body = None
        self.get_commits()
        self.get_pull_request_info()

        # self.message = None


    # def set_message(self, message):
    #     self.message = message


    @classmethod
    def from_url(cls, url: str, silence=True):
        repo, api_url, raw_url, last_id = BugReportBase.get_repo_urls_last_id(url)
        return cls(repo, last_id, api_url, raw_url, silence=silence)

    def get_commits(self):
        commits_request_url = self.api_url + '/commits'
        status, response = http_get(commits_request_url, silence=self.silence, save_type="get_commits")
        if not status:
            self.available = 0
            return
        response = response.json()
        # self.print(f"len response: {len(response)}")
        self.commit_list = []
        # iterator = tqdm(response) if not self.silence else response
        for one_commit_item in response:
            if "url" not in one_commit_item:
                continue
            commit_url = one_commit_item["url"]
            commit_message = one_commit_item["commit"]["message"].strip()
            one_commit = BugReportCommit.from_url(commit_url, silence=self.silence)
            # one_commit.get_snippets()
            one_commit.set_message(commit_message)
            if one_commit.available:
                self.commit_list.append(one_commit)
        self.num_target_snippets = sum([one_commit.num_target_snippets for one_commit in self.commit_list])
        self.num_neighbor_snippets = sum([one_commit.num_neighbor_snippets for one_commit in self.commit_list])
        if self.num_target_snippets == 0:
            self.available = 0
        if not self.silence:
            self.print(f"[{self.repo}] {self.api_url}: [{self.num_target_snippets} vs. {self.num_neighbor_snippets}]")

    def get_pull_request_info(self):
        pull_request_url = self.api_url
        status, response = http_get(pull_request_url, silence=self.silence, save_type="get_pull")
        if not status:
            self.available = 0
            return
        response = response.json()
        if "body" not in response or "title" not in response:
            self.available = 0
        else:
            self.title = response["title"]
            self.body = response["body"]



