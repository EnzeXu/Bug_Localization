import os
import pickle

from ._utils import http_get
from .bug_report_base import BugReportBase
from .bug_report_pull_request import BugReportPullRequest

class BugReportIssue(BugReportBase):
    def __init__(self, repo, issue_id: str | int, api_url=None, raw_url=None, content_json=None, silence=True):
        super().__init__(api_url, raw_url, silence)

        self.repo = repo.strip("/")
        self.issue_id = str(issue_id)
        self.content_json = content_json
        self.all_pull_request_url_list = None
        self.pull_request_list = None
        self.body = None
        self.title = None
        self.get_all_pull_request_url_list()
        self.get_issue_body()
        self.get_issue_title()
        self.get_pull_request_list()

    def get_issue_body(self):
        if self.content_json is None:
            return
        if "body" in self.content_json:
            self.body = self.content_json["body"]
        else:
            self.body = ""
            self.available = 0

    def get_issue_title(self):
        if self.content_json is None:
            return
        if "title" in self.content_json:
            self.title = self.content_json["title"]
        else:
            self.title = ""

    def get_pull_request_list(self):
        if self.all_pull_request_url_list is None:
            self.available = 0
            return
        self.pull_request_list = []
        for one_pull_request_url in self.all_pull_request_url_list:
            pull_request = BugReportPullRequest.from_url(one_pull_request_url, silence=self.silence)
            if pull_request.available:
                self.pull_request_list.append(pull_request)

    @classmethod
    def from_url(cls, url: str, silence=True):
        repo, api_url, raw_url, last_id = BugReportBase.get_repo_urls_last_id(url)
        return cls(repo, last_id, api_url, raw_url, silence=silence)

    def get_all_pull_request_url_list(self):
        if not self.content_json:
            issue_url = f"https://api.github.com/repos/{self.repo}/issues/{self.issue_id}"
            status, response = http_get(issue_url, silence=self.silence, save_type="get_issue")
            if not status:
                self.available = 0
                return
            self.content_json = response.json()

        # if "event_url" in self.content_json:
        #     issue_event_url = self.content_json["event_url"]
        #     status, response = http_get(issue_event_url, silence=self.silence)
        self.all_pull_request_url_list = []
        if "pull_request" in self.content_json:
            if "merged_at" in self.content_json["pull_request"] and self.content_json["pull_request"]["merged_at"] is not None:
                direct_pull_request_url_list = [self.content_json["pull_request"]["url"]]
                self.all_pull_request_url_list.extend(direct_pull_request_url_list)
                if not self.silence:
                    self.print(f"from direct: {direct_pull_request_url_list}")

        if "timeline_url" in self.content_json:
            issue_timeline_url = self.content_json["timeline_url"]
            status, response = http_get(issue_timeline_url, silence=self.silence, save_type="get_issue_timeline")
            if status:
                timeline_json = response.json()
                timeline_pull_request_url_list = []
                for one_time in timeline_json:
                    if "source" not in one_time:
                        continue
                    if "issue" not in one_time["source"]:
                        continue
                    if "pull_request" not in one_time["source"]["issue"]:
                        continue
                    if "merged_at" in one_time["source"]["issue"]["pull_request"] and one_time["source"]["issue"]["pull_request"]["merged_at"] is not None:
                        timeline_pull_request_url_list.append(one_time["source"]["issue"]["pull_request"]["url"])
                self.all_pull_request_url_list.extend(timeline_pull_request_url_list)
                if not self.silence:
                    self.print(f"from timeline: {timeline_pull_request_url_list}")
        self.print(f"all_pull_request_url_list: {self.all_pull_request_url_list}")
        if len(self.all_pull_request_url_list) == 0:
            self.available = 0
