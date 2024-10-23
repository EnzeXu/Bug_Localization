import os
import pickle

from ._utils import http_get_multiple_page
from .bug_report_base import BugReportBase


class BugReportIssue(BugReportBase):
    def __init__(self, repo, issue_id):
        super().__init__()

        self.repo = repo.strip("/")
        self.issue_id = issue_id
        self.available = 1
        # self.issue_list = self._get_issue_list()