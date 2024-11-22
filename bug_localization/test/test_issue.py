from ..dataset import BugReportRepo, BugReportCommit
from ..dataset import http_get, BugReportIssue
import re
import requests

br_issue = BugReportIssue.from_url("https://api.github.com/repos/actiontech/dble/issues/3884")
br_issue.get_all_pull_request()
print(br_issue)



