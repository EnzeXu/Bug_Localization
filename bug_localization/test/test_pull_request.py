from ..dataset import BugReportRepo, BugReportCommit
from ..dataset import http_get, BugReportCommit, BugReportPullRequest
import re
import requests

br_pull_request = BugReportPullRequest.from_url("https://github.com/actiontech/dble/pull/3645")
br_pull_request.get_commits()
print(br_pull_request)



