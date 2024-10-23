class BugReportBase:
    def __init__(self, api_url=None, raw_url=None):
        self.api_url = api_url
        self.raw_url = raw_url
        self.available = 1

    def list_instance_variables(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)}

    @staticmethod
    def decorate_type(value):
        if isinstance(value, list):
            return f"list (len = {len(value)})"
        else:
            return type(value)

    def __str__(self):
        res = "[BugReportRepo]:\n" + "\n".join([f"{key}: {value}" if isinstance(value, (int, float, str)) else f'{key}: {BugReportBase.decorate_type(value)}' for key, value in self.list_instance_variables().items()])
        return res

    @staticmethod
    def get_repo_urls_last_id(url):
        url = url.strip()
        assert int("/api.github.com/" in url) + int("/github.com/" in url) == 1
        assert url[:5] == "https"
        if "/api.github.com/" in url:
            api_url = url

            terms = url.split("/")
            repo = f"{terms[4]}/{terms[5]}"

            raw_url = api_url.replace(f"/api.github.com/repos/{repo}/", f"/github.com/{repo}/").replace(f"{repo}/commits", f"{repo}/commit")
            last_id = terms[-1].split(".")[0]
        else:
            raw_url = url
            terms = url.split("/")
            repo = f"{terms[3]}/{terms[4]}"
            api_url = raw_url.replace(f"/github.com/{repo}/", f"/api.github.com/repos/{repo}/").replace(f"{repo}/commit", f"{repo}/commits")
            last_id = terms[-1].split(".")[0]
        return repo, api_url, raw_url, last_id
