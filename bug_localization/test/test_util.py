from ..dataset import get_repo_from_file

repos = get_repo_from_file(file_path="data/repo_list.csv")
print(f"{len(repos)} repos: {repos}")