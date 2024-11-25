import argparse

from .bug_report_repo import BugReportRepo
from .utils import get_repo_from_file


def make_one_dataset():
    parser = argparse.ArgumentParser(description='make_dataset')
    parser.add_argument("--repo", type=str, default="robolectric/robolectric")
    args = parser.parse_args()
    br_repo = BugReportRepo(args.repo, silence=False)
    print(f"[Finished make_dataset] {args.repo}")
    return br_repo


def make_all_dataset():
    repo_list = get_repo_from_file("data/repo_list_final.csv")
    print(repo_list)
    # # repo_list = repo_list[:1]
    # repo_list = ["robolectric/robolectric"]  # [IMPORTANT] Please update it !
    status = 0
    print()
    print("$" * 80)
    print("$" * 80)
    print("$" * 80)
    print()
    for i, one_repo in enumerate(repo_list):
        # try:
        br_repo = BugReportRepo(one_repo, silence=False, description=f"[{i + 1:03d}/{len(repo_list):03d}]")
        print(f"[Finished make_dataset]")
        print()
        print("@" * 80)
        print("@" * 80)
        print("@" * 80)
        print()
        status = 1
        # except Exception as e:
        #     print(f"[Error in make_dataset]:", e)
        #     print()
        #     print("=" * 80)
        #     print("=" * 80)
        #     print("=" * 80)
        #     print()
        #     status = 0
    return status
