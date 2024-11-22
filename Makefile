.PHONY: make_one_dataset
.PHONY: test_commit
.PHONY: test_dataset
.PHONY: test_issue
.PHONY: test_pull_request
.PHONY: test_util

make_one_dataset:
	python -u make_one_dataset.py --repo cabaletta/baritone

test_commit:
	python -m bug_localization.test.test_commit

test_dataset:
	python -m bug_localization.test.test_dataset

test_issue:
	python -m bug_localization.test.test_issue

test_pull_request:
	python -m bug_localization.test.test_pull_request

test_util:
	python -m bug_localization.test.test_util
