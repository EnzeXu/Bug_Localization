from bug_localization.dataset import make_one_dataset


if __name__ == "__main__":
    res = 0
    while not res:
        res = make_one_dataset()