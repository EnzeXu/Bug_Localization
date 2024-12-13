import os
import shutil
import re
import json
import subprocess
from tqdm import tqdm
import random
import datetime
import pytz


def extract_and_rename_java_files(repo_url, destination_folder, index, index_dic, progress_bar=None):
    tmp_folder = './tmp'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(tmp_folder, repo_name)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    # print(f"Cloning the repository {repo_url} into {tmp_folder}...")
    # subprocess.run(["git", "clone", repo_url, repo_path])
    try:
        subprocess.run(
            ["git", "clone", repo_url, repo_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )
    except subprocess.TimeoutExpired:
        print(f"Error: Cloning the repository {repo_url} took too long (more than 10 seconds).")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        return index

    java_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for java_file in java_files:
        new_filename = f"{index:08d}.java"
        destination_path = os.path.join(destination_folder, new_filename)
        # shutil.copy(java_file, destination_path)
        try:
            num_of_class, num_of_func = process_java_file(java_file, destination_path)
        except Exception as e:
            print(f"Errors in file \"{java_file}\"", e)
            num_of_class, num_of_func = 0, 0
        if num_of_class > 0:
            index_dic["class"][str(index)] = num_of_class
            index_dic["func"][str(index)] = num_of_func
            index_dic["source"][str(index)] = java_file[6:]
            if not progress_bar:
                print(f"index: {index:08d} class: {num_of_class:02d} func: {num_of_func:03d} filename: {java_file[6:]}")
            else:
                progress_bar.set_description(f"index: {index:08d} class: {num_of_class:02d} func: {num_of_func:03d} filename: {java_file[6:]}")
                print(f"index: {index:08d} class: {num_of_class:02d} func: {num_of_func:03d} filename: {java_file[6:]}")
            index += 1


    shutil.rmtree(repo_path)
    return index


def process_java_file(java_file, destination_path):
    num_of_class = 0
    num_of_func = 0

    with open(java_file, "r") as f:
        content = f.read()

    content = re.sub(r'//.*?(\n|$)', '', content)  # remove single-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # remove multi-line comments

    class_match = re.search(r'public class\s+\w+', content)

    if class_match:
        # get the position of the class and find the matching closing brace "}"
        class_start = class_match.start()
        class_end = find_class_end(content, class_start)
        content = content[class_start:class_end]
        num_of_class = 1
    else:
        # if no "public class" is found, clear the content
        content = ""

    # symbols = [",", ".", "}", "{", ";", "(", ")", "[", "]", "+", "-", "*", "/", "<", ">", "=", "@", "#", "~", "&"]
    symbols = [".", ",", ";", "@", "#", ":", "(", ")", "[", "]", "{", "}", ">", "<"]
    # for symbol in symbols:
    #     content = content.replace(symbol, f' {symbol} ')

    # content = re.sub(r'\s+', ' ', content).strip()

    # count number of functions
    function_pattern = r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{'
    num_of_func = len(re.findall(function_pattern, content))

    with open(destination_path, 'w') as f:
        f.write(f"{java_file.replace('./tmp/', '')}\n")
        f.write(content)

    return num_of_class, num_of_func


def find_class_end(content, class_start):
    brace_count = 0
    for i in range(class_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return i + 1
    return len(content)


def build_dataset_files(raw_repo_names, start=None, end=None):
    if isinstance(raw_repo_names, str):
        repo_name_list = raw_repo_names.split()
    else:
        assert isinstance(raw_repo_names, list)
        repo_name_list = raw_repo_names
    if start and end:
        repo_name_list = repo_name_list[start:end]
    index_save_path = "index.json"

    if os.path.exists(index_save_path):
        with open(index_save_path, "r") as f:
            index_dic = json.load(f)
        index = index_dic["index"]
        print(f"Loaded {index_save_path}: index = {index}")
    else:
        index = 0
        index_dic = dict()
        index_dic["class"] = dict()
        index_dic["func"] = dict()
        index_dic["source"] = dict()
        index_dic["repo_list"] = []

    with tqdm(total=len(repo_name_list)) as progress_bar:
        # for one_repo_name in tqdm(repo_name_list):
        for one_repo_name in repo_name_list:
            if one_repo_name in index_dic["repo_list"]:
                print(f"skip repo {one_repo_name}")
            else:
                repo_url = f"https://github.com/{one_repo_name}"
                index = extract_and_rename_java_files(repo_url, f"traverse/{one_repo_name.replace('/', '@')}", index, index_dic, progress_bar)
            progress_bar.update(1)
            index_dic["repo_list"].append(one_repo_name)
    index_dic["index"] = index

    sum_class = sum([index_dic["class"][str(i)] for i in range(index) if str(i) in index_dic["class"]])
    sum_func = sum([index_dic["func"][str(i)] for i in range(index) if str(i) in index_dic["func"]])

    index_dic["sum_class"] = sum_class
    index_dic["sum_func"] = sum_func

    print(f"sum_class: {sum_class}, sum_func: {sum_func}")


    with open(index_save_path, "w") as f:
        json.dump(index_dic, f, indent=4)

    print(f"Index file saved to {index_save_path}")


# def traverse_one_repo(repo: str):
#     repo = repo.strip("/")
#     repo_save = repo.replace("/", "@")

if __name__ == "__main__":
    build_dataset_files(["robolectric/robolectric"])

