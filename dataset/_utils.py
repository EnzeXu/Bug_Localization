import requests
import time
import os
import pickle
import pandas as pd
import datetime
import pytz
import re
from urllib.parse import urlencode

from .PAT import TOKEN


def extract_repo_name(url):
    # Pattern for raw.githubusercontent, api.github, and github URLs
    patterns = [
        r'/raw\.githubusercontent\.com/([^/]+)/([^/]+)/',  # For raw.githubusercontent.com
        r'/api\.github\.com/repos/([^/]+)/([^/]+)/',  # For api.github.com
        r'/github\.com/([^/]+)/([^/]+)/'  # For github.com (commits, pulls, etc.)
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            owner = match.group(1)  # The owner of the repo
            repo = match.group(2)  # The repo name
            return f"{owner}@{repo}"
    print(f"Error is extract_repo_name on {url}")
    return "unknown@repo"
    # return None  # If no match found

def load_http_get(repo_name):
    cache_folder = "data/cache_http_get"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cache_folder_path = os.path.join(cache_folder, f"{repo_name}.pkl")
    if os.path.exists(cache_folder_path):
        with open(cache_folder_path, "rb") as f:
            dic = pickle.load(f)
    else:
        dic = dict()
    return dic


def save_http_get(repo_name, dic):
    cache_folder = "data/cache_http_get"
    cache_folder_path = os.path.join(cache_folder, f"{repo_name}.pkl")
    with open(cache_folder_path, "wb") as f:
        pickle.dump(dic, f)


def http_get(url, params=None, headers=None, repeat_threshold=2, silence=False, timeout=5):
    repo_name = extract_repo_name(url)
    if not params:
        params = {}
    status = 1
    http_get_dic = load_http_get(repo_name)  # http_get_dic
    query_string = str(urlencode(params))  # http_get_dic
    full_url = f"{url}?{query_string}"  # http_get_dic
    if full_url in http_get_dic:  # http_get_dic
        if not silence:
            print(f"key '{full_url}' found in http_get_dic")
        response = http_get_dic[full_url]  # http_get_dic
        if response is None:
            status = 0
        else:
            status = 1
        return status, response  # http_get_dic

    else:
        if not headers:
            headers = {"Authorization": f"token {TOKEN}"}
        else:
            headers["Authorization"] = f"token {TOKEN}"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.Timeout:
            print(f"[http_get] Error: TIMEOUT")
            status = 0
            response = None
            # return 0, None
        except requests.RequestException as e:
            print(f"[http_get]Error:", e)
            status = 0
            response = None
            # return 0, None
        if status:
            if not silence:
                print(f"[http_get] {response.url}")

            if response.status_code != 200:
                print(f"[http_get] Failed to retrieve info from: {response.url} (status_code: {response.status_code})")
                again = 1
                while again <= repeat_threshold:
                    response = requests.get(url, params=params)
                    print(f"[http_get] Try again ({again}/{repeat_threshold}) status_code: {response.status_code}")
                    if response.status_code == 200:
                        break
                    time.sleep(0.1 * again)
                    again += 1

                if response.status_code != 200:
                    status = 0
                    response = None
                    # return 0, None

    http_get_dic[full_url] = response  # http_get_dic
    save_http_get(repo_name, http_get_dic)  # http_get_dic
    if not silence:  # http_get_dic
        print(f"key '{full_url}' saved to http_get_dic")  # http_get_dic
    return status, response


def http_get_multiple_page(url, params=None, headers=None, repeat_threshold=2, silence=False, timeout=5):
    res_list = []
    page_id = 1
    if not params:
        params = {}
    if not headers:
        headers = {"Authorization": f"token {TOKEN}"}
    else:
        headers["Authorization"] = f"token {TOKEN}"
    while url:
        params["page"] = page_id
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if not silence:
            print(f"[http_get] {response.url}")

        # Check if the response is successful
        if response.status_code != 200:
            print(f"[http_get_multiple_page] Failed to retrieve info from: {response.url} (status_code: {response.status_code})")
            again = 1
            while again <= repeat_threshold:
                response = requests.get(url, params=params)
                print(f"[http_get_multiple_page] Try again ({again}/{repeat_threshold}) status_code: {response.status_code}")
                if response.status_code == 200:
                    break
                time.sleep(0.1 * again)
                again += 1

            if response.status_code != 200:
                return 0, []

        # Add the current page of issues to the list
        current_list = response.json()
        if len(current_list) == 0:
            url = None
        if not silence:
            print(f"get {len(current_list)} items from {response.url}")
        res_list.extend(current_list)
        #
        # # Check for the 'Link' header to get the URL for the next page
        # link_header = response.headers.get('Link')
        # if link_header:
        #     links = {rel[6:-1]: url[1:-1] for url, rel in (link.split(';') for link in link_header.split(','))}
        #     url = links.get('next')  # Update the URL to the next page
        # else:
        #     url = None  # No more pages left
        # time.sleep(0.1)
        page_id += 1

    return 1, res_list


def get_repo_from_file(file_path, total_issue_threshold=1000, closed_issue_threshold=500):
    df = pd.read_csv(file_path)
    filtered_df = df[(df['totalIssues'] >= total_issue_threshold) & ((df['totalIssues'] - df['openIssues']) >= closed_issue_threshold)]
    return sorted(list(set(filtered_df["name"])))


def get_now_string(time_string="%Y%m%d_%H%M%S_%f"):
    est = pytz.timezone('America/New_York')
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)

    # Return the time in the desired format
    return est_now.strftime(time_string)