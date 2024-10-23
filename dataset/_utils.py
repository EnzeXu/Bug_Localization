import requests
import time

from .PAT import TOKEN


def http_get(url, params=None, headers=None, repeat_threshold=5, silence=False):
    if not params:
        params = {}
    if not headers:
        headers = {"Authorization": f"token {TOKEN}"}
    else:
        headers["Authorization"] = f"token {TOKEN}"
    response = requests.get(url, params=params, headers=headers)
    if not silence:
        print(f"[http_get] {response.url}")

    if response.status_code != 200:
        print(f"[http_get] Failed to retrieve info from: {response.url} (status_code: {response.status_code})")
        again = 1
        while again <= repeat_threshold:
            response = requests.get(url, params=params)
            print(f"[http_get_multiple_page] Try again ({again}/{repeat_threshold}) status_code: {response.status_code}")
            if response.status_code == 200:
                break
            time.sleep(0.1 * again)
            again += 1

        if response.status_code != 200:
            return 0, None
    res = response
    status = 1
    return status, res


def http_get_multiple_page(url, params=None, headers=None, repeat_threshold=5, silence=False):
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
        response = requests.get(url, params=params, headers=headers)
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

