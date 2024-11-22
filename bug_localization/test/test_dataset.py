import requests
import json
from bs4 import BeautifulSoup


def one_time_test_bs():
    # url = 'https://github.com/apache/ant-ivy/commit/899f9fd0a9690581f33a715bb5f000397f1bafef#diff-9f36586ec0a0ca371556969468518af32b083e39be86746dbefaf8a015da2b85'  # Replace with your actual URL
    url = "https://api.github.com/repos/apache/ant-ivy/commits/899f9fd0a9690581f33a715bb5f000397f1bafef#diff-9f36586ec0a0ca371556969468518af32b083e39be86746dbefaf8a015da2b85"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Try to parse the response as JSON
            commit_data = response.json()
            print(json.dumps(commit_data, indent=4))
            print(commit_data["sha"])
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print("Response text:", len(response.text))  # Print raw response for debugging
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response text:", response.text)  # Print raw response for debugging


if __name__ == "__main__":
    one_time_test_bs()