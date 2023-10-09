import requests
import jinja2

######## basic definition ###########
headers = {
    'Accept': 'application/vnd.github+json',
}
proxies = {
    'http': 'http://child-prc.intel.com:913',
    'https': 'http://child-prc.intel.com:913',
}
base_url = 'https://api.github.com/search/issues?q=is:open+is:issue+sort:updated-asc+repo:pytorch/pytorch'
#####################################

def specify_lable(lable_filter, url):
    lable_filter = lable_filter.replace(" ","%20")
    return url + f'+label:"{lable_filter}"'

def query_open_issues_with_label(lable_filter):
    query_url = specify_lable(lable_filter, base_url)
    response = requests.get(query_url, headers=headers).json()
    filtered_response = []
    for item in response["items"]:
        filtered_response.append(item)
    return filtered_response

def generate_html(content, file_name):
    loader = jinja2.FileSystemLoader('./template.html')
    env = jinja2.Environment(loader=loader)
    template = env.get_template('')
    items = []
    i = 0;
    for issue in content:
        item = dict(id=str(i),
                    created_time=issue['created_at'][0:10],
                    updated_time=issue['updated_at'][0:10],
                    state=issue['state'],
                    issue_id=issue['number'],
                    author=issue['user']['login'],
                    title=issue['title'],
                    link=issue['html_url'])
        items.append(item)
        i = i + 1
    with open(file_name, "w") as f:
        print(template.render(items=items), file=f)

issue_content =  query_open_issues_with_label("module: cpu inductor")
generate_html(issue_content, "index.html") 