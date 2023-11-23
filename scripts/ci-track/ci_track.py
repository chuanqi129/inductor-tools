from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Max number of per_page equals 100
url = "https://hud.pytorch.org/hud/pytorch/pytorch/main/1?per_page=100&name_filter=inductor_torchbench_cpu"

head=f'<!DOCTYPE html> \
<html> \
<head> \
    <title>Track Report</title> \
</head> \
<body> \
    <p><h3> \
        CI JOBS Failure Track Report \
        <a href={url}> \
            HUD \
        </a> \
    </p></h3> \
    <h4>Failures Tracked :</h4> \
    <table border="1"> \
        <tr> \
            <th>Time</th> \
            <th>Commit</th> \
            <th>Author</th> \
        </tr> \
'

tail='</table><h4>Thanks.</h4></body></html>'

def get_pr_url(commit):
    pr_number =commit.split("#")[1].split(")")[0].strip()
    pr_url="https://github.com/pytorch/pytorch/pull/"+pr_number
    return pr_url

def web_refresh(p_url,output):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage') 
    driver = webdriver.Chrome(options=chrome_options)

    driver.get(p_url)
    # use last() method instead of specific number due to we don't konw SevBox whether exist.
    xp = '//*[@id="__next"]/div[last()]/div/table'
    pos = 1
    # make sure page render completed
    WebDriverWait(driver, 1200).until(EC.visibility_of_element_located((By.XPATH, xp)))
    # get jobheader name index which could be used to locate ci_job_index [Page PR status you see just repsets this ci job conclusion]
    jobheader_elements = driver.find_elements(By.CLASS_NAME, 'hud_jobHeaderName__AEbaQ')
    for jobheader_element in jobheader_elements:
        if jobheader_element.text == "inductor / linux-jammy-cpu-py3.8-gcc11-inductor / test (inductor_torchbench_cpu_accuracy, 1, 1, linux.4xlarge)":
            break
        else:
            pos+=1
    real_index = pos+5
    print("real_index: ", real_index)
    for index in range(1,101):
        element = driver.find_element(By.XPATH,f'//*[@id="__next"]/div[last()]/div/table/tbody/tr[{index}]/td[{real_index}]/div/div/span/span')
        if element.text == "X":
            time = driver.find_element(By.XPATH,f'//*[@id="__next"]/div[last()]/div/table/tbody/tr[{index}]/td[1]/span')
            commit = driver.find_element(By.XPATH,f'//*[@id="__next"]/div[last()]/div/table/tbody/tr[{index}]/td[3]/div/a')
            author = driver.find_element(By.XPATH,f'//*[@id="__next"]/div[last()]/div/table/tbody/tr[{index}]/td[5]/div/a')
            print(f"time: {time.text}, commit: {commit.text},author: {author.text}")
            pr_url=get_pr_url(commit.text)
            output+=f'<tr><td><p style="text-align:center">{time.text}</p></td><td><a href="{pr_url}">{commit.text}</a></td><td>{author.text}</td></tr>'
    driver.quit()
    return output

if __name__=='__main__':
    data = " "
    result=web_refresh(url,data)
    if result.isspace():
        print("CI JOBS FAILURES NOT FOUND")
    else:
        with open("CI_JOB_FAILURE_TRACK.html",mode = "a") as f:
            f.write(head+result+tail)
        f.close()
