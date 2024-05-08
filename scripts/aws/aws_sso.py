# This script runs on mlt-ace.sh.intel.com server for none-root users.
# Use selenium 3.4.2 + geckodriver 0.33.0 + Mozilla Firefox 114.0
# firefox in: /home2/yudongsi/workspace/firefox/firefox
# geckodriver in: /home2/yudongsi/workspace/geckodriver
# You need to specify the FirefoxBinary and executable_path explicitly to advoid use default outdated one

import argparse
import sys
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

parser = argparse.ArgumentParser(description="aws sso")
parser.add_argument('-f','--ff',type=str,help='firefox path')
parser.add_argument('-d','--gd',type=str,help='geckodrive path')
parser.add_argument('-c','--code',type=str,help='enter code')
parser.add_argument('-u','--user',type=str,help='IAP mail address')
parser.add_argument('-p','--passwd',type=str,help='IAP password')
args=parser.parse_args()

url = "https://device.sso.us-west-2.amazonaws.com/?user_code="

def authorize_request(browser):
    # aws login approval
    print("APPROV...")
    sleep(10)
    approval= '//*[@data-testid="allow-access-button"]/span'
    get_element_refresh(browser, approval, "click", 5)
    # success='//*[@id="LoginForm"]/div/span'
    # WebDriverWait(browser, 60).until(EC.visibility_of_element_located((By.XPATH, success)))
    print("AWS SSO Refresh Done")
    browser.quit()

def get_element_refresh(driver, ele_reg, op, max_try, param=""):
    try_num = 0
    sleep_time = 10
    while(True):
        sleep(sleep_time)
        try:
            elem = driver.find_element(By.XPATH, ele_reg)
        except NoSuchElementException as e:
            print(e)
            driver.refresh()
            print("refreshing...")
            sleep_time += 10
            try_num += 1
            if (try_num == max_try):
                print("Timeout...")
                sys.exit(2)
        else:
            print("Get the element!")
            if (op == "send_keys"):
                WebDriverWait(driver, 200).until(EC.visibility_of_element_located((By.XPATH, ele_reg)))
                elem.send_keys(param)
            elif (op == "click"):
                WebDriverWait(driver, 200).until(EC.element_to_be_clickable((By.XPATH, ele_reg)))
                elem.click()
            else:
                print("op not support!!")
            #getattr(elem, op)(param)
            break

def sso_refresh(ff,gd,code,user,passwd):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')        
    ff_binary = FirefoxBinary(ff)
    driver = webdriver.Firefox(firefox_options=options,firefox_binary=ff_binary,executable_path=gd)
    print(url+code)
    driver.get(url+code)
    try:
        # cli verification for enter code [new]
        print("cli_verification_btn...")
        cvb= '//*[@id="cli_verification_btn"]'
        get_element_refresh(driver, cvb, "click", 5)
        # intel azure portal login
        print("IAP...")
        iap='//*[@id="i0116"]'
        get_element_refresh(driver, iap, "send_keys", 10, user)

        print("Next...")
        nxt='//*[@id="idSIButton9"]'
        get_element_refresh(driver, nxt, "click", 5)
        #driver.find_element(By.XPATH, next).click()

        print("PSWD...")
        sleep(5)
        pswd='//*[@id="i0118"]'
        get_element_refresh(driver, pswd, "send_keys", 10, passwd)

        print("SIGNIN...")
        sleep(5)
        signin='//*[@id="idSIButton9"]'
        get_element_refresh(driver, signin, "click", 5)

        authorize_request(driver)
    except:
        print("May has exception occur!!")
        authorize_request(driver)

sso_refresh(args.ff,args.gd,args.code,args.user,args.passwd)
