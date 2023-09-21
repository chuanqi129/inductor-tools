# This script runs on mlt-ace.sh.intel.com server for none-root users.
# Use selenium 3.4.2 + geckodriver 0.33.0 + Mozilla Firefox 114.0
# firefox in: /home2/yudongsi/workspace/firefox/firefox
# geckodriver in: /home2/yudongsi/workspace/geckodriver
# You need to specify the FirefoxBinary and executable_path explicitly to advoid use default outdated one

import argparse
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
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
    approval= '//*[@id="cli_login_button"]/span'
    WebDriverWait(browser, 200).until(EC.element_to_be_clickable((By.XPATH, approval)))   
    sleep(10)       
    browser.find_element(By.XPATH, approval).click()
    # success='//*[@id="LoginForm"]/div/span'
    # WebDriverWait(browser, 60).until(EC.visibility_of_element_located((By.XPATH, success)))
    print("AWS SSO Refresh Done")
    browser.quit()

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
        # intel azure portal login
        print("IAP...")
        iap='//*[@id="i0116"]'
        WebDriverWait(driver, 1200).until(EC.visibility_of_element_located((By.XPATH, iap)))
        driver.find_element(By.XPATH, iap).send_keys(user)
        
        print("Next...")
        next='//*[@id="idSIButton9"]'
        driver.find_element(By.XPATH, next).click()

        print("PSWD...")
        sleep(5)
        pswd='//*[@id="i0118"]'
        WebDriverWait(driver, 1200).until(EC.visibility_of_element_located((By.XPATH, pswd)))
        driver.find_element(By.XPATH, pswd).send_keys(passwd)
        sleep(5)
        signin='//*[@id="idSIButton9"]'
        WebDriverWait(driver, 200).until(EC.element_to_be_clickable((By.XPATH, signin)))      
        driver.find_element(By.XPATH, signin).click()
        authorize_request(driver)
    except:
        authorize_request(driver)

sso_refresh(args.ff,args.gd,args.code,args.user,args.passwd)