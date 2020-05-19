from selenium import webdriver
import time
from selenium.webdriver.firefox.options import Options
import re
import urllib.request
import os


captcha_home_path = os.path.abspath(__file__ + "/../../")

url = ""
options = Options()
options.headless = True
driver = webdriver.Firefox(firefox_options=options,
                           executable_path= captcha_home_path + '/geckodriver-v0.26.0-win64/geckodriver.exe')

# download 500 images
for n in range(500):
    driver.get(url)
    time.sleep(5)
    htmlSource = driver.page_source
    htmlSource2 = driver.execute_script("return document.body.innerHTML;")

    link = re.findall('src="([^"]*)".*', htmlSource2)[0]  # first match
    jpg_number = str(n).zfill(4)
    urllib.request.urlretrieve(link, captcha_home_path + '/images/' + jpg_number + '.jpg')  # download images
    print('Downloaded image', jpg_number, '.jpg')

    driver.refresh()  # refresh page
    # driver.execute_script("location.reload()")

driver.quit()