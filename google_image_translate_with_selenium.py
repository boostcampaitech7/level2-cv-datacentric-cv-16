from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from pyvirtualdisplay import Display
import os
from create_annotation import UFOAnnotationWithEasyocr
from tqdm import tqdm

import shutil

# 기본 세팅 방법
# https://github.com/password123456/setup-selenium-with-chrome-driver-on-ubuntu_debian?tab=readme-ov-file#step-6-create-hello_world

# Selenium python document

if __name__=="__main__":
    
    ## 번역 옵션들
    translate_opt = {'text': 'translate',
                     'images' : 'images',
                     'docs' : 'docs'}
    lang_opt = {'english' : 'en',
                'korean' : 'ko',
                'vietnamese' : 'vi',
                'thai' : 'th',
                'japanese' : 'ja',
                'chinese' : 'zh-CN'}
    
    ## 구글 번역기
    _from = 'japanese'
    _to = 'vietnamese'
    _with = 'images'

    img_folder = 'data/japanese_receipt/img/train'
    
    ## 기본 경로 세팅 
    dataset_name = 'ja2vi'
    
    data_root = f'data/{dataset_name}_receipt'
    
    img_path = os.path.join(data_root, 'img')
    img_train_path = os.path.join(img_path, 'train')
    
    ufo_path = os.path.join(data_root, 'ufo')
    
    download_path = '/data/ephemeral/home/Downloads'
    
    ## Custom Dataset 폴더 생성 & rename
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(ufo_path, exist_ok=True)
    
    # 가상 디스플레이 시작
    display = Display(visible=1, size=(1920, 1080), backend="xvfb")
    display.start()
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('user-data-dir=C:\\User Data')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("disable-gpu")   # 가속 사용 x
    options.add_argument("lang=ko_KR")    # 가짜 플러그인 탑재
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # driver.get("https://python.org")
    # print(driver.title)
    
    url = f"https://translate.google.co.kr/?sl={lang_opt[_from]}&tl={lang_opt[_to]}&op={translate_opt[_with]}"

    driver.get(url)

    for path in tqdm(sorted(os.listdir(img_folder))):
        abs_path = os.path.abspath(os.path.join(img_folder, path))
        
        file_input = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[4]/c-wiz/div[2]/c-wiz/div/div/div/div[1]/div[2]/div[2]/div[1]/input')
        file_input.send_keys(abs_path)
        
        time.sleep(2)
        
        try:
            download_btn = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[4]/c-wiz/div[2]/c-wiz/div/div[1]/div[2]/div[2]/button'))
            )
            download_btn.click()
        except Exception as e:
            print("다운로드 버튼을 찾지 못했습니다.", e)        
        finally:
            time.sleep(2)
        
        try:
            rm_curr_btn = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[4]/c-wiz/div[2]/c-wiz/div/div[1]/div[2]/span[3]/button'))
            )
            rm_curr_btn.click()
        except Exception as e:
            print("이미지 제거 버튼을 찾지 못했습니다.", e)
        finally:
            time.sleep(2)
        
    driver.close()
    display.stop()
    
    shutil.move(download_path, img_path)
    os.rename(os.path.join(img_path, 'Downloads'), img_train_path)
    
    for path in os.listdir(img_train_path):
        new_name = path.split('.')
        new_name[1] = dataset_name
        os.rename(os.path.join(img_train_path, path), os.path.join(img_train_path, ".".join(new_name)))
    
    UFOAnnotator = UFOAnnotationWithEasyocr(img_train_path, os.path.join(ufo_path, 'train.json'), ['en', 'ko'])
    UFOAnnotator.create_annotation()