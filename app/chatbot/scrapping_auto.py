from app.chatbot.chatbot_utils import CreateRetriever

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import time
import numpy as np
import pandas as pd
import requests
import os
import regex as re

def check_link(link, driver):
    driver.get(link)

    try:
        element_present = EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'hd-banner-p404')]"))
        WebDriverWait(driver, 1).until(element_present)
        return -1
    except:
        return 1
    
def get_feature(driver, feature_names):
    feature_values = []
    try:
        feature_boxes = driver.find_elements(By.XPATH, "//div[@class='property' and .//div[contains(@class, 'drug-list')] and .//div[contains(@class, 'drug-detail')]]")
        features = {feature.find_element(By.XPATH, ".//div[@class='drug-list']").text: 
                    feature.find_element(By.XPATH, ".//div[@class='drug-detail']").text 
                    for feature in feature_boxes}
    except:
        features = {}

    try:
        image_feature = driver.find_element(By.XPATH, "//img[@class='product-image']").get_attribute("src")
    except:
        image_feature = "No Info"

    feature_values.append(image_feature)

    for feature_name in feature_names:
        if feature_name != "Link Gambar":
            if feature_name in features.keys():
                feature_values.append(features[feature_name])
            else:
                feature_values.append("No Info")

    return feature_values

def create_new_drug_df(drug_names, features, old_drug_df, driver):
    drug_links = [re.sub(r"&", "dan", drug_name.lower()) for drug_name in drug_names]
    drug_links = [re.sub(r"((?<=-)-|^-|-$)", "", re.sub(r"[^A-Za-z0-9]", "-", drug_link.lower())) for drug_link in drug_links]
    drug_links = [f"https://www.halodoc.com/obat-dan-vitamin/{drug_link}" for drug_link in drug_links]

    df = {"Nama Obat": drug_names,
          "Link Obat": drug_links,
          "Check": [0 for i in range(len(drug_names))]}
    
    for feature in features:
        df[feature] = ["No Info" for i in range(len(drug_names))]

    df = pd.DataFrame(df)
    print(len(df))

    try:
        for i in df[df["Check"] == 0].index:
                if i % 10 == 0:
                    print(i)
                    print("hi")
                df.loc[i, "Check"] = check_link(df.loc[i, "Link Obat"], driver)
                if df.loc[i, "Check"] != -1:
                    df.loc[i, features] = get_feature(driver, features)

        driver.close()
    except:
        print("ehe")
        df.loc[i-1, "Check"] = 0
        df.loc[i-1, features] = "No Info"
        new_df = pd.concat([old_drug_df, df]).reset_index(drop=True)

        return new_df

    new_df = pd.concat([old_drug_df, df]).reset_index(drop=True)
    new_df.dropna(axis=0, how="any")

    return new_df

if __name__ == "__main__":

    scrapping_df = pd.read_csv("scrapping_obat_final.csv")
    all_drugs = scrapping_df["Nama Obat"].to_list()

    link = "https://www.halodoc.com/obat-dan-vitamin/kategori/obat-dan-perawatan"

    button_xpath = "//button[@class='custom-container__pagination--btn']"
    drug_name_xpath = "//p[@class='hd-base-product-search-card__title']"

    drug_names = []
    len_prev = len(drug_names)
    count = 0

    button_exist = True

    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')

    driver = webdriver.Firefox(options=options)
    driver.get(link)

    while button_exist:
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, drug_name_xpath)))

        drug_names_element = driver.find_elements(By.XPATH, drug_name_xpath)
        drug_names += [drug_name.text for drug_name in drug_names_element]
        drug_names = list(set(drug_names))
        element_present = EC.presence_of_element_located((By.XPATH, button_xpath))
        
        try:
            next_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
            next_button.click()
            time.sleep(2)
        except:
            button_exist = False

        if len_prev == len(drug_names):
            count += 1
        
        len_prev = len(drug_names)
        print(len_prev)

        if count == 5:
            button_exist = False

    driver.close()
    driver = webdriver.Firefox(options=options)

    new_drugs = [drug_name for drug_name in drug_names if drug_name not in all_drugs]

    features = scrapping_df.drop(columns=["Nama Obat", "Link Obat", "Check"]).columns.to_list()

    new_df = create_new_drug_df(new_drugs, features, scrapping_df, driver)

    new_df.to_csv("scrapping_auto_df.csv", index=False)