import os
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_with_gemini(price_data: dict) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    You are a market analyst.
    Here is the market data:

    Item: {price_data['item']}
    Current Price: {price_data['current_price']}
    7-Day Price History: {json.dumps(price_data['price_history'], indent=2)}

    Provide insights:
    - Is the price increasing, decreasing, or stable?
    - What could be the reasons for this trend?
    - Give advice for sellers and buyers.
    """
    response = model.generate_content(prompt)
    return getattr(response, "text", "") or ""

def get_item_price(item_name: str) -> dict:
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)
    url = "https://vegetablemarketprice.com/market/kerala/today"
    scraped_data = None

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)

        xpath_query = f"//tr[td[contains(., '{item_name}')]]"
        item_row = wait.until(EC.presence_of_element_located((By.XPATH, xpath_query)))

        cells = item_row.find_elements(By.TAG_NAME, 'td')
        current_price = cells[2].text.strip()

        driver.execute_script("arguments[0].click();", item_row)

        history_row_xpath = f"{xpath_query}/following-sibling::tr[1]"
        history_row_element = wait.until(
            EC.visibility_of_element_located((By.XPATH, history_row_xpath))
        )

        soup = BeautifulSoup(history_row_element.get_attribute('innerHTML'), 'html.parser')

        price_history = []
        table = soup.find('table')
        if table:
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) == 2:
                    date = cols[0].text.strip()
                    price = cols[1].text.strip()
                    price_history.append({'date': date, 'price': price})
        else:
            for row in soup.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) == 2:
                    date = cols[0].text.strip()
                    price = cols[1].text.strip()
                    price_history.append({'date': date, 'price': price})

        scraped_data = {
            'item': item_name,
            'current_price': current_price,
            'price_history': price_history
        }

    except Exception as e:
        print(f"Error while scraping '{item_name}': {e}")
    finally:
        driver.quit()

    return scraped_data
