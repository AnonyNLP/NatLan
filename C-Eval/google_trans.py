import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def google_translate(text, api_key, source="zh-cn", target="en"):
    '''
    Related Work: Google-MT
    '''
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'q': text,
        'source': source,
        'target': target,
        'format': 'text',
        'key': api_key
    }
    retries = 20
    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status_forcelist=[500, 502, 503, 504],
        backoff_factor=0.3
    )
    
    # Mount it with a session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        response = session.get(url, params=params)
        response.raise_for_status()  # Will raise an exception for HTTP error codes
        
        return response.json()['data']['translations'][0]['translatedText']