import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_df_RSS():
    feeds = ['https://smartliquidity.info/feed/', 'https://finance.yahoo.com/news/rssindex',
             'https://blog.buyucoin.com/feed/', 'https://cointelegraph.com/rss/tag/altcoin',
             'https://cryptopotato.com/feed/', 'https://cointelegraph.com/rss/category/top-10-cryptocurrencies',
             'https://cointelegraph.com/rss/tag/regulation', 'https://cointelegraph.com/rss',
             'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml',
             'https://u.today/rss', 'https://coinpedia.org/feed/']
    output = []

    for url in feeds:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'xml')

        for entry in soup.find_all('item'):
            item = {'Title': entry.find('title').text, 'Link': entry.find('link').text}
            output.append(item)

    df = pd.DataFrame(output)
    print(df)
    return df
