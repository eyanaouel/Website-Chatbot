import os

import asyncio

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

async def crawl_and_save():
    # Configure a 2-level deep crawl
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=3, 
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun("https://www.tek-up.de", config=config)

        print(f"Crawled {len(results)} pages in total")

        # # Access individual results
        # for result in results[:3]:  # Show first 3 results
        #     print(f"URL: {result.url}")
        #     print(f"Depth: {result.metadata.get('depth', 0)}")

        print(result.markdown for result in results)
        
    print('Document creation ...')
    documents = [r.markdown for r in results if r.success and r.markdown]
    
    print('Chunking ...')
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    docs = splitter.create_documents(documents)

    print('VDB creation...')
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vdb = FAISS.from_documents(docs, embeddings)

    os.makedirs("vdb", exist_ok=True)
    vdb.save_local("vdb")
    print("FAISS DB saved to ./vdb")

if __name__ == "__main__":
    asyncio.run(crawl_and_save())

