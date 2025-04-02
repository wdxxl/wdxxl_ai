
import httpx
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup

# initialize FastMCP server
mcp = FastMCP(
    'Your MCP Tools',
    dependencies=['beautifulsoup4']
)

@mcp.tool(
    name= 'Extract-Web-Page-Content-Tool',
    description='Tool to extract page content in text format'
)
def extract_web_content(url: str) -> str| None:
    try:
        response = httpx.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
            },
            timeout=10.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text().replace('\n', ' ').replace('\r', ' ').strip()
    except Exception as e:
        return f'Error fetching content: {str(e)}'
    
if __name__ == "__main__":
    # Initialized and run the server
    mcp.run(transport='stdio')