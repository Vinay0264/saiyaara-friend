from ddgs import DDGS
from bs4 import BeautifulSoup

def _clean(text: str) -> str:
    """Clean raw snippet text using BeautifulSoup."""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)
    return " ".join(cleaned.split())

def search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for: {query}"
        lines = ["Search results:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['href']}")
            lines.append(f"    {_clean(r['body'])}\n")  # cleaned here
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {str(e)}"

if __name__ == "__main__":
    print(search_web("current weather Visakhapatnam"))