# frontend/utils/api_client.py
import httpx
from typing import List, Dict, Any

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
    
    def chat(self, message: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Send chat request to backend and return response."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "message": message,
                    "conversation_history": conversation_history
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "message": f"❌ API Error: {e.response.status_code} - {e.response.text}",
                "retrieved_documents": []
            }
        except Exception as e:
            return {
                "message": f"❌ Connection Error: {str(e)}",
                "retrieved_documents": []
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status from backend."""
        try:
            response = self.client.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}