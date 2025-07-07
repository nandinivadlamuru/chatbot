from typing import Any, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionSemanticFAQ(Action):
    def name(self) -> str:
        return "action_semantic_faq"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]):
        query = tracker.latest_message.get("text")
        try:
            response = requests.post(
                "http://localhost:8000/query-haystack",
                json={"question": query}
                                        )
            if response.status_code == 200:
                answer = response.json().get("answer")
                if answer:
                    dispatcher.utter_message(text=answer)
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't find an answer.")
            else:
                dispatcher.utter_message(text="Error reaching the AI backend.")
        except Exception as e:
            dispatcher.utter_message(text=f"LangChain error: {str(e)}")

        return []
