import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o", max_history_tokens: int = 3000):
        # Fallback to env variable if api_key isn't provided directly
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.conversation_history = []

        # Middleware Configurations
        self.max_history_tokens = max_history_tokens
        # Use a cheaper, faster model specifically for compaction to save costs
        self.summary_client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.summary_model = "gpt-4o-mini"

    def set_system_context(self, data_summary: dict):
        """
        Called once after CSV is loaded.
        Injects the data summary as the system prompt —
        every subsequent message shares this context.
        """
        system_prompt = f"""You are a data analyst assistant. 
The user has uploaded a CSV dataset with the following profile:

{json.dumps(data_summary, indent=2)}

Use this profile to answer questions, detect anomalies, identify trends, 
and suggest ML models. Be specific — reference actual column names, 
values, and statistics from the profile above. 
When returning structured data (anomalies, trends, model suggestions), 
respond in JSON only, with no extra prose."""

        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

    def _estimate_tokens(self, text: str) -> int:
        """Helper middleware method to safely estimate character-to-token count."""
        return len(text) // 4

    def _run_summarization_middleware(self):
        """
        Intercepts history before LLM calls to prevent context overflow.
        Compresses middle conversational bulk while preserving System Context and 
        the last 4 structural messages intact.
        """
        # Calculate approximate active tokens in history
        total_tokens = sum(self._estimate_tokens(
            m["content"]) for m in self.conversation_history)

        # Trigger compaction only if threshold is reached and we have history to compress
        if total_tokens < self.max_history_tokens or len(self.conversation_history) <= 6:
            return

        # Isolate critical structures: Keep System (index 0) and the 4 most recent interactions
        system_message = self.conversation_history[0]
        messages_to_keep = self.conversation_history[-4:]
        messages_to_summarize = self.conversation_history[1:-4]

        # Structure middleware instruction
        summary_prompt = (
            "Progressively summarize the conversation history below. Concurrently distill "
            "the core user analytical queries and discoveries. Avoid verbose system details."
        )

        try:
            # Call a cheaper model for the middleware task to save costs
            response = self.summary_client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": json.dumps(
                        messages_to_summarize)}
                ],
                temperature=0.2
            )

            summary_content = response.choices[0].message.content

            # Reconstruct history: [System Context] -> [Compressed Memory Block] -> [Recent Context]
            self.conversation_history = [
                system_message,
                {"role": "system", "content": f"Summary of previous data conversation: {summary_content}"}
            ] + messages_to_keep

        except Exception as e:
            print(
                f"Middleware Compaction Failed: {e}. Falling back to default execution.")

    def chat(self, user_message: str) -> str:
        """Send a message, get a response, and append both to history."""
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )

        # Execute summarization pipeline checking before generating response
        self._run_summarization_middleware()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.3,
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        return assistant_message

    def chat_json(self, user_message: str) -> dict:
        """Like chat(), but forces JSON output and parses it."""
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )

        # Execute summarization pipeline checking before generating response
        self._run_summarization_middleware()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        return json.loads(assistant_message)

    def reset(self):
        """Clear history — useful when a new CSV is loaded."""
        self.conversation_history = []
