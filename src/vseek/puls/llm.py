from openai import OpenAI
import datetime
import json
import os


class LLM:
    def __init__(
        self,
        model="gpt-4o",
        history=None,
        openai_api_key=None,
        save_dir="/nas/mars/experiment_result/nsvqa/9_post_submission/llm_conversation_history/"
        ): # change save_dir as needed
        """Initialize LLM"""
        self.client = OpenAI()
        self.model = model
        if history:
            self.history = history
        else:
            self.history = []
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def prompt(self, p):
        """Send a prompt to the LM and update conversation history"""
        user_message = {"role": "user", "content": [{"type": "text", "text": p}]}
        self.history = []
        self.history.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            store=False,
        )
        assistant_response = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}

        return assistant_response
    
    def clear_history(self):
        self.history = []
