import os
import time

import openai
import torch
import transformers
from prompts import identity
from transformers import AutoTokenizer


def get_model(args):
    model_name, temperature = args.model, args.temperature
    backend = getattr(args, "backend", "openai")
    if backend == "hf" or "llama" in model_name.lower():
        return LLaMA(model_name, temperature)
    if backend == "openai":
        return GPT(
            model_name=model_name,
            temperature=temperature,
            api_key=getattr(args, "api_key", ""),
            base_url=getattr(args, "base_url", ""),
            max_retries=getattr(args, "max_retries", 6),
            request_timeout=getattr(args, "request_timeout", 120.0),
        )
    raise ValueError(f"Unsupported backend: {backend}")


class Model(object):
    def __init__(self):
        self.post_process_fn = identity

    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn


class GPT(Model):
    def __init__(self, model_name, temperature, api_key="", base_url="", max_retries=6, request_timeout=120.0):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        self.badrequest_count = 0
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        # vLLM OpenAI-compatible endpoint, e.g. http://127.0.0.1:8001/v1
        self.base_url = base_url if base_url else None
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_response(self, **kwargs):
        attempt = kwargs.pop("_attempt", 0)
        try:
            res = self.client.chat.completions.create(timeout=self.request_timeout, **kwargs)
            return res
        except openai.APIConnectionError:
            print("APIConnectionError")
            if attempt >= self.max_retries:
                return None
            time.sleep(5)
            return self.get_response(_attempt=attempt + 1, **kwargs)
        except openai.RateLimitError as e:
            print("RateLimitError")
            if attempt >= self.max_retries:
                return None
            time.sleep(10)
            return self.get_response(_attempt=attempt + 1, **kwargs)
        except openai.APITimeoutError as e:
            print("APITimeoutError")
            if attempt >= self.max_retries:
                return None
            time.sleep(10)
            return self.get_response(_attempt=attempt + 1, **kwargs)
        except openai.BadRequestError as e:
            print("BadRequestError")
            self.badrequest_count += 1
            print("badrequest_count", self.badrequest_count)
            return None
        except Exception as e:
            print(f"Unexpected API error: {e}")
            if attempt >= self.max_retries:
                return None
            time.sleep(5)
            return self.get_response(_attempt=attempt + 1, **kwargs)

    def forward(self, head, prompts):
        messages = [{"role": "system", "content": head}]
        info = {}
        for i, prompt in enumerate(prompts):
            messages.append({"role": "user", "content": prompt})
            response = self.get_response(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

            if response is None:
                info["response"] = None
                info["message"] = None
                return None, info
            else:

                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                usage = response.usage.model_dump() if response.usage is not None else {}
                info = dict(usage)  # completion_tokens, prompt_tokens, total_tokens
                info["response"] = messages[-1]["content"]
                info["message"] = messages
                # print("response: ", info['response'])
                return self.post_process_fn(info["response"]), info


class LLaMA(Model):
    def __init__(self, model_name, temperature):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=tokenizer,
            temperature=temperature,
        )

    def forward(self, head, prompts):
        prompt = prompts[0]
        sequences = self.pipeline(
            prompt,
            do_sample=False,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = sequences[0]["generated_text"]  # str
        info = {"message": prompt, "response": response}
        return self.post_process_fn(info["response"]), info
