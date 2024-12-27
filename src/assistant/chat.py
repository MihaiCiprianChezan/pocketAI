from threading import Thread
from transformers import TextIteratorStreamer
from app_logger import AppLogger


class ChatManager:
    BASE_SYSTEM_PROMPT = (
        "You are Opti, a friendly AI assistant equipped with tools to handle specialized queries efficiently.\n"
        "Always respond in clear, plain unformatted text using natural, conversational human language.\n"
        "You must NOT use in outputs: markdown, headings, bold, bullet points, numbered lists, or any formatting in your responses.\n"
        "Respond in a way that's clear and conversational, just as if you were speaking directly to someone in live chat.\n"
    )

    def __init__(self, model_manager, tool_manager, history_manager):
        self.logger = AppLogger()
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.history_manager = history_manager

    @staticmethod
    def preprocess_messages(history):
        chat_messages = []
        for message in history:
            chat_messages.append(message)
        chat_messages.append({"role": "system", "content": ChatManager.BASE_SYSTEM_PROMPT})
        return chat_messages

    def get_kwargs(self, model_inputs, streamer, max_length=150, top_p=0.8, temperature=0.7):
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.2,
            "eos_token_id": self.model_manager.tokenizer.convert_tokens_to_ids("<|user|>")
        }

    def predict(self, history, max_length=150, top_p=0.8, temperature=0.7):
        self.history_manager.validate(history)
        chat_messages = self.preprocess_messages(history)

        applied_template = self.model_manager.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=self.tool_manager.tools_summary,
        )
        self.logger.debug(f"[ChatManager] Message template: `{applied_template}`")

        model_inputs = self.model_manager.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            tools = self.tool_manager.tools_summary,
        ).to(self.model_manager.device)
        # self.logger.debug(f"[ChatManager] Model inputs: `{model_inputs}`")
        streamer = TextIteratorStreamer(self.model_manager.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = self.get_kwargs(
            model_inputs,
            streamer,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature)

        t = Thread(target=lambda: self.model_manager.model.generate(**generate_kwargs))
        t.start()
        response_parts = []
        for new_token in streamer:
            response_parts.append(new_token)
            yield "".join(response_parts)
        t.join()
        self.logger.debug(f"[ChatManager] Complete answer: `{response_parts}`")