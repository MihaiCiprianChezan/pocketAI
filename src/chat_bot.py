from threading import Thread, Lock
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


class ChatBot:
    def __init__(self, model_name="THUDM/glm-edge-1.5b-chat", device_map="auto"):
        # Initialize lock
        self.history_lock = Lock()
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)

    def preprocess_messages(self, history):
        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        messages.append({
            "role": "system",
            "content": "You are Opti, a female-voiced AI assistant (based on ChatGLM（智谱清言）by Zhipu AI Company and Tsinghua University KEG Lab). Provide brief, natural language responses, concise and no longer than 256 characters. Your responses should be like a natural live spoken chat with the user. You communicate naturally, avoiding numbered lists, and maintains a friendly and engaging tone in all interactions."
        })

        return messages

    def predict(self, history, max_length=100, top_p=0.9, temperature=0.8):
        messages = self.preprocess_messages(history)

        # Tokenize inputs
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.2,
            "eos_token_id": self.tokenizer.convert_tokens_to_ids("<|user|>")
        }

        def generate_in_thread():
            try:
                self.model.generate(**generate_kwargs)
            except Exception as e:
                print(f"Error during generation: {e}")

        # Start generation in a thread
        t = Thread(target=generate_in_thread)
        t.start()

        response = ""
        for new_token in streamer:
            if new_token:
                with self.history_lock:
                    history[-1][1] += new_token
                response += new_token
            yield history

        t.join()
        return response

    def chat(self, history, message, return_iter=False):
        """
        Get the entire response at once
        response = self.chat(history, "Hello!")
        print(response) > Final output as a single string

        Get the response incrementally as an iterator
        response_stream = self.chat(history, "Hello!", return_iter=True)
        for partial_response in response_stream:
        print(partial_response)  >  intermediate updates as they are generated
        """
        if not history:
            history = []
        history.append([message, ""])  # Append user message with empty reply
        # Full response generation
        response = ""

        def response_iterator():
            nonlocal response  # Allow updating the outer `response` variable
            for updated_history in self.predict(history):
                # Cache the entire response
                response = updated_history[-1][1]
                yield response

        if return_iter:
            return response_iterator()  # Return an iterator
        else:
            # If `return_iterator` is False, just compute the full response
            for _ in response_iterator():
                pass
            return response
