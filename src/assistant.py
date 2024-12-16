from threading import Thread, Lock

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.quantization import quantize_dynamic

class ChatAssistant:
    def __init__(self, model_name="THUDM/glm-edge-1.5b-chat", device_map="auto"):
        # Initialize lock
        self.history_lock = Lock()

        # device_map = infer_auto_device_map(self.model, max_memory={0: "12GB", 1: "12GB"})

        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading model to device: ", self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(self.device)
        print("Optimizing model for inference...")
        self.model = torch.compile(self.model, mode="max-autotune")

        # Pre-warm the model to initialize CUDA kernels
        dummy_input = self.tokenizer("Warm-up round!", return_tensors="pt").to(self.device)
        self.model.generate(**dummy_input)

    @staticmethod
    def preprocess_messages(history):
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
            "content": {
                "You are Opti, a friendly AI assistant. "
                "Provide concise, natural responses. "
                "Limit your responses in less than 256 characters when possible. "
                "Use a warm, engaging, live chat tone. "
                "Respond in continuous paragraphs, avoid lists and tables, avoid markup or formatting."
                "Do not salute or greet the user (consider salutations have already been done before)."
            }
        })

        return messages

    # def predict(self, history, max_length=100, top_p=0.9, temperature=0.8):
    def predict(self, history, max_length=150, top_p=0.8, temperature=0.7):
        messages = self.preprocess_messages(history)

        # Tokenize inputs
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(self.device)

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

    def get_response(self, history, message, return_iter=True, lang="en"):
        """
        Get the entire response at once
        response = self.get_response(history, "Hello!")
        print(response) > Final output as a single string

        Get the response incrementally as an iterator
        response_stream = self.get_response(history, "Hello!", return_iter=True)
        for partial_response in response_stream:
        print(partial_response)  >  intermediate updates as they are generated
        """
        if lang:
            # TODO: handle multilanguage prompts ...
            pass

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
