from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_path = 'D:/KPMG-POC/weights/llama-2-7b-chat.Q5_K_M.gguf'


class Loadllm:
    @staticmethod
    def load_llm():
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # Prepare the LLM

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=1024,
            n_ctx=4096,
            f16_kv=True, 
            callback_manager=callback_manager,
            verbose=True,
        )

        return llm