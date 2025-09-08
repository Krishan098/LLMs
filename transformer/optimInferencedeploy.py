# Optimized Inference Deployment

'''
Applications such as Text Generation Inference, vLLM and llama.cpp are primarily used in production environments to serve LLMs to users.
'''

## Memory Management and Performance

'''
TGI is designed to be stable and predictable in production, using fixed sequence lengths to keep memory usage consistent. TGI manages memory
using Flash attention 2 and continuous batching techniques. This means it can process attention calculations very efficiently and keep the GPU
busy by constantly feeding it work. The system can move parts of the model between CPU and GPU when needed, which helps handle larger model.
'''


'''
Flash attention is a technique that optimizes the attention mechanism in transformer models by addressing memory bandwidth bottlenecks.The attention 
mechanism has quadratic complexity and memory usage, making it inefficient for long sequences.

The key innovation is in how it manages memory transfers between High Bandwidth Memory and faster SRAM cache. Traditional attention repeatedly transfers 
data between High Bandwidth Memory(HBM) and SRAM, creating bottlenecks by leaving the GPU idle. Flash attention loads data once into SRAM and performs all 
calucations there, minimizing expensive memory transfers.
'''

# vLLM

'''
vLLM takes a different approach by using PagedAttention.

vLLM splits the model's memory into smaller blocks.
It can handle different-sized requests more flexibly and doesn't waste memory space.
'''
#PageAttention
'''
PageAttention overcomes another issue that is KV cache management. During text generation, the model stores attention keys and values for each generated token to reduce
redundant computations. This cache though could grow large and be enormous especially for longer sequences or multiple concurrent requests.

vLLM's key innovation lies in how it manages this cache:
1. Memory Paging: Instead of treating the KV cache as one large block, it's divided into fixed-size pages.
2. Non-contiguous storage: Pages don't need to be contiguous in the GPU memory, allowing for more flexible memory allocation.
3. Page Table Management: A page table tracks which pages belong to which sequence, enabling efficient lookup and access.
4. Memory sharing: For operations like parallel sampling, pages storing the KV cache for the prompt can be shared across multiple sequences.
'''

# llama.cpp

'''
it is a higlhy optimized C/C++ implementation. It uses quantization to reduce model size and memory requirements while maintaining good performance. It implements optimized kernels
for various CPU architectures and supports basic KV cache management for efficient token generation.
'''

'''
Quantization reduces the precision of model weights from 32-bit or 16-bit floating point to lower precision formats like 8-bit integers, 4bit or even lower. This significantly reduces memory 
usage and improves inference speed with minimal quality loss.

1. Multiple Quantization levels: Supports 8-bit, 4-bit,3-bit and even 2-bit quantization
2. GGML/GGUF format: uses custom tensor formats optimized for quantized inference
3. Mixed Precision: Can apply different quantization levels to different parts of the model.
4. Hardware-specific optimizations: Includes optimized code paths for various CPU architectures
'''
from huggingface_hub import InferenceClient
client=InferenceClient(model='http://localhost:8080',)
response=client.text_generation("Sing me a song",max_new_tokens=100,temperature=0.7,top_p=0.95,details=True,stop_sequences=[])
#print(response.generated_text)
response=client.chat_completion(
    messages=[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Tell me a story"},
    ], max_tokens=100,temperature=0.7, top_p=0.95,
)
#print(response.choices[0].message.content)

