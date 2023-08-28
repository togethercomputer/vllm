
import argparse
import torch

from vllm import EngineArgs, LLMEngine, SamplingParams
import time
from torch.profiler import profile, record_function, ProfilerActivity

prompt = ""
for i in range(0, 10):
    prompt = prompt + "Hi,"

ngen = 300
batchsize = 1

def main(args: argparse.Namespace):
    
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)

    engine = LLMEngine.from_engine_args(engine_args)

    test_prompts = []
    for i in range(0, batchsize):
        test_prompts.append(("A robot may not injure a human being", SamplingParams(temperature=0.0)))
        engine.add_request(str(i), prompt, SamplingParams(temperature=0.0, max_tokens=300))

    # Run the engine by calling `engine.step()` manually.
    start = time.time()
    steps = 0
    while steps<ngen:

        request_outputs = engine.step()
        steps = steps + 1

        if not (engine.has_unfinished_requests() or test_prompts):
            break
    end = time.time()
    elapsed = end - start

    single_q_throughput = 1.0 * ngen / elapsed
    global_q_throughput = 1.0 * ngen *  len(request_outputs) / elapsed

    print("Batch size", len(request_outputs))
    print("Single Q:", single_q_throughput, "tokens/s")
    print("Global Q:", global_q_throughput, "tokens/s")

    print(request_outputs[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
