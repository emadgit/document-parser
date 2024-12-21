<!-- image -->

# How to fine-tune open LLMs in 2025 with Hugging Face

Large Language Models (LLMs) continued their important role in 2024, with several major developments completely outperforming previous models. The focus continued to more smaller, more powerful models from companies like Meta, Qwen, or Google. These models not only became more powerful, but also more efficient. We got Llama models as small as 1B parameters outperforming Llama 2 13B.

LLMs can now handle many tasks out-of-the-box through prompting, including chatbots, question answering, and summarization. However, for specialized applications requiring high accuracy or domain expertise, fine-tuning remains a powerful approach to achieve higher quality results than prompting alone, reduce costs by training smaller, more efficient models, and ensure reliability and consistency for specific use cases.

Contrary to last years guide How to Fine-Tune LLMs in 2024 with Hugging Face this guide focuses more on optimization, distributed training and being more customizable. This means support for different PEFT methods from Full-Finetuning to QLoRA and Spectrum, optimizations for faster and more efficient training, with Flash Attention or Liger Kernels and how to scale training to multiple GPUs using DeepSpeed.

This guide is created using a script rather than notebook. If you are compeltely new to fine-tuning LLMs, I recommend you to start with the How to Fine-Tune LLMs in 2024 with Hugging Face guide and then come back to this guide.

You will learn how to:

1. Define a good use case for fine-tuning
2. Setup the development environment
3. Create and prepare the dataset
4. Fine-tune the model using trl and the SFTTrainer with QLoRA as example
5. Test and evaluate the model using GSM8K

What is Qlora?

QLoRA (Quantized Low-Rank Adaptation) enables efficient fine-tuning of LLMs using 4-bit quantization and minimal parameter updates, reducing resource needs but potentially impacting performance due to quantization trade-offs.

What is Spectrum?

Spectrum is a fine-tuning method that identifies the most informative layers of a LLM using Signal-to-Noise Ratio (SNR) analysis and selectively fine-tunes them, offering performance comparable to full fine-tuning with reduced resource usage, especially in distributed training setups.

Note: This guide is designed for consumer GPUs (24GB+) like the NVIDIA RTX 4090/5090 or A10G, but can be adapted for larger systems.

## 1. Define a good use case for fine-tuning

Open LLMs became more powerful and smaller in 2024. This often could mean fine-tuning might not be the first choice to solve your problem. Before you think about fine-tuning, you should always evaluate if prompting or already fine-tuned models can solve your problem. Create an evaluation setup and compare the performance of existing open models.

However, fine-tuning can be particularly valuable in several scenarios. When you need to:

- Consistently improve performance on a specific set of tasks
- Control the style and format of model outputs (e.g., enforcing a company's tone of voice)
- Teach the model domain-specific knowledge or terminology
- Reduce hallucinations for critical applications
- Optimize for latency by creating smaller, specialized models
- Ensure consistent adherence to specific guidelines or constraints

As an example, we are going to use the following use case:

We want to fine-tune a model, which can solve high-school math problems to teach students how to solve math problems.

This can be a good use case for fine-tuning, as it requires a lot of domain-specific knowledge about math and how to solve math problems.

Note: This is a made-up example, as existing open models already can solve this task.

## 2. Setup development environment

Our first step is to install Hugging Face Libraries and Pyroch, including trl, transformers and datasets. If you haven't heard of trl yet, don't worry. It is a new library on top of transformers and datasets, which makes it easier to fine-tune, rlhf, align open LLMs.

<!-- image -->

We will use the Hugging Face Hub as a remote model versioning service. This means we will automatically push our model, logs and information to the Hub during training. You must register on the Hugging Face for this. After you have an account, we will use the login util from the huggingface\_hub package to log into our account and store our token (access key) on the disk.

<!-- image -->

## 3. Create and prepare the dataset

Once you've determined that fine-tuning is the right solution, you'll need a dataset. Most datasets are now created using automated synthetic workflows with LLMs, though several approaches exist:

- Synthetic Generation with LLMs: Most common approach using frameworks like Distilabel to generate high-quality synthetic data at scale
- Existing Datasets: Using public datasets from Hugging Face Hub
- Human Annotation: For highest quality but most expensive option

The LLM Datasets provides an overview of high-quality datasets to fine-tune LLMs for all kind of purposes. For our example, we'll use Orca-Math dataset including 200,000 Math world problems.

Modern fine-tuning frameworks like trl support standard formats:

<!-- image -->

Note: If you are interested in a guide on how to create high-quality datasets, let me know.

To prepare our datasets we will use the Datasets library and then convert it into the the conversational format, where we include the schema definition in the system message for our assistant. We'll then save the dataset as jsonl file, which we can then use to fine-tune our model.

Note: This step can be different for your use case. For example, if you have already a dataset from, e.g. working with OpenAI, you can skip this step and go directly to the fine-tuning step.

<!-- image -->

## 4. Fine-tune the model using trl and the SFTTrainer with QLoRA

We are now ready to fine-tune our model. We will use the SFTTrainer from trl to fine-tune our model. The SFTTrainer makes it straightfoward to supervise fine-tune open LLMs. The SFTTrainer is a subclass of the Trainer from the transformers library and supports all the same features, including logging, evaluation, and checkpointing, but adds additiional quality of life features, including:

- Dataset formatting, including conversational and instruction format
- Training on completions only, ignoring prompts
- Packing datasets for more efficient training
- PEFT (parameter-efficient fine-tuning) support including Q-LoRA, or Spectrum
- Preparing the model and tokenizer for conversational fine-tuning (e.g. adding special tokens)
- distributed training with accelerate and FSDP/DeepSpeed

We prepared a run\_sft.py scripts, which supports providing a yaml configuration file to run the fine-tuning. This allows you to easily change the model, dataset, hyperparameters, and other settings. This is done by using the TrlParser, which parses the yaml file and converts it into the TrainingArguments arguments. That way we can support Q-LoRA, Spectrum, and other PEFT methods with the same script. See Appendix A for execution examples for different models and PEFT methods and distributed training.

Question: Why don't we use frameworks like axolotl?

That's a great question! Axolotl is a fantastic framework, it is used by many open source builders and is well tested. However, it is good to know how to do things manually. This will give you a better understanding of the inner workings and how it can be customized. Especially when you ran into an issue or want to extend the scripts and add new features.

Before we can start our training lets take a look at our training script. The script is kept very simple and is easy to understand. This should help you understand, customize and extend the script for your own use case. We define dataclasses for our arguments. Every argument can then be provided either via the command line or by providing a yaml configuration file. That way we have better type safety and intellisense support.

<!-- image -->

We can customize behavior for different training methods and use them in our script with script\_args. The training script is separated by ####### blocks for the different parts of the script. The main training function:

1. Logs all hyperperparameters
2. Loads the dataset from Hugging Face Hub or local disk
3. Loads the tokenizer and model with our training strategy (e.g. Q-LoRA, Spectrum)
4. Initializes the SFTTrainer
5. Starts the training loop (optionally continue training from a checkpoint)
6. Saves the model and optionally pushes it to the Hugging Face Hub

Below is an example recipe of how we can fine-tune a Llama-3.1-8B model with Q-LoRA.

<!-- image -->

This config works for single-GPU training and for multi-GPU training with DeepSpeed (see Appendix for full command). If you want to use Spectrum check the Appendix for more information.

<!-- image -->

I ran several experiments with different optimization strategies, including Flash Attention, Liger Kernels, Q-Lora, and the Spectrum method to compare the time it takes to fine-tune a model. The results are summarized in the following table:

| Model        | Train samples   | Hardware   | Method         |   train sequence length |   per device batch size |    gradient accumulation | packing   |  Flash Attention   | Liger Kernels   |   est. optimization steps | est. train time   |
|--------------|-----------------|------------|----------------|-------------------------|-------------------------|--------------------------|-----------|--------------------|-----------------|---------------------------|-------------------|
| Llama-3.1-8B | 10,000          | 1x L4 24GB | Q-LoRA         |                    1024 |                       1 |                        2 | ❌         | ❌                  | ❌               |                      5000 | ~360 min          |
| Llama-3.1-8B | 10,000          | 1x L4 24GB | Q-LoRA         |                    1024 |                       2 |                        2 | ✅         | ❌                  | ❌               |                      1352 | ~290 min          |
| Llama-3.1-8B | 10,000          | 1x L4 24GB | Q-LoRA         |                    1024 |                       2 |                        4 | ✅         | ✅                  | ❌               |                       676 | ~220 min          |
| Llama-3.1-8B | 10,000          | 1x L4 24GB | Q-LoRA         |                    1024 |                       4 |                        4 | ✅         | ✅                  | ✅               |                       338 | ~135 min          |
| Llama-3.1-8B | 10,000          | 4x L4 24GB | Q-LoRA         |                    1024 |                       8 |                        2 | ✅         | ✅                  | ✅               |                        84 | ~33 min           |
| Llama-3.1-8B | 10,000          | 8x L4 24GB | Q-LoRA         |                    1024 |                       8 |                        2 | ✅         | ✅                  | ✅               |                        42 | ~18 min           |
| Llama-3.1-8B | 10,000          | 8x L4 24GB | Spectrum (30%) |                    1024 |                       8 |                        2 | ✅         | ✅                  | ✅               |                        42 | ~21 min           |

Notes:

- Q-Lora included training the embedding layer and the lm\_head, as we use the Llama 3.1 chat template and in the base model the special tokens are not trained.
- For distributed training Deepspeed (0.15.4) with ZeRO3 and Hugging Face Accelerate was used.
- Spectrum with 30% SNR layers took slightly longer than Q-Lora, but achieves 58% accuracy on GSM8K dataset, which is 4% higher than Q-Lora.

Using Q-LoRA only saves the trained adapter weights. If you want to use the model as standalone model, e.g. for inference you might want to merge the adapter and base model. This can be done using the following command:

<!-- image -->

## 5. Test Model and run Inference

After the training is done we want to evaluate and test our model. As we trained our model on solving math problems, we will evaluate the model on GSM8K dataset. GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

Evaluating Generative AI models is not a trivial task since 1 input can have multiple correct outputs. If you want to learn more about evaluating generative models, check out:

- Evaluate LLMs and RAG a practical example using Langchain and Hugging Face.
- Evaluate LLMs using Evaluation Harness and Hugging Face TGI/vLLM
- LLM Evaluation doesn't need to be complicated
- Evaluating Open LLMs with MixEval: The Closest Benchmark to LMSYS Chatbot Arena

We are going to use Evaluation Harness an open-source framework to evaluate language models on a wide range of tasks and benchmarks. The frameworks support evaluating models behind OpenAI compatible API endpoints, those can be locally or remotely. This super helpful as we can evaluate our model in the same environment we will use for production.

We are going to use Text Generation Inference (TGI) for testing and deploying our model. TGI is a purpose-built solution for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation using Tensor Parallelism and continous batching. If you are or want to use vLLM you can check the Appendix on how to start the inference server.

Note: Make sure that you have enough GPU memory to run the container. Restart kernel to remove all allocated GPU memory from the notebook.

We will start the on 1 GPU detached. Meaning we can can continue to use the notebook while the container is running. If you have more GPUs you can change the --gpus and --num-shard flags to the number of GPUs.

<!-- image -->

Our container will now start in the background and download the model from Hugging Face Hub. We can check the logs to see the progress with docker logs -f tgi.

Once our container is running we can send requests using the openai or huggingface\_hub sdk. Here we ll use the openai sdk to send a request to our inference server. If you don't have the openai sdk installed you can install it using pip install openai.

<!-- image -->

Awesome that looks great! Now we can evaluate our model with the Evaluation Harness.

Note: Make sure to change the model id to your fine-tuned model.

<!-- image -->

Wow, 54% accuracy with only using 10k samples is pretty good! We successfully validated that our model can solve math problems. Now, don't forget to stop your container once you are done.

<!-- image -->

## Conclusion

This guide provides the foundation for fine-tuning LLMs in 2025. The modular training scripts and configurations make it easy to adapt to your specific use case, whether you're training on a single GPU or scaling across multiple nodes.

If you encounter issues, have questions, or want to contribute improvements to the training pipeline, please open a PR on the repository.

# Appendix

The Appendix contains additional commands and documentation on how to run distributed training, inference and how to use Spectrum.

## Distributed Training

### Deepspeed + Q-LoRA

Note: change the num\_processes to the number of GPUs you want to use.

<!-- image -->

## Inference

### vLLM

Note: Replace the model id with your fine-tuned model.

<!-- image -->

## Spectrum

Spectrum uses Signal-to-Noise Ratio (SNR) analysis to select the most useful layers for fine-tuning. It provides scripts and pre-run scanned for different models. If your model isn't scanned it will prompt you for the batch size to use for scanning. Batch size of 4 for 70b models requires 8xH100. But popular models like Llama 3.1 8B are already scanned. You can find the scanned models here.

The script will generate a yaml configuration file in the model\_snr\_results with the name of the model and the top-percent, e.g. for meta-llama/Llama-3.1-8B and 30 it will generate it at snr\_results\_meta-llama-Meta-Llama-3.1-8B\_unfrozenparameters\_30percent.yaml.

- --model-name: Specify the local model path or the Hugging Face repository.
- --top-percent: Specify the top percentage of SNR layers you want to retrieve.

<!-- image -->

After the yaml configuration is generated we can use it to fine-tune our model. We need to define the yaml configuration file in our train config yaml file and provide the path to the yaml file as spectrum\_config\_path. Take a look at receipes/llama-3-1-8b-spectrum.yaml for an example.

Then we can start the training with the following command for single GPU training:

<!-- image -->

Note: Spectrum requires a more memory than Q-Lora. According to the paper ~30-50GB on a single GPU.

For multi-GPU training with FSDP and Deepspeed you can use the following command:

<!-- image -->

Note: Training on 8x L4 GPUs with Spectrum takes ~21 minutes. Q-Lora on the same config took 18 minutes.

Results:

- Spectrum model trained for 1 epoch with 30% SNR layers on GSM8K dataset achieved 58% accuracy, which is 4% higher than Q-Lora.
- Spectrum model trained for 3 epochs with 30% SNR layers on GSM8K dataset achieved 60% accuracy.