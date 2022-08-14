---
title: 'Training Memory-Intensive Deep Learning Models with PyTorch’s Distributed Data Parallel'
summary: A comprehensive post on distributed training in PyTorch. I provide a working code example, discuss a few potential pitfalls and give solutions to some commoly encountered bugs.
date: "2020-07-01T18:00:00Z"
reading_time: true  # Show estimated reading time?
share: false  # Show social sharing links?
profile: false  # Show author profile?
comments: true  # Show comments?
categories: ["PyTorch"]
tags:
  - PyTorch
  - Parallel Processing
  - Distributed Training
  - 2020
---

This post is intended to serve as a comprehensive tutorial for training (very) deep and memory-intensive models using PyTorch’s parallel processing tools. My general observation about these official documentations is that they tell you ***how*** to run things, but when you stumble upon an error, you are on your own, hopping from GitHub's issues and StackExchange's frantic discussions in the hope of finding a quick-fix solution. It is precisely during these times I think that a "potential pitfalls" or "common errors" section (or even as an appendix) would do no harm and help save some precious time. I know this definitely sounds na&iuml;ve but hear me out, things do get complicated when it involves coordinating between multiple GPUs. Keeping that in mind, here is an outline for this post: I start by explaining some fundamentals of PyTorch’s parallel processing tools, namely the DataParallel and DistributedDataParallel packages that I learned during my own research. As an example, I have provided a working example for training a Resnet101 model on CIFAR10 dataset with 4 GPUs on a single node. Finally, I discuss the [commonly encountered errors/bugs](#potential-pitfalls-and-commonly-encountered-bugs) in a distributed training environment and some solutions for the same that worked for me (I really want this to be the take-away from this post!). Though not extensive, I have made a list of errors that I encountered and discuss the fixes. Although the example shown here is relatively simple, the same method can be applied to other applications involving NLP and generative modelling tasks also. Hope it helps!

[//]: # (Although the official documentation for the same is enough for getting started, it is lacking in terms of layman definitions of some important terms. For all its complexity and esoteric language, writing bug-free code for distributed training is not an easy task, with solutions scattered all over the internet from Github issues to StackExchange’s discussions. Therefore, I try to explain some fundamentals of PyTorch’s DistributedDataParallel package that I learned during my own research. As an example, I have provided some starter code for training a Resnet101 model on CIFAR10 dataset with 4 GPUs on a single node. What I really want to be the take-away from this post is the part where I discuss the commonly encountered errors/bugs while distributed training and some work-arounds for the same. I spent hours trying to solve each bug, hopping from website to website in the hope of getting a quick-fix solution. I could eventually train a model, but I believe that the time spent for making things work could have been reduced had I been pointed the right sources. Though not extensive, I have made a list of errors that I encountered and discuss the fixes. Although the example shown here is relatively simple, the same method can be applied to other applications involving NLP and generative modelling tasks also. Hope it helps! )

### Why bother about Parallel Processing in the first place?

Anyone with a single powerful enough NVIDIA GPU and some experience with PyTorch probably knows that models can be easily transferred to the GPU by simply using `.to()` method. But, I believe that getting even a slightest hang of what PyTorch has to offer in parallel processing would go a long way. Instead of waiting patiently (or, impatiently) while your entire model runs on only one graphic card, why not “distribute” the load to multiple GPUs and make everything, well, *faster*? A few examples where having a single GPU is not enough include, working with 3D medical images, complex GANs (CycleGAN, PG-GANs), or training an ImageNet model from scratch, etc. To put things in perspective here, [OpenAI's GPT-3][1] language model has about 175 billion parameters and has been trained using **10,000 GPUs** (Yep, let that sink in for a moment).

### Distinction between DataParallel and DistributedDataParallel

It is worth mentioning that there are two ways by which the power of parallel processing can be harnessed. For the sake of brevity, I will be referring to `torch.nn.DataParallel` as *DP* and `torch.nn.parallel.DistributedDataParallel` as *DDP* from here on. First, DP is a simple model-wrapper which is quite elegant in its usage (as most of the heavy-lifting is done in the background) and it is quite simpler in terms of changing the code to reflect parallelism, compared to its counterpart, DDP. However, as easy as it sounds, there are some important caveats that prevent it from being used often.

### DataParallel

DataParallel implements a module-level parallelism, meaning, given a module and some GPUs, the input is divided along the batch dimension while all other objects are replicated once per GPU. In short, it is a single-process, multi-GPU module wrapper. 

[//]: # (That is, it works only in settings where there is a single-machine with mulitple GPUs or a single node on a remote compute cluster with mulitple GPUs.) 

To see why DDP is better (and faster), it is important to understand how DP works. A series of steps are performed under the hood involving both forward and backward passes. For the former, (1) The master GPU scatters the inputs (chunked along the batch dimension) to other GPUs, (2) the models and their parameters are replicated once per GPU, (3) the computations are performed in parallel, and (4) the master GPU gathers the outputs from each GPU. As for the latter, the loss gradients are computed on the master GPU. They are scattered (again) across all GPUs and backpropagated in parallel. Finally, the master GPU reduces all the gradients. This marks the end of one forward and backward pass in a DP training setup. Figure 1 shows these processes. For simplicity, I have made it look similar to the working of a neural network (succumbed to the [availability heuristic][2] there!)

{{< figure library="true" src="ddp-figures/bothPasses.png" title="Left panel - the input is scattered and the models are replicated across all GPUs. Parallel processing is done and the master GPU (GPU-0) gathers all the resulting outputs. Right Panel - Loss gradients are computed on the master GPU and data is scattered again for parallel processing. Finally, the master GPU reduces all the outputs. (Notice the number of times the GPU-0 is called in action)" numbered="true" >}}

This is all well and good but what happens in practice? DP suffers from a serious problem of imbalanced memory usage on the primary (master) GPU. Not to mention the fact that it becomes significantly slower performance-wise due to the additional overhead of transferring data (mainly tensors) to and from the master GPU. Let’s back up a little and understand what that means. As can be seen from the figure, it is the master GPU on which the loss gradients are computed and final outputs are gathered and summed. These inherent operations put a heavy load on it by consuming more memory. As a result, the hideous `CUDA: out-of-memory` error is very commonly encountered. 

### DistributedDataParallel

This is where we’ll get to the meat of this post. One of the major selling points of DDP is the fact that it works in multi-node, multi-GPU settings, unlike DP. To understand how, we need to borrow some definitions from torch’s `distributed` module in general. 

The `torch.distributed` package provides the necessary communication primitives for parallel processing across several nodes, processes, or compute cluster environments. DDP is essentially a wrapper that allows synchronous communication between these nodes. Note that it is different from the `torch.multiprocessing` package in that it allows distributed training across a unified network-connected machines. Even though it is well-suited for multi-node and multi-GPU settings, it is recommended over DP for the following reasons:

  1. **Reduced time spent between data transfers**: With distributed training, each process that is spawned has its own optimizer and performs its own optimization step. This prevents the time-consuming process of transferring tensors between nodes because each process has the same gradients.
  2. **No additional overhead**: The communication overhead borne by the interpreter for copying models, multithreading and running processes from a master GPU is eliminated. This is especially a problem with large models having recurrent layers as they often crash the Python interpreter.
  
To realize this, synchronous communication between processes and nodes is paramount. Let us see how that is accomplished. The first step is to setup a distributed environment (more on it later). We need to define an IP address and a port so that our processes can communicate with each other, coordinated through a master. The processes have to run on top of *something*, typically, a backend that supports (and optimizes) all point-to-point communications. The environment we referred to earlier, is defined by the following variables that help the spawned processes to be connected to the master and exchange information among them (or, in communication parlance, *handshake* with them).

[//]: # (`torch.distributed`’s init_process_group method along with a few other initializations helps us do just  that.)

  1. **Rank**: This decides whether a process is a “worker” or a “master”. Typically, a process with rank 0 is the de-facto master. 
  2. **World size**: It is just a fancy name for the total number of processes to be run. World size informs the master about the number of workers it has to wait for (for synchronization).
  3. **Master Address**: The IP address of the machine that can host the process with rank 0. 
  4. **Master Port**: A free port on the machine hosting a process with rank 0.
  5. **Backend**: This provides a platform that supports communication between different processes. Different backend options are available depending on the task. For instance, the NCCL backend is used for parallel GPU computations because it provides optimized communication between CUDA tensors.
  
An important feature of DDP that requires special mention is the ***all-reduce*** operation. Recall in the case of DP, the master GPU gathers the outputs for calculating gradients and scatters them again for parallel computing, only to be reduced by the master GPU again (notice the amount of data transfer). With DDP, instead of aggregating (gathering) the gradients on the master GPU, they are efficiently all-reduced across GPUs (see figure 2 for a clear idea of reducing operations). This way, each GPU (or, each process in our case) has its  own set of gradients to work with and the backward pass is also concurrently done without any performance bottleneck. In order for the model weights to be synchronized all the processes, the gradients are averaged. 

{{< figure library="true" src="ddp-figures/reduce.png" title="A typical Reduce operation." width="340" >}}
{{< figure library="true" src="ddp-figures/all_reduce.png" title="A typical All-Reduce operation. Notice how the summed outputs are used in both operations. Source: [PyTorch Docs](https://pytorch.org/tutorials/intermediate/dist_tuto.html)" width="340" numbered="true" >}}

Now you know what distributed training means and why should you use DDP instead of DP. Let us see a few important code snippets below that lets us perform distributed training. Note: In the interest of the length of this post,  I will not be posting entire code for training the ResNet101 model (it is available on my GitHub), rather, my focus would be on these snippets that are essential for wrapping the model around the distributed package ***AND*** the section on the [typical errors/bugs encountered](#potential-pitfalls-and-commonly-encountered-bugs) that comes right after. There are many resources out there that show how to write code that performs distributed training, but I believe it is much more important to have an idea about the common errors and the tricks to solve them. Also, it is worth mentioning that a single process can be used to control multiple GPUs, but that is slower than using a single process per GPU. I will be demonstrating an example only for the latter. 

#### Importing packages
The following are the additional packages required for distributed training:
```py
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
```
#### Setting up the distributed environment
The code snippet below is all it takes to initialize a distributed training environment. I have included additional comments for further understanding.
```py
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successful communication across multiple processes involving multiple GPUs.
```
#### Cleaning up after training (optional, but considered as a good practice)
```py
def cleanup():
    dist.destroy_process_group()
```
#### Loading the data
In distributed training, the data is loaded differently compared to the standard way (`torch.utils.data.DataLoader`). This is to account for the fact that the data must be divided among the GPUs. More importantly, by passing a `DistributedSampler` instance to the `DataLoader` it is ensured that each process loads an exclusive subset of the entire training data.
```py

training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)

# torch.distributed's own data loader method which loads the data such that they are non-overlapping
train_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=training_set, 
								     num_replicas=args.world_size, rank=rank)
trainLoader = torch.utils.data.DataLoader(dataset=training_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True,
                                              sampler=train_data_sampler)

```

That's it! The code snippets above show the only considerable changes needed for wrapping your model around the distributed package. And now for the final act.

### Potential pitfalls and commonly encountered bugs
  
  * **Setting a random seed for all GPUs** - Having read this far, you know that synchronized communication between multiple processes is the cornerstone of distributed training. Therefore, to ensure that gradients across all processes are same, a seed value must be set initially. This ensures that the (replicated) models are initialized with the same weights.

  * **Batch size greater than number of GPUs** - While this makes sense intuitively because the greater the batch size the more it will be divided across all GPUs, it is also important to be cognizant of the GPU memory. Large models do not allow the batch sizes to go beyond 2 or 4, in which case having more GPUs prevents them from being utilized fully. 

  * **Saving a model on only one local rank** - It is highly recommended that when checkpointing a model, its parameters are saved on one local rank (preferably on the de-facto master (rank 0)). This is particularly important when a model has more than 1 network (typically in GANs with generator and discriminator networks). It is easy to wrap models around the distributed package and forget about how they are saved during training (happened with me, had to learn it the hard way). Recall that each process has its own model and optimizer and they are all coordinated through a master IP address and a port. So, when a generic method for saving models is used, chances are that those saved models are being overwritten by multiple processes, hence, corrupting the information and *making them irretrievable*. Here is how the error looks when loaded using `torch.load()`:
  > RuntimeError: storage has wrong size: expected -7659745797817883467 got 512
  
  ```py
  # The generic method of saving models (say, at the end of every epoch)
  torch.save(model.state_dict(), save_path)	# this generally results in a RuntimeError as shown above.
  
  # A work-around for it:
  if (rank == 0):
      torch.save(model.state_dict(), save_path)	# this ensures that only one process saves the model to save_path.	
  ```
	
  * **Including “main()” function block in the code** - It is important that your code contains the "main()" block in it. Due to the way the new processes are spawned, the child process needs to be able to import the script containing the target function. Wrapping the main part of the application in a check for "main()" ensures that it is not run recursively in each child as the module is imported. 

  * **Issue with loading checkpointed models** When the saved models are loaded to resume training from the last saved epoch, there is a chance of encountering the hideous “CUDA: out-of-memory” error. The quick-fix solution for this is to first load the models on the CPU (using `map_location=’cpu’` argument in `torch.load()`) and then transfer those loaded models onto the GPU (using `.to(device)`). The reason as to why this works is not entirely known. It seems that it has got something to do with PyTorch’s caching allocator for GPUs.
  
  * **Location of where the model is loaded** - When a model is saved using `nn.DataParallel` or `nn.parallel.DistributedDataParallel`, it is stored in module. But, typically we don’t load the model using `DataParallel` and this throws an error. There are two ways of solving this: (1) a simple way is wrap the model around DP just for the sake of loading, or (2) create a new dict file by removing the `module.` prefix and load it in the generic way using `torch.load()`. Here is a snippet for the same: 
  ```
  parallel_state_dict_model = torch.load('path/to/your/model', map_location='cuda')
  state_dict_model = OrderedDict() 
  for k, v in parallel_state_dict_model.items():
  	name = k[7:]    # removing "module."
	state_dict_model[name] = v
  # Loading the updated (renamed) model
  model.load_state_dict(state_dict_A2B)
  ```
  Now you know what to do when this error pops up: 
  > KeyError: 'unexpected key "module. ..... .weight" in state_dict'

This concludes my post on distributed training with PyTorch. It has been an incredibly long post. Thank you for going through it. Your patience is highly appreciated. One thing I did not mention earlier that this post is motivated by my fascination with the sheer amount of computational power that GPUs bring to the table. They are certainly the drivers of today's machine learning research. I personally have witnessed a massive decrease in the total training times.

[//]: # (I would not be surprised if harnessing GPU computations on-the-fly would bring us one step closer towards artificial general intelligence (AGI). Exciting times!)

A working example in its entirety is available [here][10]. This post also inspired by a few other posts on the same topic (see references). I definitely recommend going through them as they are probably much better in some aspects. 

### Acknowledgements
Many thanks to [Karan Praharaj][8] for his suggestions on earlier versions of this post. Special thanks to [Compute Canada][9] for their valuable resources without which this post would have not been possible.

**Edit** - Thanks to Rapha&euml;l Royer-Rivard for his suggestion on including a code snippet at pitfall #3. You can find him [here][11].

References:
	
  1. [Seb Arnold’s brilliant tutorial on distributed applications with PyTorch][3] 
    
  2. [Getting started with DistributedDataParallel][4]
  
  3. [Training Neural Nets on Larger Batches][5]
  
  4. [Quick Primer on Distributed Training with PyTorch][6]
  
  5. [Distributed data parallel training in Pytorch][7]

[1]: https://news.developer.nvidia.com/openai-presents-gpt-3-a-175-billion-parameters-language-model/#:~:text=%E2%80%9CThe%20supercomputer%20developed%20for%20OpenAI,companies%20stated%20in%20a%20blog.
[2]: https://en.wikipedia.org/wiki/Availability_heuristic#:~:text=The%20availability%20heuristic%2C%20also%20known,%2C%20concept%2C%20method%20or%20decision.
[3]: https://pytorch.org/tutorials/intermediate/dist_tuto.html
[4]: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
[5]: https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
[6]: https://levelup.gitconnected.com/quick-primer-on-distributed-training-with-pytorch-ad362d8aa032
[7]: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
[8]: https://karanpraharaj.github.io/
[9]: https://www.computecanada.ca/
[10]: https://github.com/naga-karthik/ddp-resnet-cifar
[11]: https://www.polymtl.ca/liv4d/equipe
