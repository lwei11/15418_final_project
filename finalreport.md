# Final Report

## Summary
We implemented data and pipeline model parallelism for a neural network that we built from scratch. We used MPI and tested both forms of parallelism on GHC. We varied different parameters such as batch size and number of hidden units to see how these affect the performance of each type of parallelism.

## Background
For our project, we implemented a deep neural network entirely from scratch (meaning no preexisting libraries or functions like Pytorch). Our network was made of a sequence of various types of layers such as linear layers, a sigmoid activation layer, and a softmax output layer. We used the linear and sigmoid layers for most of the calculations and the softmax layer to finalize the output. We wanted to test our parallelization in a model that used both linear and non-linear computations to make sure that we saw the benefits for different types of neural networks. 
### Key Data Structures
We have a few main data structures that we use for our deep neural network. The first is the weight matrices, which are used to actually train the model and keep track of the parameters for the computational layers. We frequently update these layers since they need to be changed every time the data is passed through one of our layers. Another data structure we use a lot is the bias vector. This is used to introduce a little more freedom into the data and allow the neural network to be more accurate when testing it at the end. Finally, we store the activations for each layer during forward propagation so we can use them again during the backward step. For training the network, we have some data and associated labels stored in a csv file which we import into our main file whenever we run the network. 
### Key Operations on Data Structures
When working with these data structures, the main tasks include matrix multiplications, element-wise operations (like applying the same function to each value), and vector transformations. Forward propagation is where the input data gets multiplied by weight matrices, bias values are added, and then a sigmoid activation function is applied to each result. Backpropagation is a bit more complex—it calculates how much the model’s predictions are off by using gradients. This involves transposing weight matrices and doing more matrix-vector multiplications. Since these calculations are repeated for every example or batch of data, they take a lot of computing power, which makes them perfect for parallel processing to save time.
### Algorithm Inputs and Outputs
The neural network starts by taking in feature vectors and at the other end, it outputs probabilities for each class (its predictions). During training, it also uses labels and calculates a loss function to measure how far off its guesses are. This loss function is important because it guides the adjustments to the weights and biases through gradient descent. At the end of the training process, the model has optimized weights and biases that can be used to predict new, unseen data.
### Why Parallelization Is Needed
The most time-consuming part of training a neural network is the matrix multiplications in forward and backward propagation. These calculations can really slow things down, especially with large networks or high-dimensional data. The runtime grows based on the number of neurons in each layer and how much data there is. That’s why parallelizing these operations is so useful—it cuts down on training time. There are a few ways to implement this, some of which we will showcase later.
### Workload Breakdown and Dependencies
The workload in a neural network naturally follows the sequence of layers. In forward propagation, each layer relies on the output of the layer before it. The same thing happens in backpropagation, but in reverse: each layer’s gradients depend on the ones calculated in the layer after it. However, between layers the calculations do not affect each other. This allows us to perform pipeline model parallelism since we can move data along the pipeline in parallel. Similarly, data parallelism works because gradients for different batches of data can be calculated at the same time. There is a bit of a challenge that comes when it’s time to sync everything up and make sure the gradients are properly combined.
### Parallelism and Locality
Neural networks offer plenty of opportunities for parallelism. Data parallelism splits the input data across multiple processes, which makes it very scalable and great for handling large datasets. Model parallelism is a bit more complicated to set up but works well for large networks that don’t fit into one process’s memory. The calculations themselves, like matrix multiplications and element-wise operations, would also work well with SIMD (Single Instruction, Multiple Data), where the same operation can be applied to many numbers at once.

## Approach
### Overview
We created two new training functions for our deep neural network that each utilized a different type of parallelization technique. The first used data parallelism where we split the data into batches and trained each batch in parallel. After every processor had trained on its batch and updated its weights, we aggregated them to get the final model. The other technique was pipeline model parallelism. We split the model so that every processor had its own layer to compute. The data was then split into micro batches and moved across the pipeline where each processor would start on the next batch of data as soon as it finished its previous one.
### Technologies Used
Our implementation was written in C++ and we used MPI for all parallelization/communication. The computers we targeted were multi core CPU processors where we attempted to use MPI to distribute work across them and communication information between them. For the technologies used for the neural network itself, it was completely implemented from scratch and did not use any preexisting libraries for setup, computation, or analysis. This allowed us to have a greater degree of control over what was parallelized and made sure no auto optimizations were implemented without our knowledge.
### Mapping the Problem to Parallel Machines
We utilized two main forms of parallelism: data and pipeline model parallelism. For our data parallel training function, we partitioned the data into chunks and assigned each to its own processor. At the end of the epoch or batch round, all gradients were aggregated using MPI_Allreduce and synched across all processors. This allowed for consistent models and better accuracy. See the diagram below for a visual of how this worked:
### DIAGRAM
For our pipeline model parallel training function, we divided the layers of a neural network and assigned them each to a processor. Because this required the division of functions, we had to hardcode it for different numbers of processors rather than dynamically allocate data as in the data parallel version. For this version of parallelism, the data was also split into chunks and these chunks were passed along the processors in a pipeline format. Each processor would receive a chunk, perform its assigned layer’s calculations on the chunk, and then pass it forward through the pipeline. At the end, each processor would then send the chunk backwards through the pipeline for the backpropagation step. Finally, the weights would be updated at the end. See the diagram below for the visualization of this version:
### DIAGRAM
### Changes to the Serial Algorithm
The main changes we made to the serial algorithm were batching and communication. Each form of parallelism used batching in some capacity and we had to change the algorithm to account for this. Additionally, for pipeline parallelism we had to change the forward and backward functions since we were splitting up the layers of the network. However, we did not parallelize the computations themselves i.e. the actual layers were left unchanged.
### Optimization Process
We tried a few different approaches to optimize both forms of parallelism. For data, we initially tried our own ring structure to pass messages around, but this ended up being fairly inefficient since we were passing weight matrices. There would be a lot of large messages passed, and there wasn’t a great opportunity to perform work while these were being passed since each processor needed updated gradients before working on the next batch or the accuracy would suffer a lot. So instead we ended up using MPI_Allreduce since we wanted to sum of the gradients to divide by the number of processors. This worked better overall. For pipeline model parallelism, we initially didn’t have a pipeline and just used model parallelism. But since we didn’t have an enormous neural network, this did not provide a whole lot of speedup, so we introduced the pipeline to make it better. We first tried doing it with smaller numbers of processors and asynchronous communication, but similar to before there wasn’t really a lot of time in between messages to do computation since each process needed the results from the previous layer to start. So instead we decided to use synchronous sends and receives to pass the parameter vectors through the network.
### Existing Code Base
The implementation was developed from scratch, with no reliance on external neural network libraries. This allowed us to maintain full control over the architecture and parallelization strategies, tailoring them to our specific use case. Our new training functions were designed specifically to integrate MPI operations into the training process, ensuring efficient use of parallel hardware.

## Results
Performance Measure and Experimental Speedup: We measure performance using speedup. Specifically, we compare the time of each run to the sequential program runtime on CPU for 200 or 400  hidden units per layer, 8 layers, batch size of 80 for mini-batch stochastic gradient descent, and 20 epochs, unless otherwise specified. We ran the experiments on GHC machines with this command: mpirun -np <number of processors> ./neuralnet inputs/small/small_train.csv inputs/small/small_validation.csv inputs/small/small_train_out.labels inputs/small/small_validation_out.labels inputs/small/small_metrics_out.txt <number of epochs> <number of hidden units> <init flag> <learning rate> <batch size>. 

### Speedup Graphs: 
1. Number of processors’ effect on speedup for data parallel implementation: (400 hidden units per layer, 8 layers, batch size of 80, 20 epochs)

### GRAPH
(We do not make a similar graph for pipeline model parallel implementation, because in our implementation of pipeline model parallelization, each processor will process a fixed number of layers, and with 8 layers, and the number of processors are fixed to be 8.)

2. Batch size’s effect on speedup for pipeline model parallelism and data parallelism: (200 hidden units per layer, 20 epochs)
### GRAPH

3. The number of hidden units’ effect on speedup for pipeline model parallelism and data parallelism:
(8 layers, batch size of 80, 20 epochs)
### GRAPH

### Problem size vs Performance: Yes, reporting results for different problem sizes is very important because problem size directly impacts performance. Different workloads (batch size vs hidden units) exhibit different execution behavior, which means scaling problem size is necessary to fully utilize parallelization. In our experiments, speedup is mainly influenced by these parameters:
Batch Size: As batch size increases, speedup improves for both data parallelism and model parallelism. For example, in data parallelism, speedup rises from 1.0 to 3.5 as batch size goes from 0 to 80. Similarly, in model parallelism, speedup grows from 3.0 to 3.9. This shows larger batch sizes lead to better parallelism because the work per processor increases, which reduces the overhead.
Hidden Units: Speedup also improves as the number of hidden units increases. For example, in data parallelism, speedup jumps from 0.2 to 4.0 when hidden units increase from 10 to 400. For model parallelism, the behavior is more complex, showing fluctuations initially but then reaching over 3.7 speedup.
Speedup Limitations: We identified and tested several reasons for how speedup is limited. 
1. Processors initialization and communication overhead: in the data parallel graph for different numbers of processors, speedup increases linearly but only up to ~4.1x for 8 cores. Ideally, it should be 8x. The limitation mainly comes from dependencies and communication overhead when splitting data across cores. To verify our idea, we tested a completely sequential version without any implementation of parallelism and launched multiple processors with the sequential version, and the initialization and overall run time had an apparent increase when the number of processors increased. 
2. Communication Overhead: When the number of hidden units is small, the pipeline model parallelism suffers from significant performance dip. This is likely due to these two reasons: a. pipeline model communications more frequently (once per batch per layer) than the data model (once per batch), and b. The number of hidden units is the size weight, which is the size of the message being communicated, and with the number of hidden units is small, the message is not fully filled, and the communication overhead is more significant relative to message size. 
3. Data compatibility: We noticed that there is a dip for pipeline model parallelism when the number of hidden units is 100. We think that this is likely because the data happens to be unfit for 100 hidden units, because the performance for data parallelism is also not as good as this size of hidden units, but it is less obvious since the communication is less frequent. 
4. Sequential training: in the data parallel approach, each processor still trains the model sequentially with a subset of a mini-batch, and in the pipeline parallel approach, each processor still computes the weight of each layer by calculating the batch sequentially. Thus, each model has some part running sequentially. We think that it would be optimal if the two approaches are combined: in each iteration, a subset of processors processes a layer collectively, and each processor in the subset processes a subset of the mini-batch. This was our stretch goal, but we ended up not having enough time to implement this, and we are planning to experiment with the combined model more during the winter break. 
5. Memory-Bound Issues:
In model parallelism, where each processor handles different layers, performance improves with hidden units, but fluctuations at lower sizes suggest data transfer or memory bandwidth limitations.
In conclusion, even though the speedups did not achieve theoretical limits, a solid analysis shows where bottlenecks occur: in data parallelism, increasing batch size and hidden units improves performance, but communication overhead caps speedup; in model parallelism, distributing layers efficiently improves performance at large hidden unit counts, but data transfer issues cause early dips.
### Deeper analysis
We overlapped communication and computation to hide data transfer delays, but based on our estimates, here are the three main components and an estimated time breakdown of each component: 
Initialization: Time to initialize the processors.
Computation: Time spent on processing forward/backward passes of the neural network.
Communication: Overhead of synchronizing gradients and results across processors.
Communication: Overhead of synchronizing gradients and results across processors.
Parallel Method
Initialization (%)
Computation (%)
Communication (%)
Data Parallelism
10% Initialization
70% Computation
20% Communication
Pipeline Model Parallelism
15% Initialization
70% Computation
40% Communication
(In pipeline Model parallelism, the number do not sum up to 100%, because some computation and communication are overlapped). 
In data parallelism, the majority of the time is spent on computation, approximately 70%, as each processor independently performs forward and backward passes on a subset of the data. Initialization takes a small fraction of the time (10%), as it mainly involves setting up processors and distributing the data. However, communication overhead accounts for 20% of the time, particularly as the number of processors increases, requiring frequent synchronization of gradients.
Model parallelism, on the other hand, spends a much larger portion of time on communication (around 40%) because processors need to synchronize at layer boundaries. Initialization is 15%, slightly higher at 10% from the data parallel approach, since splitting the network across processors involves additional setup. Computation contributes to about 70% of the total time, with its dominance increasing as the number of hidden units grows. This division highlights that model parallelism is more sensitive to communication overhead, while data parallelism is heavily influenced by the amount of computation and batch size. The number do not sum up to 100%, because some computation and communication are overlapped in this approach.



### Choice of machine
The GHC machine (CPU-based) was a sound choice for smaller batch sizes and fewer hidden units. However, a GPU might have been better for large batch sizes and larger hidden units because GPUs excel at parallelizing matrix multiplications in neural networks. For data parallelism, a GPU could reduce synchronization and computation time.
For model parallelism, the CPU works well for smaller hidden unit counts but struggles with memory bandwidth. However at the same time, dividing the data too much would cause the accuracy to suffer, so for this to actually be beneficial we would likely also need an extremely large dataset.



