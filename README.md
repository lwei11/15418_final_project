# 15418 Final Project: Neural Network Parallelization

## Title
Model vs Data Parallelism in Neural Networks

## URL
Webpage: https://lwei11.github.io/15418_final_project/ \
Github Repository: https://github.com/lwei11/15418_final_project/ \
Milestone Report: https://lwei11.github.io/15418_final_project/midpoint \
Final Report: https://lwei11.github.io/15418_final_project/finalreport

## Summary
Our project aims to explore the differences between model and data parallelism in neural networks. Specifically, we will look at how these types of parallelism differ when they are used in hyperparameter tuning and distributed neural network training. We will use MPI to perform these parallelizations.

## Background
A deep neural network (DNN) is a type of artificial neural network (ANN) with multiple layers between the input and output layers. These intermediate layers are called "hidden layers" because they do not directly interact with the external inputs or outputs, but they process the data through numerous nonlinear transformations. Each layer consists of nodes, or neurons, which are connected to the neurons in the previous and subsequent layers. DNNs are particularly powerful for complex tasks like image recognition, language translation, and natural language processing, where they can capture intricate patterns and relationships in data. 

In this project, we will train DNN for Optical Character Recognition (OCR). The inputs are black-and-white images of 26 handwritten English letters, and the output will be the most probable English letter recognized by the model. In our implementation, the modules we will define for each layer of our DNN contains a Linear layer and a Sigmoid layer. Each layer contains a forward method b = *.FORWARD(a), and a backward method ga = *.BACKWARD(gb). In other words, the forward method yields the output, b, given the input, a; meanwhile, the backward method yields the gradient with respect to the input, ga, given the gradient with respect to the output, gb. A peusocode for training a 2-hidden-layer Neural Network with stochastic gradient descent is as follows:
<img width="949" alt="Forward" src="https://github.com/user-attachments/assets/a13bf00a-fe8e-413a-9e7d-ffeaef346f23">
<img width="941" alt="Backward" src="https://github.com/user-attachments/assets/3b50343b-aa61-4de5-a848-d9fe50cc5739">
<img width="975" alt="Train" src="https://github.com/user-attachments/assets/152b1cca-68e0-4235-9cd8-0424299bea4a">
<img width="948" alt="Test" src="https://github.com/user-attachments/assets/c6bc1dc2-2863-436d-9ceb-babfae9a832b">

The forward and backward computation step involves applying similar instructions to a large amount of data, thus implies great potential for parallelism. In addition, choosing the best hyper parameters (number of hidden layers, number of neurons each layer, learning rate, number of epochs) with grid search or random search also implies great potential for parallelism.


## The Challenge
Some of the popular neural network libraries such as PyTorch and Tensorflow have automatic parallelism in them to help speed up the process of training networks. We will not use these libraries and instead implement our own neural network from scratch. This will provide better control over what processes are parallelized and eliminate any automatic optimizations provided by these libraries. However, this will also be significantly more challenging since we will need to create all of the functions ourselves. Moreover, when split the training data across multiple processes and each node trained on a portion of the data, their learned weights need to be periodically averaged. This creates difficulty in synchronizing among the processors and assigning them reasonable weights when computing the average. In addition, while each of the layers depends on the previous one and the computations of them should seem to behave linearly, we need to identify and parallelize independent subparts in forward and backward propagation. 

Workload: The workload of training a neural network for OCR involves several dependencies due to the sequential nature of forward and backward propagation across layers. Each layer in the network depends on the output from the previous layer, meaning that memory access has a strong dependency chain with minimal locality; computations for one layer must be completed and stored before moving to the next. In data-parallel scenarios, the input data is split across multiple processors, each computing gradients independently, which introduces a high communication-to-computation ratio when synchronizing the gradients. This ratio increases as the number of processors increases, requiring frequent communication to average gradients across processors, especially in distributed environments. Additionally, divergent execution can occur due to varying computational loads across processors, as certain layers (like Softmax and CrossEntropy) might require more complex operations than others, leading to inefficiencies when some processors finish tasks earlier and remain idle while waiting for synchronization.

Constraints: Mapping the workload to a parallel system presents several challenges due to both the neural network's architecture and the system's limitations. In a model-parallel approach, where each processor is responsible for different layers, the strict dependency between layers means that any imbalance in layer computation time can lead to significant waiting times, impacting performance. Furthermore, each layer’s outputs must be communicated to the next processor, creating a bottleneck if network bandwidth is limited. In data-parallel systems, synchronizing model parameters across processors requires efficient communication protocols, which can be challenging when dealing with large neural networks and limited inter-processor bandwidth. Memory constraints also pose challenges, as storing intermediate outputs for backpropagation consumes significant memory, especially in deeper networks. Thus, achieving an optimal balance between computation and communication, as well as effectively managing memory across processors, is essential but difficult, given the diverse computational demands and dependencies in neural network training.

## Resources
We will implement our neural network from scratch since using PyTorch or Tensorflow would trivialize some portions of our task. We will not use any existing code or libraries to implement our project, and we are not using any books or papers as initial references. However, the pseudocode we are going off of for implementing our network does come from 10301 Introduction to Machine Learning. The main resource/special computer access we need is the PSC machines since we plan to use MPI to run and test our parallel implementations.

## Goals and Deliverables
Achievements: We plan to achieve working implementations of both versions of parallelism independently i.e. we will have a working version that uses data parallelism only and a working version that uses model parallelism only. If time permits, we will try to create a hybrid version that utilizes both methods of parallelism and see if we can optimize this hybrid version beyond the separate versions. We aren’t sure of the exact performance improvements, but we expect to see at least a 4x speedup in training the neural network on 8 cores. This is because we are implementing the neural network from scratch and only parallelizing certain parts of it, so we will likely not see true linear speedup. However, we think that we can achieve this goal since our from scratch neural network will not have any of the optimizations that the pre existing frameworks will have, thus our parallelization optimizations should have a noticeable impact on their performance.

Demo: For our demo, we mainly plan to show speedup graphs since that will be the best way to demonstrate how our parallelizations benefited our network. We also plan to actively train the network during the poster session to demonstrate that it can be trained faster and accurately. Additionally, if we have time we would like to implement a simple interface that allows users to interact with the neural network by drawing their own characters and testing them on the network. The primary way of demonstrating that our project was successful will be the speedup graphs though, since this is the main focus of the project. 

Questions: Our project is a type of analysis project since we will be looking at the strengths and weaknesses of different parallelization approaches with regards to neural networks. We hope to understand how data and model parallelism can help speedup training neural networks and which method (or a combination) will best fit the network depending on what we are training it for. We want to see how much of an impact these parallelizations have in isolation when there are no auto optimizations on the rest of the network. And if time permits, we also hope to see how these approaches can best be combined to achieve greater speedup than they can get individually. 

## Platform Choice
For the coding language, we plan to use C++ since it works well with MPI and the types of parallelism that we will be implementing. It is also a bit easier to work with than C and will allow us to focus more on the higher level aspects of our project rather than the nitty gritty details of the underlying implementation. We are using data and model parallelism for our workload because they both work well with different types of neural networks. Using both of them will provide a good analysis of the strengths and weaknesses of each and when a neural network would benefit from one or the other. For the computer choice, we are just going to write the code on our normal computers, but we will run the tests on the PSC machines since we need to use MPI for our experiments. 

## Schedule
Week 0 (Nov 11 - Nov 17): Complete project proposal and work on starter code - Complete \
Week 1 (Nov 18 - Nov 24): Finish starting code for neural network and begin implementing data parallelism - Complete \
Week 2 (Nov 25 - Dec 1): Finish implementing data parallelism and start working on model parallelism - Complete \
Week 3 First Half: (Dec 2 - Dec 5): Begin model parallism and complete milestone report \
Week 3 Second Half: (Dec 6 - Dec 8): Finish model parallelism and begin testing \
Week 4 First Half (Dec 9 - Dec 12): Finish testing and create poster for demonstration \
Week 4 Second Half (Dec 13 - Dec 15): Create final report and clean up code \
Divison of Work: We are collaborating on everything

