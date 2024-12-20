# Midpoint Checkin

## Schedule
Week 0 (Nov 11 - Nov 17): Complete project proposal and learn the basics of NCCL - Complete \
Week 1 (Nov 18 - Nov 24): Create starting code for neural network and begin implementing data parallelism - Complete \
Week 2 (Nov 25 - Dec 1): Finish implementing data parallelism and start working on model parallelism - Complete \
Week 3 First Half: (Dec 2 - Dec 5): Begin model parallelism and complete milestone report \
Week 3 Second Half: (Dec 6 - Dec 8): Finish model parallelism and begin converting to NCCL \
Week 4 First Half (Dec 9 - Dec 12): Finish conversion, test on PSC machines, and create poster for demonstration \
Week 4 Second Half (Dec 13 - Dec 15): Create final report and clean up code \
Division of Work: We are collaborating on everything

## Completed Work
We have fully implemented our neural network in C++. It works and has expected error and loss when run without using any parallelization/mpi commands. We have also implemented data parallelism in our network. The user can decide how many processors they want to use and what batch size they would like to use for the training data. A larger batch size means that each processor trains on a larger portion of data before sharing its weights with the rest of the processors. This results in faster computation since there are less messages passed, but it is less accurate since there is less collaboration among the processors. All messages are the same size currently since we are passing the weight matrices around, adding them up, and taking the average of them.

## Goal Attainability
For the core goals and deliverables of our project, we are on track to complete them. We shuffled the work around a little bit compared to our initial schedule since we found that implementing each type of parallelism was more straightforward than each task. However, there is still the same amount of total work and progress is going well. We fully expect to finish both types of parallelism and test them with both training the network and hyperparameter tuning. Finally, we are using MPI for the initial testing since we need the PSC machines to run NCCL. After we have fully implemented the parallelism types using MPI, we will convert to NCCL since the commands we are using are basically the same for both interfaces. For the nice to have we mentioned (a hybrid model) we may or may not be able to achieve this. It is possible, but it'll depend on our progress and how much work we have in other classes.

## Poster Session Demonstration
Our poster session will be a combination of a demo and a graph. We will demonstrate how our model can be trained fastly and produce accurate results on character / handwriting recognition. We will go over the accuracy and speedup for different numbers of processors and different amounts of data and produce speedup graphs. Last but not least, we hope to provide a comparison and an overview of the improvements that can be gained from using NCCL to parallelize the training and hyperparameter tuning of neural networks.

## Preliminary Results
We have only tested the networks on small datasets for correctness and we can't use NCCL yet since the GHC machines do not support it. So while we don't have preliminary speedup results at this time, we can verify that the network is working correctly with MPI and getting fairly good error/loss depending on the parameters we input.
<img width="774" alt="Screenshot 2024-12-04 at 8 06 37 PM" src="https://github.com/user-attachments/assets/56977c94-d4e0-497a-a7ee-d208bb33be1e">
<img width="773" alt="Screenshot 2024-12-04 at 8 07 29 PM" src="https://github.com/user-attachments/assets/7eda3483-9c76-4171-ae3e-0971d1c4f0f1">

## Issues
There aren't any issues at this time. The only thing is that we have to use MPI for our initial implementation and testing since the GHC machines do not have NCCL. This was not expected when we wrote the project proposal, but as we found out later, we added this step into our work schedule. When we convert to NCCL we will have to run tests on PSC which we can't do a lot of, so we need to make sure everything is fully working beforehand.


