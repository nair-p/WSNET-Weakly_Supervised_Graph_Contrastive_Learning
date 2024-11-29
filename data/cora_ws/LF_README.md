|  | **Class**     | **Keywords**                                                                                                                                                                                        |
|------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      | Genetic Algorithms     | {['evolution', 'genetic programming', 'population', 'chromosome', 'gene', 'generation',  	'breeding', 'selection', 'crossover', 'mutation', 'fitness']}                                                      |
|      | Reinforcement Learning | ['reward', 'agent', 'environment', 'action', 'state', 'thompson sampling', 'reinforce', 'bellman', 'actor','critic', 'Q-Network', 'episode', 'greedy','epsilon','experience','replay', 'policy']             |
|      | Theory                 | ['complexity', 'analysis', 'theorem', 'theoretical', 'bounds', 'bias', 'estimate', 'variance', 	'error', 'approximation']                                                                                    |
|      | Rule Learning          | ['algorithm', 'induction', 'rule', 'relational', 'association', 'knowledge', 'decision', 'logic', 'expert']                                                                                                  |
|      | Neural Networks        | ['neuron', 'non-linear', 'linear', 'dense', 'convolution', 'recurrent', 'activation', 'multi-layer', 'perceptron', 'fully-connected', 'weights', 'training', 'gradient', 'loss', 'backpropagation', 'batch'] |
|      | Probabilistic Methods  | ['probability', 'monte carlo', 'markov','bayesian', 'process', 'causal', 'expectation', 'inference',	'graphical', 'convergence', 'belief', 'mixture', 'distribution']                                        |
|      | Case-Based             | ['cases', 'design', 'system', 'case', 'based']                                                                                                                                                               |

For each of the 7 classes, we manually crafted a list of keywords related to those classes. We then did a strict-match search for mentions of any of those keywords in the abstract and title of the papers in Cora. If there is a match in the abstract, then that paper is assumed to belong to that class. The LFs for abstract were separate from the ones for titles. 

The LF analysis is provided below:

|    | Polarity   |   Coverage |   Overlaps |   Conflicts |   Correct |   Incorrect |   Emp. Acc. |
|---:|:-----------|-----------:|-----------:|------------:|----------:|------------:|------------:|
|  0 | [0]        |  0.197526  |  0.175602  |   0.173432  |       414 |         496 |   0.454945  |
|  1 | [1]        |  0.201433  |  0.199913  |   0.199913  |       275 |         653 |   0.296336  |
|  2 | [2]        |  0.356631  |  0.349685  |   0.348383  |       304 |        1339 |   0.185027  |
|  3 | [3]        |  0.419579  |  0.412416  |   0.409811  |       142 |        1791 |   0.0734609 |
|  4 | [6]        |  0.317994  |  0.311265  |   0.31018   |       628 |         837 |   0.428669  |
|  5 | [5]        |  0.3117    |  0.306924  |   0.304971  |       355 |        1081 |   0.247214  |
|  6 | [4]        |  0.436076  |  0.426742  |   0.424571  |       316 |        1693 |   0.157292  |
|  7 | [0]        |  0.0790102 |  0.0757543 |   0.0735837 |       132 |         232 |   0.362637  |
|  8 | [1]        |  0.0223573 |  0.021489  |   0.021489  |        32 |          71 |   0.31068   |
|  9 | [2]        |  0.0824832 |  0.0800955 |   0.0787931 |       100 |         280 |   0.263158  |
| 10 | [3]        |  0.14695   |  0.144563  |   0.141958  |        66 |         611 |   0.0974889 |
| 11 | [6]        |  0.0601259 |  0.0566529 |   0.0555676 |       169 |         108 |   0.610108  |
| 12 | [5]        |  0.0950727 |  0.0918168 |   0.0898633 |       255 |         183 |   0.582192  |
| 13 | [4]        |  0.0546994 |  0.0538311 |   0.0516605 |       111 |         141 |   0.440476  |