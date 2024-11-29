| **LF Name**       | **Coverage** | **Overlaps** | **Conflicts** | **Correct** | **Incorrect** | **Emp. Acc.** |
|--------------|--------------|--------------|---------------|-------------|---------------|---------------|
| gpt          | 0.99         | 0.99         | 0.44          | 5456        | 979           | 0.85          |
| gpt-cohere   | 1.00         | 1.00         | 0.44          | 5654        | 814           | 0.87          |
| gpt-DDG      | 0.99         | 0.99         | 0.44          | 5544        | 858           | 0.86          |
| haiku        | 1.00         | 1.00         | 0.44          | 5005        | 1463          | 0.77          |
| haiku-cohere | 0.98         | 0.98         | 0.43          | 4928        | 1397          | 0.78          |
| haiku-DDG    | 0.97         | 0.97         | 0.43          | 4455        | 1804          | 0.71          |


The labeling function analysis of the weak labels for LIAR-WS dataset. 
Each LF corresponds to an LLM that provides a misinformation label for a given query.

The prompt provided as input to the LLM allows it to invoke a search with respect to determining if a given query is true (not misinformation) or false (misinformation).

Misinformation labeling prompt:
```
Given the {query}, state "True statement; Factuality: 1" if you think the statement is factual, or "False statement; Factuality: 0" otherwise. You have access to a search engine tool. To invoke search, begin your query with the phrase "SEARCH: ". You may invoke the search tool as many times as needed. 
```