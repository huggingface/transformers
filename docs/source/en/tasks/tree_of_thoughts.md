# Tree of Thoughts

## Overview

Tree of Thoughts (ToT) is a framework that generalizes over the popular chain-of-thought approach to prompting language models (LLMs). It allows LLMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices. ToT has been shown to significantly enhance LLMs' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords.

1. Structured Decomposition: ToT breaks down complex problems into a series of interconnected thoughts, akin to the branches of a tree. Each thought represents an intermediate step in the reasoning process, gradually refining the LLM's understanding of the problem and guiding it towards a solution.

2. Divergent Exploration: Unlike traditional linear prompting, ToT encourages the LLM to consider multiple reasoning paths simultaneously, allowing it to explore a wider range of potential solutions. This divergent exploration mirrors the human tendency to brainstorm multiple ideas before converging on the most promising one.

3. Self-Evaluation and Backtracking: ToT empowers the LLM to self-assess the quality of each thought, evaluating its relevance and contribution to the overall solution. If a thought leads astray, the LLM can backtrack to a previous state and explore alternative paths, ensuring that it remains on track to find the optimal solution.

3. Dynamic Decision-Making: By considering multiple reasoning paths and evaluating their effectiveness, ToT enables the LLM to make informed decisions about the next course of action. This dynamic decision-making process mirrors human adaptability and strategic thinking.

4. Global Reasoning: ToT allows the LLM to consider the overall context of the problem when making decisions, ensuring that its actions align with the ultimate goal. This global reasoning capability is crucial for solving complex problems that require long-term planning and consideration of multiple factors.

The implementation of ToT involves several key components:

1. Prompt Design: Carefully crafted prompts provide the LLM with the necessary context and instructions to guide its reasoning process.

2. State Evaluation Function: This function assesses the quality of each thought, evaluating its progress towards the solution and its consistency with the problem constraints.

3. Backtracking Function: This function enables the LLM to retrace its steps and explore alternative reasoning paths when necessary.

ToT Algorithm: The core algorithm orchestrates the entire process, guiding the LLM through the tree of thoughts, evaluating thoughts, and making decisions about the next course of action.

The benefits of ToT extend beyond problem-solving:

1. Creativity Enhancement: ToT's ability to explore diverse reasoning paths can spark creativity by generating novel ideas and perspectives.

2. Explainability Improvement: The structured nature of ToT facilitates explainability by providing a clear roadmap of the LLM's reasoning process.

3. Generalizability: ToT's principles can be applied to a wide range of tasks, not just problem-solving, making it a versatile tool for LLMs.