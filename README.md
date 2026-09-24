This contains the key implementation components and prompts used in our proposed FLAG framework.
All files included here demonstrate the core innovations of the method.

1. Agent_statement_induction.py

--This script implements the statement induction module. It calls the DeepSeek-V3 API to iteratively induce atomic factual statements from labeled and pseudo-labeled student answers.

--Input:
   -Correct and incorrect labeled samples
   -Unlabeled samples

--Outputs:
   -Structured statement set:
	{
	"support": [s1, s2, s3, ...],
	"contradict": [s1, s2, ...],
	"neutral": [s1, s2, ...]
	}
   -Cluster index mapping:
	{
	"support": [0, 0, 1, ...],
	"contradict": [0, 1, ...],
	"neutral": [0, 0, ...]
	}


2. Agent_logical_alignment.py

--This script implements the logical alignment process based on the induced statements.
It aligns the logical expressions of unlabeled texts with those of labeled templates and assigns pseudo-labels accordingly.


3. other_prompts.pdf

--This file contains all prompts used throughout the FLAG framework, including:


Usage notice:

Replace all model and data paths with your local files.

Ensure that your API endpoint and key configuration match your runtime environment.

The provided code and prompts are intended for reproducibility and demonstration purposes.
