"""IFEval instruction checkers, vendored from google-research/instruction_following_eval
(Apache-2.0, files carry their own headers). Changes: package-relative imports, and the
sentence tokenizer loads punkt_tab (nltk >= 3.9 refuses the pickled punkt).
Deps: absl-py, langdetect, immutabledict, nltk."""
