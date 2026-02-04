This project shows that I can train and evaluate a deep learning text classifier using a pretrained transformer, even on a CPU, in a disciplined and production-style way.

In slightly more detail (still simple):

We take a pretrained language model (DistilBERT)

Adapt it to a text classification task

Train it efficiently using transfer learning

Evaluate it properly using classification metrics

Analyze where and why the model makes mistakes