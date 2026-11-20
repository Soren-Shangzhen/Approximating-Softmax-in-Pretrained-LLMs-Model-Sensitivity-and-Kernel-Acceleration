"""Code for "Approximating Softmax in Pretrained LLMs: Model Sensitivity and Kernel Acceleration".

Modules
  attention          attention backend: softmax and the Section 4 interventions on the score-to-probability map
  model_loader       Hugging Face loading with the backend registered (one GPU); multigpu/ for the 70B/72B models
  data               WikiText-103 evaluation token stream
  characterization   evaluation pipeline and tasks T00-T13 (Section 4, Appendices A, C, D)
  lattice_operators  Section 5 / Appendix B operators: absolute and row-maximum anchors, Rowmax-H15 semantics,
                     exponential maps
  stats              paired block bootstrap and perplexity change (numpy only)
"""
__version__ = "1.0.0"
