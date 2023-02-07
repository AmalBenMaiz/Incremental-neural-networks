# Incremental-neural-networks
Implementation of constructive methods (incremental) of neural networks namely I-ELM, EM-ELM and TLFN in order to have high accuracy and fast training time in classification problems 
This project includes training and inference phases of three incremental NN models named (I-ELM, EM-ELM, and TLFN).
Training phase: Simulation using Python software of the proposed incremental methods: 
      The python script contains the main program "main-prog.py"that gets as input a database chosen by the ”make_data.py” class 
      and an incremental model among "I_ELM.py", "EM_ELM.py", and "TLFN.py" classes. This program gives CSV files containing simulation 
      results such as accuracy rate, loss rate, training time,and epochs. "plotting.py" can be used to plot evaluation results of the models.
Inference phase: codes can be used to implement the proposed incremental methods on a Raspberry board:
      - "Inference_ELM.py": choose to connect with "I_ELM" or "EM_ELM".
      - "Inference_TLFN.py".
