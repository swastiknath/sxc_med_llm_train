"""
Copyright (C) 2024 Swastik Nath.
@author: SwastikN
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_transformer_outputs(model1_name, model2_name, input_text, ground_truth_label):
    """
    Evaluate transformer model outputs against ground truth.

    Args:
        model1_name (str): Name of the first transformer model (e.g., 'bert-base-uncased').
        model2_name (str): Name of the second transformer model.
        input_text (str): Input text for which we want to generate outputs.
        ground_truth_label (str): Ground truth label corresponding to the input text.

    Returns:
        float: Evaluation score (e.g., accuracy, F1-score, etc.).
    """
    # Load tokenizer and models
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    model1 = AutoModelForSequenceClassification.from_pretrained(model1_name)

    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    model2 = AutoModelForSequenceClassification.from_pretrained(model2_name)

    # Tokenize input text
    inputs = tokenizer1(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate outputs from both models
    with torch.no_grad():
        output1 = model1(input_ids, attention_mask=attention_mask).logits
        output2 = model2(input_ids, attention_mask=attention_mask).logits

    # Compare outputs with ground truth
    predicted_label1 = torch.argmax(output1, dim=1).item()
    predicted_label2 = torch.argmax(output2, dim=1).item()

    # Evaluate the generation (you can use any metric here)
    correct1 = predicted_label1 == ground_truth_label
    correct2 = predicted_label2 == ground_truth_label

    # Example: Accuracy
    accuracy1 = int(correct1)
    accuracy2 = int(correct2)

    return accuracy1, accuracy2


import numpy as np

def generate_competitive_rewards(agent_scores, opponent_scores):
    """
    Generates reward vectors based on competitive scores.

    Args:
        agent_scores (numpy.ndarray): Array of scores obtained by the agent.
        opponent_scores (numpy.ndarray): Array of scores obtained by opponents.

    Returns:
        numpy.ndarray: Reward vector for the agent.
    """
    assert len(agent_scores) == len(opponent_scores), "Scores arrays must have the same length"

    # Calculate relative performance (agent score - opponent score)
    relative_performance = agent_scores - opponent_scores

    # Assign rewards based on relative performance
    rewards = np.where(relative_performance > 0, 1.0, 0.0)  # Win = 1, Lose = 0

    return rewards


import torch
from transformers import AutoModelForSequenceGeneration, AutoTokenizer

def calculate_perplexity(logits, target):
    """
    Calculates perplexity from model logits and target labels.

    Args:
        logits (torch.Tensor): Model output logits (shape: batch_size x seq_length x vocab_size).
        target (torch.Tensor): Target labels (shape: batch_size x seq_length).

    Returns:
        float: Perplexity value.
    """
    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='sum')

    # Calculate perplexity
    num_tokens = target.numel()
    perplexity = torch.exp(loss / num_tokens)

    return perplexity.item()

def environment_update_policy_weight(model1_name, model2_name, input_text):
    # Load tokenizer and models
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    model1 = AutoModelForSequenceGeneration.from_pretrained(model1_name)

    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    model2 = AutoModelForSequenceGeneration.from_pretrained(model2_name)

    # Generate outputs from both models
    with torch.no_grad():
        input_ids = tokenizer1.encode(input_text, return_tensors="pt")
        output1 = model1.generate(input_ids)
        output2 = model2.generate(input_ids)

    # Calculate perplexity for each output
    target = input_ids  # Assuming target is the same as input for simplicity
    perplexity1 = calculate_perplexity(output1, target)
    perplexity2 = calculate_perplexity(output2, target)

    # Apply rewards based on perplexity (lower perplexity is better)
    if perplexity1 < perplexity2: 
       return model1
    else:
        

        return model2


