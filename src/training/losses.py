# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss(logits, labels):
    """
    Computes the standard cross-entropy loss.

    Args:
        logits (torch.Tensor): Raw output from the model (batch_size, num_classes).
        labels (torch.Tensor): Ground truth labels (batch_size).

    Returns:
        torch.Tensor: The cross-entropy loss.
    """
    return F.cross_entropy(logits, labels)

def l1_similarity_loss(output_logits, target_logits_detached):
    """
    Computes the L1 similarity loss between the softmax probabilities of two sets of logits.
    This encourages the output_logits to remain similar to the target_logits (e.g., ZS-CLIP logits).

    Args:
        output_logits (torch.Tensor): Logits from a model branch (e.g., s_MFA, s_IRA).
        target_logits_detached (torch.Tensor): Target logits to compare against,
                                               detached from the computation graph
                                               (e.g., s_ZS.detach()).

    Returns:
        torch.Tensor: The L1 similarity loss.
    """
    prob_output = F.softmax(output_logits, dim=-1)
    prob_target = F.softmax(target_logits_detached, dim=-1)
    return F.l1_loss(prob_output, prob_target)

def compute_rac_model_losses(model_outputs, labels, config):
    """
    Computes the overall loss components based on the RACModel's output dictionary.
    This function essentially replicates the loss calculation logic from RACModel.forward()
    and might be used if you prefer to calculate losses outside the model's forward pass,
    though it's often more convenient to have it within the model forward for clarity.

    Args:
        model_outputs (dict): A dictionary containing various logits like
                              's_ZS', 's_MFA', 's_IRA', 's_ALF'.
        labels (torch.Tensor): Ground truth labels.
        config (dict): Configuration dictionary containing lambda_tradeoff_similarity.

    Returns:
        tuple:
            - total_loss (torch.Tensor): The final combined loss.
            - loss_dict (dict): Dictionary of individual loss components.
    """
    loss_dict = {}
    lambda_sim = config.get('lambda_tradeoff_similarity', 1.0)

    # Cross-entropy losses
    if 's_MFA' in model_outputs:
        loss_dict['ce_mfa'] = cross_entropy_loss(model_outputs['s_MFA'], labels)
    if 's_IRA' in model_outputs:
        loss_dict['ce_ira'] = cross_entropy_loss(model_outputs['s_IRA'], labels)
    if 's_ALF' in model_outputs: # Final logits
        loss_dict['ce_alf'] = cross_entropy_loss(model_outputs['s_ALF'], labels)
    else: # Fallback if s_ALF is not present for some reason
        raise ValueError("s_ALF (final_logits) not found in model_outputs for CE loss calculation.")


    total_ce_loss = sum(loss_dict[k] for k in loss_dict if k.startswith('ce_'))
    loss_dict['total_ce'] = total_ce_loss

    # Similarity losses
    # s_ZS must be present and detached if used as target
    s_ZS_detached = model_outputs.get('s_ZS_detached', model_outputs.get('s_ZS', torch.zeros_like(model_outputs['s_ALF'])).detach())


    loss_sim_mfa = torch.tensor(0.0, device=labels.device)
    if 's_MFA' in model_outputs and 's_ZS' in model_outputs:
        loss_sim_mfa = l1_similarity_loss(model_outputs['s_MFA'], s_ZS_detached)
        loss_dict['sim_mfa'] = loss_sim_mfa

    loss_sim_ira = torch.tensor(0.0, device=labels.device)
    if 's_IRA' in model_outputs and 's_ZS' in model_outputs:
        loss_sim_ira = l1_similarity_loss(model_outputs['s_IRA'], s_ZS_detached)
        loss_dict['sim_ira'] = loss_sim_ira

    total_sim_loss = loss_sim_mfa + loss_sim_ira
    loss_dict['total_sim'] = total_sim_loss

    # Total loss
    total_loss = total_ce_loss + lambda_sim * total_sim_loss
    loss_dict['total_loss'] = total_loss

    if 'alpha' in model_outputs: # Log mean alpha if available
        loss_dict['alpha_mean'] = model_outputs['alpha'].mean()

    return total_loss, loss_dict


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    print("--- Testing Loss Functions ---")
    batch_s = 4
    num_cls = 5

    # Dummy data
    dummy_logits1 = torch.randn(batch_s, num_cls, requires_grad=True)
    dummy_logits2 = torch.randn(batch_s, num_cls) # Target, detached
    dummy_labels = torch.randint(0, num_cls, (batch_s,))
    dummy_config = {'lambda_tradeoff_similarity': 0.5}

    # Test cross_entropy_loss
    ce_loss = cross_entropy_loss(dummy_logits1, dummy_labels)
    print(f"Cross Entropy Loss: {ce_loss.item()}")
    assert ce_loss.requires_grad

    # Test l1_similarity_loss
    sim_loss = l1_similarity_loss(dummy_logits1, dummy_logits2.detach())
    print(f"L1 Similarity Loss: {sim_loss.item()}")
    assert sim_loss.requires_grad


    print("\n--- Testing compute_rac_model_losses (if losses computed outside model) ---")
    # This example demonstrates how compute_rac_model_losses would be used
    # if the model forward pass returned a dictionary of all logits.
    mock_model_outputs = {
        's_ZS': torch.randn(batch_s, num_cls),
        's_MFA': torch.randn(batch_s, num_cls, requires_grad=True),
        's_IRA': torch.randn(batch_s, num_cls, requires_grad=True),
        's_ALF': torch.randn(batch_s, num_cls, requires_grad=True), # Final logits
        'alpha': torch.rand(batch_s, 1)
    }
    # Detach s_ZS when using it as a target for similarity loss
    mock_model_outputs['s_ZS_detached'] = mock_model_outputs['s_ZS'].detach()


    total_l, l_dict = compute_rac_model_losses(mock_model_outputs, dummy_labels, dummy_config)
    print(f"Computed Total Loss: {total_l.item()}")
    print("Individual Loss Components:")
    for k, v in l_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item()}")
        else:
            print(f"  {k}: {v}") # For non-tensor values like alpha_mean
    assert total_l.requires_grad
    assert 'ce_alf' in l_dict
    assert 'total_ce' in l_dict
    assert 'total_sim' in l_dict

    print("\nLoss functions test completed.")