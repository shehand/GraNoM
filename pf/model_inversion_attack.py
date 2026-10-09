"""/!\
 * Copyright (c) Shehan Edirimannage, 2025
 *
 * This file is part of the GraNoM project.
 * 
 * Licensed for evaluation and personal testing purposes only.
 * Redistribution, modification, or commercial use of this file,
 * in whole or in part, is strictly prohibited without explicit 
 * written permission from the copyright holder.
 *
 * For license inquiries, contact developers.
 */
Model Inversion Attack Implementation for Federated Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from torchvision import transforms
import json


class ModelInversionAttack:
    """
    Implementation of Model Inversion Attack for federated learning.
    
    This attack attempts to reconstruct training data by optimizing
    input data to maximize the confidence of a target class.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        attack_config: Dict[str, Any] = None
    ):
        """
        Initialize the model inversion attack.
        
        Args:
            model: The target model to attack
            device: Device to run the attack on
            attack_config: Configuration parameters for the attack
        """
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Default attack configuration
        self.config = {
            'num_iterations': 1000,
            'learning_rate': 0.1,
            'batch_size': 1,
            'num_classes': 10,
            'image_size': (1, 28, 28),  # Default for Fashion-MNIST
            'regularization_weight': 0.01,
            'tv_weight': 0.01,  # Total variation regularization
            'l2_weight': 0.01,  # L2 regularization
            'early_stopping_patience': 50,
            'confidence_threshold': 0.9,
            'use_optimization': True,
        }
        
        if attack_config:
            self.config.update(attack_config)

        # Input normalization. Models are trained on normalized inputs, but the
        # reconstruction is optimized in [0,1] pixel space (clamped each step).
        # Feeding [0,1] straight to the model is an input-space mismatch: the model
        # sees out-of-distribution inputs, optimization degenerates, and average
        # target confidence collapses to ~chance (with one dominant class at ~1.0).
        # Normalizing the reconstruction before the forward pass fixes this.
        mean = self.config.get("norm_mean")
        std = self.config.get("norm_std")
        if mean is not None and std is not None:
            c = len(mean)
            self._norm_mean = torch.tensor(mean, device=self.device).view(1, c, 1, 1)
            self._norm_std = torch.tensor(std, device=self.device).view(1, c, 1, 1)
        else:
            self._norm_mean = None
            self._norm_std = None

    def _model_forward(self, dummy_data: torch.Tensor) -> torch.Tensor:
        """Forward pass with the training-time input normalization applied."""
        x = dummy_data
        if self._norm_mean is not None:
            x = (x - self._norm_mean) / self._norm_std
        return self.model(x)
    
    def total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss for image regularization.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Total variation loss
        """
        batch_size = x.size(0)
        h_tv = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        v_tv = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return (h_tv + v_tv) / batch_size
    
    def l2_regularization_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 regularization loss.
        
        Args:
            x: Input tensor
            
        Returns:
            L2 regularization loss
        """
        return torch.norm(x, p=2) ** 2
    
    def model_inversion_loss(
        self,
        dummy_data: torch.Tensor,
        target_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute model inversion loss for a target class.
        
        Args:
            dummy_data: Reconstructed data
            target_class: Target class to invert
            
        Returns:
            Tuple of (loss, confidence)
        """
        # Forward pass (with training-time normalization)
        outputs = self._model_forward(dummy_data)
        confidence = F.softmax(outputs, dim=1)

        # Target confidence for the specific class
        target_confidence = confidence[:, target_class]
        
        # Loss: minimize negative log confidence (maximize confidence)
        loss = -torch.log(target_confidence + 1e-8)
        
        return loss.mean(), target_confidence.mean()
    
    def invert_model_for_class(
        self,
        target_class: int,
        initial_data: Optional[torch.Tensor] = None,
        num_attempts: int = 1
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """
        Invert the model to reconstruct data for a specific class.
        
        Args:
            target_class: Target class to invert
            initial_data: Initial guess for data reconstruction
            num_attempts: Number of attempts with different initializations
            
        Returns:
            Tuple of (reconstructed_data, final_confidence, metrics)
        """
        batch_size = self.config['batch_size']
        image_size = self.config['image_size']
        
        best_data = None
        best_confidence = 0.0
        best_metrics = {}
        
        for attempt in range(num_attempts):
            # Initialize dummy data
            if initial_data is not None and attempt == 0:
                dummy_data = initial_data.clone().detach().requires_grad_(True)
            else:
                # Random initialization
                dummy_data = torch.randn(batch_size, *image_size, device=self.device, requires_grad=True)
            
            # Setup optimizer
            optimizer = optim.Adam([dummy_data], lr=self.config['learning_rate'])
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            metrics = {
                'inversion_loss': [],
                'confidence': [],
                'tv_loss': [],
                'l2_loss': [],
                'total_loss': []
            }
            
            for iteration in range(self.config['num_iterations']):
                optimizer.zero_grad()
                
                # Compute losses
                inv_loss, confidence = self.model_inversion_loss(dummy_data, target_class)
                tv_loss = self.total_variation_loss(dummy_data)
                l2_loss = self.l2_regularization_loss(dummy_data)
                
                # Total loss
                total_loss = (
                    inv_loss +
                    self.config['tv_weight'] * tv_loss +
                    self.config['l2_weight'] * l2_loss
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp data to valid range
                with torch.no_grad():
                    dummy_data.clamp_(0, 1)
                
                # Record metrics
                metrics['inversion_loss'].append(inv_loss.item())
                metrics['confidence'].append(confidence.item())
                metrics['tv_loss'].append(tv_loss.item())
                metrics['l2_loss'].append(l2_loss.item())
                metrics['total_loss'].append(total_loss.item())
                
                # Early stopping
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['early_stopping_patience']:
                    # print(f"Early stopping at iteration {iteration}")
                    break
                
                # Print progress
                # if iteration % 100 == 0:
                #     print(f"Attempt {attempt+1}, Iteration {iteration}: Loss = {total_loss.item():.6f}, Confidence = {confidence.item():.4f}")
            
            # Check if this attempt is better
            if confidence.item() > best_confidence:
                best_confidence = confidence.item()
                best_data = dummy_data.detach().clone()
                best_metrics = metrics
        
        return best_data, best_confidence, best_metrics
    
    def invert_model_for_all_classes(
        self,
        classes_to_invert: Optional[List[int]] = None
    ) -> Dict[int, Tuple[torch.Tensor, float, Dict[str, float]]]:
        """
        Invert the model for multiple classes.
        
        Args:
            classes_to_invert: List of classes to invert (None for all classes)
            
        Returns:
            Dictionary mapping class indices to (data, confidence, metrics)
        """
        if classes_to_invert is None:
            classes_to_invert = list(range(self.config['num_classes']))
        
        results = {}
        
        for target_class in classes_to_invert:
            # print(f"Inverting model for class {target_class}")
            data, confidence, metrics = self.invert_model_for_class(target_class)
            results[target_class] = (data, confidence, metrics)
            # print(f"Class {target_class}: Final confidence = {confidence:.4f}")
        
        return results
    
    def evaluate_inversion_quality(
        self,
        inverted_data: torch.Tensor,
        target_class: int,
        reference_data: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate the quality of model inversion.
        
        Args:
            inverted_data: Inverted data
            target_class: Target class
            reference_data: Reference data for comparison (optional)
            
        Returns:
            Dictionary of quality metrics
        """
        with torch.no_grad():
            # Get model predictions (with training-time normalization)
            outputs = self._model_forward(inverted_data)
            confidence = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(confidence, dim=1)
            
            # Basic metrics
            target_confidence = confidence[:, target_class].mean().item()
            correct_classification = (predicted_class == target_class).float().mean().item()
            
            # Image quality metrics
            if inverted_data.size(1) == 1:  # Grayscale
                mean_intensity = inverted_data.mean().item()
                std_intensity = inverted_data.std().item()
            else:  # RGB
                mean_intensity = inverted_data.mean().item()
                std_intensity = inverted_data.std().item()
            
            # Total variation
            tv = self.total_variation_loss(inverted_data).item()
            
            # L2 norm
            l2_norm = torch.norm(inverted_data, p=2).item()
            
            metrics = {
                'target_confidence': target_confidence,
                'correct_classification': correct_classification,
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'total_variation': tv,
                'l2_norm': l2_norm,
            }
            
            # Compare with reference data if available
            if reference_data is not None:
                mse = F.mse_loss(inverted_data, reference_data).item()
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                metrics['mse_to_reference'] = mse
                metrics['psnr_to_reference'] = psnr
        
        return metrics
    
    def visualize_inversion_results(
        self,
        inversion_results: Dict[int, Tuple[torch.Tensor, float, Dict[str, float]]],
        save_path: Optional[str] = None,
        num_classes_to_show: int = 10
    ):
        """
        Visualize the model inversion results.
        
        Args:
            inversion_results: Results from invert_model_for_all_classes
            save_path: Path to save the visualization
            num_classes_to_show: Number of classes to show
        """
        classes = list(inversion_results.keys())[:num_classes_to_show]
        num_classes = len(classes)
        
        fig, axes = plt.subplots(1, num_classes, figsize=(2*num_classes, 3))
        if num_classes == 1:
            axes = [axes]
        
        for i, class_idx in enumerate(classes):
            data, confidence, _ = inversion_results[class_idx]
            
            # Display the inverted image
            if data.size(1) == 1:  # Grayscale
                axes[i].imshow(data[0, 0].cpu().numpy(), cmap='gray')
            else:  # RGB
                axes[i].imshow(data[0].permute(1, 2, 0).cpu().numpy())
            
            axes[i].set_title(f'Class {class_idx}\nConf: {confidence:.3f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_attack_results(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and aggregate model inversion attack results.
        
        Args:
            attack_results: Results from perform_attack_on_global_model
            
        Returns:
            Aggregated analysis results
        """
        quality_metrics = attack_results['quality_metrics']
        
        if not quality_metrics:
            return {'error': 'No quality metrics available'}
        
        # Aggregate metrics across all classes
        confidences = [metrics['target_confidence'] for metrics in quality_metrics.values()]
        correct_classifications = [metrics['correct_classification'] for metrics in quality_metrics.values()]
        total_variations = [metrics['total_variation'] for metrics in quality_metrics.values()]
        l2_norms = [metrics['l2_norm'] for metrics in quality_metrics.values()]
        
        analysis = {
            'num_classes_inverted': len(quality_metrics),
            'average_target_confidence': np.mean(confidences),
            'average_correct_classification': np.mean(correct_classifications),
            'average_total_variation': np.mean(total_variations),
            'average_l2_norm': np.mean(l2_norms),
            'std_target_confidence': np.std(confidences),
            'std_correct_classification': np.std(correct_classifications),
            'std_total_variation': np.std(total_variations),
            'std_l2_norm': np.std(l2_norms),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
        }
        
        return analysis


class FederatedModelInversionAttack:
    """
    Wrapper class for performing model inversion attacks in federated learning settings.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize the federated model inversion attack.
        
        Args:
            model: The model architecture
            device: Device to run attacks on
        """
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attack = ModelInversionAttack(model, self.device)
    
    def perform_attack_on_global_model(
        self,
        global_parameters: List[torch.Tensor],
        classes_to_invert: Optional[List[int]] = None,
        attack_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform model inversion attack on the global model.
        
        Args:
            global_parameters: Global model parameters
            classes_to_invert: Classes to invert
            attack_config: Attack configuration
            
        Returns:
            Dictionary containing attack results
        """
        # Update model with global parameters
        from pf.task import set_weights
        try:
            set_weights(self.model, global_parameters)
            # Ensure model is in evaluation mode and on correct device
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to set weights: {e}")
            raise e
        
        # Update attack configuration
        if attack_config:
            self.attack.config.update(attack_config)
        
        # Perform inversion
        inversion_results = self.attack.invert_model_for_all_classes(classes_to_invert)
        
        # Evaluate quality for each class
        quality_metrics = {}
        for class_idx, (data, confidence, metrics) in inversion_results.items():
            quality = self.attack.evaluate_inversion_quality(data, class_idx)
            quality_metrics[class_idx] = quality
        
        return {
            'inversion_results': inversion_results,
            'quality_metrics': quality_metrics,
            'global_parameters': global_parameters
        }
    
    
    
    def get_attack_summary_json(self, attack_results: Dict[str, Any]) -> str:
        """
        Get a JSON summary of the attack results for transmission.
        
        Args:
            attack_results: Results from perform_attack_on_global_model
            
        Returns:
            JSON string summary
        """
        analysis = self.analyze_attack_results(attack_results)
        
        # Create a simplified summary
        summary = {
            'attack_type': 'model_inversion',
            'num_classes_inverted': analysis['num_classes_inverted'],
            'average_target_confidence': analysis['average_target_confidence'],
            'average_correct_classification': analysis['average_correct_classification'],
            'max_confidence': analysis['max_confidence'],
            'min_confidence': analysis['min_confidence'],
        }
        
        return json.dumps(summary) 