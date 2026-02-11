"""
Hybrid Prompt Generation Module

This module generates detailed text prompts for ControlNet training.
According to PROJECT(prepare_control).md:

Structure: [Sub-class characteristics] + [Background type] + [Surface condition]
Example: "A high-linearity scratch on a vertical striped metal surface with smooth texture."

The prompt combines defect subtype with background context to guide ControlNet.
"""
from typing import Dict, List


class PromptGenerator:
    """
    Generates hybrid prompts combining defect and background information.
    """
    
    # Defect subtype descriptions with characteristics
    DEFECT_DESCRIPTIONS = {
        'linear_scratch': {
            'base': 'a linear scratch defect',
            'detailed': 'a high-linearity elongated scratch',
            'characteristics': ['linear', 'scratch-like', 'elongated']
        },
        'elongated': {
            'base': 'an elongated defect',
            'detailed': 'a moderately elongated defect region',
            'characteristics': ['elongated', 'stretched']
        },
        'compact_blob': {
            'base': 'a compact blob defect',
            'detailed': 'a solid compact defect spot',
            'characteristics': ['compact', 'blob-like', 'solid']
        },
        'irregular': {
            'base': 'an irregular defect',
            'detailed': 'an irregular defect with complex boundaries',
            'characteristics': ['irregular', 'complex-shaped']
        },
        'general': {
            'base': 'a defect',
            'detailed': 'a general surface defect',
            'characteristics': ['defect']
        }
    }
    
    # Background type descriptions
    BACKGROUND_DESCRIPTIONS = {
        'smooth': {
            'surface': 'smooth metal surface',
            'texture': 'uniform texture',
            'pattern': 'no visible pattern'
        },
        'textured': {
            'surface': 'textured metal surface',
            'texture': 'grainy texture',
            'pattern': 'subtle surface texture'
        },
        'vertical_stripe': {
            'surface': 'vertical striped metal surface',
            'texture': 'directional texture',
            'pattern': 'vertical line pattern'
        },
        'horizontal_stripe': {
            'surface': 'horizontal striped metal surface',
            'texture': 'directional texture',
            'pattern': 'horizontal line pattern'
        },
        'complex_pattern': {
            'surface': 'complex patterned metal surface',
            'texture': 'multi-directional texture',
            'pattern': 'complex surface pattern'
        }
    }
    
    # Surface quality modifiers based on stability score
    SURFACE_QUALITY = {
        'high': ['pristine', 'well-maintained', 'clean'],
        'medium': ['standard', 'typical', 'normal'],
        'low': ['worn', 'weathered', 'aged']
    }
    
    def __init__(self, style: str = 'detailed', include_class_id: bool = True):
        """
        Initialize prompt generator.
        
        Args:
            style: 'simple', 'detailed', or 'technical'
            include_class_id: Whether to include class ID in prompt
        """
        self.style = style
        self.include_class_id = include_class_id
    
    def get_surface_quality(self, stability_score: float) -> str:
        """
        Get surface quality description based on stability score.
        
        Args:
            stability_score: Background stability (0-1)
            
        Returns:
            Surface quality descriptor
        """
        if stability_score >= 0.8:
            quality = 'high'
        elif stability_score >= 0.5:
            quality = 'medium'
        else:
            quality = 'low'
        
        import random
        return random.choice(self.SURFACE_QUALITY[quality])
    
    def generate_simple_prompt(self, defect_subtype: str, 
                               background_type: str,
                               class_id: int) -> str:
        """
        Generate simple prompt.
        
        Args:
            defect_subtype: Defect classification
            background_type: Background classification
            class_id: Defect class (1-4)
            
        Returns:
            Simple prompt string
        """
        defect_desc = self.DEFECT_DESCRIPTIONS.get(
            defect_subtype, self.DEFECT_DESCRIPTIONS['general']
        )['base']
        
        bg_desc = self.BACKGROUND_DESCRIPTIONS.get(
            background_type, self.BACKGROUND_DESCRIPTIONS['smooth']
        )['surface']
        
        prompt = f"{defect_desc} on {bg_desc}"
        
        if self.include_class_id:
            prompt += f", class {class_id}"
        
        return prompt
    
    def generate_detailed_prompt(self, defect_subtype: str,
                                 background_type: str,
                                 class_id: int,
                                 stability_score: float,
                                 defect_metrics: Dict) -> str:
        """
        Generate detailed prompt with more context.
        
        Args:
            defect_subtype: Defect classification
            background_type: Background classification
            class_id: Defect class (1-4)
            stability_score: Background stability score
            defect_metrics: Defect characterization metrics
            
        Returns:
            Detailed prompt string
        """
        defect_desc = self.DEFECT_DESCRIPTIONS.get(
            defect_subtype, self.DEFECT_DESCRIPTIONS['general']
        )['detailed']
        
        bg_info = self.BACKGROUND_DESCRIPTIONS.get(
            background_type, self.BACKGROUND_DESCRIPTIONS['smooth']
        )
        
        surface_quality = self.get_surface_quality(stability_score)
        
        # Build prompt components
        prompt_parts = [defect_desc, "on", bg_info['surface']]
        
        # Add texture information
        prompt_parts.extend(["with", bg_info['texture']])
        
        # Add surface quality
        if stability_score >= 0.6:
            prompt_parts.extend([f"({surface_quality} condition)"])
        
        prompt = " ".join(prompt_parts)
        
        if self.include_class_id:
            prompt += f", steel defect class {class_id}"
        
        return prompt
    
    def generate_technical_prompt(self, defect_subtype: str,
                                  background_type: str,
                                  class_id: int,
                                  stability_score: float,
                                  defect_metrics: Dict,
                                  suitability_score: float) -> str:
        """
        Generate technical prompt with quantitative information.
        
        Args:
            defect_subtype: Defect classification
            background_type: Background classification
            class_id: Defect class (1-4)
            stability_score: Background stability score
            defect_metrics: Defect characterization metrics
            suitability_score: ROI suitability score
            
        Returns:
            Technical prompt string
        """
        defect_info = self.DEFECT_DESCRIPTIONS.get(
            defect_subtype, self.DEFECT_DESCRIPTIONS['general']
        )
        
        bg_info = self.BACKGROUND_DESCRIPTIONS.get(
            background_type, self.BACKGROUND_DESCRIPTIONS['smooth']
        )
        
        # Extract key metrics
        linearity = defect_metrics.get('linearity', 0.0)
        solidity = defect_metrics.get('solidity', 0.0)
        aspect_ratio = defect_metrics.get('aspect_ratio', 1.0)
        
        # Build technical description
        characteristics = []
        
        if linearity > 0.8:
            characteristics.append("highly linear")
        elif linearity > 0.6:
            characteristics.append("moderately linear")
        
        if solidity > 0.9:
            characteristics.append("solid")
        elif solidity < 0.7:
            characteristics.append("irregular shape")
        
        if aspect_ratio > 5.0:
            characteristics.append("very elongated")
        elif aspect_ratio > 3.0:
            characteristics.append("elongated")
        
        char_str = ", ".join(characteristics) if characteristics else defect_info['characteristics'][0]
        
        prompt = (
            f"Industrial steel defect: {char_str} defect "
            f"(class {class_id}) on {bg_info['surface']}, "
            f"{bg_info['pattern']}, "
            f"background stability {stability_score:.2f}, "
            f"match quality {suitability_score:.2f}"
        )
        
        return prompt
    
    def generate_prompt(self, defect_subtype: str,
                       background_type: str,
                       class_id: int,
                       stability_score: float = 0.5,
                       defect_metrics: Dict = None,
                       suitability_score: float = 0.5) -> str:
        """
        Generate prompt based on configured style.
        
        Args:
            defect_subtype: Defect classification
            background_type: Background classification
            class_id: Defect class (1-4)
            stability_score: Background stability score
            defect_metrics: Defect characterization metrics
            suitability_score: ROI suitability score
            
        Returns:
            Generated prompt string
        """
        if defect_metrics is None:
            defect_metrics = {}
        
        if self.style == 'simple':
            return self.generate_simple_prompt(defect_subtype, background_type, class_id)
        
        elif self.style == 'technical':
            return self.generate_technical_prompt(
                defect_subtype, background_type, class_id,
                stability_score, defect_metrics, suitability_score
            )
        
        else:  # detailed (default)
            return self.generate_detailed_prompt(
                defect_subtype, background_type, class_id,
                stability_score, defect_metrics
            )
    
    def generate_negative_prompt(self) -> str:
        """
        Generate negative prompt for ControlNet training.
        
        Returns:
            Negative prompt string
        """
        negative_prompts = [
            "blurry, low quality, artifacts, noise",
            "distorted, warped, unrealistic",
            "oversaturated, cartoon, painting",
            "text, watermark, logo"
        ]
        
        return ", ".join(negative_prompts)
    
    def batch_generate_prompts(self, roi_metadata: List[Dict]) -> List[Dict]:
        """
        Generate prompts for a batch of ROIs.
        
        Args:
            roi_metadata: List of ROI metadata dictionaries
            
        Returns:
            Updated list with 'prompt' and 'negative_prompt' fields
        """
        for roi_data in roi_metadata:
            prompt = self.generate_prompt(
                defect_subtype=roi_data.get('defect_subtype', 'general'),
                background_type=roi_data.get('background_type', 'smooth'),
                class_id=roi_data.get('class_id', 1),
                stability_score=roi_data.get('stability_score', 0.5),
                defect_metrics=roi_data,
                suitability_score=roi_data.get('suitability_score', 0.5)
            )
            
            roi_data['prompt'] = prompt
            roi_data['negative_prompt'] = self.generate_negative_prompt()
        
        return roi_metadata
