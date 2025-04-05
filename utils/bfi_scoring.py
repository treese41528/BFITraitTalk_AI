# utils/bfi_scoring.py

import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BFIScorer:
    """
    Handles scoring and interpretation of Big Five Inventory responses.
    """
    
    # The five traits
    TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    # Descriptive labels for different levels
    LEVEL_LABELS = {
        "very_low": "Very Low",
        "low": "Low",
        "moderate": "Moderate",
        "high": "High",
        "very_high": "Very High"
    }
    
    def __init__(self, questions_data):
        """
        Initialize with BFI questions data.
        
        Args:
            questions_data: List of question objects with trait and reverse information
        """
        self.questions = questions_data
        self.questions_by_id = {str(q['id']): q for q in questions_data} 
        
        # Extract trait information
        self.trait_questions = {}
        for trait in self.TRAITS:
            self.trait_questions[trait] = [
                q for q in questions_data if q.get('trait') == trait
            ]
        
        logger.info(f"BFI Scorer initialized with {len(questions_data)} questions")
        for trait in self.TRAITS:
            logger.info(f"  {trait}: {len(self.trait_questions[trait])} questions")
    
    def score_trait(self, trait: str, answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a specific trait based on the provided answers.
        
        Args:
            trait: Name of the trait to score
            answers: Dictionary mapping question IDs to answer values
            
        Returns:
            Dict: Scoring results for the trait
        """
        if trait not in self.TRAITS:
            logger.warning(f"Invalid trait: {trait}")
            return {
                "trait": trait,
                "score": 0,
                "raw_score": 0,
                "count": 0,
                "possible": 0,
                "level": None,
                "percentage": 0
            }
            
        trait_questions = self.trait_questions[trait]
        if not trait_questions:
            logger.warning(f"No questions found for trait: {trait}")
            return {
                "trait": trait,
                "score": 0,
                "raw_score": 0,
                "count": 0,
                "possible": 0,
                "level": None,
                "percentage": 0
            }
            
        # Calculate the score
        total = 0
        count = 0
        total_possible = len(trait_questions) * 5  # Max score is 5 per question
        
        for question in trait_questions:
            q_id = str(question['id'])
            if q_id in answers:
                answer = answers[q_id]
                
                # Skip questions that were skipped
                if answer == 'skipped':
                    continue
                    
                # Convert to integer if stored as string
                if isinstance(answer, str) and answer.isdigit():
                    answer = int(answer)
                    
                # Handle reverse-scored items
                if question.get('reverse', False):
                    value = 6 - answer  # 5 -> 1, 4 -> 2, 3 -> 3, 2 -> 4, 1 -> 5
                else:
                    value = answer
                    
                total += value
                count += 1
        
        # Calculate the average score (1-5 scale)
        score = 0
        if count > 0:
            score = round(total / count, 2)
            
        # Calculate percentage of maximum possible
        percentage = 0
        if count > 0:
            percentage = round((total / (count * 5)) * 100, 1)
            
        # Determine level based on score ranges
        level = self._determine_level(score)
        
        return {
            "trait": trait,
            "score": score,  # Average score (1-5 scale)
            "raw_score": total,  # Sum of all scores
            "count": count,  # Number of questions answered
            "possible": len(trait_questions),  # Total possible questions
            "level": level,  # Descriptive level
            "percentage": percentage  # Percentage of maximum possible
        }
    
    def _determine_level(self, score: float) -> str:
        """
        Determine the descriptive level based on the score.
        
        Args:
            score: Numeric score (1-5 scale)
            
        Returns:
            str: Descriptive level
        """
        if score < 1.5:
            return self.LEVEL_LABELS["very_low"]
        elif score < 2.5:
            return self.LEVEL_LABELS["low"]
        elif score < 3.5:
            return self.LEVEL_LABELS["moderate"]
        elif score < 4.5:
            return self.LEVEL_LABELS["high"]
        else:
            return self.LEVEL_LABELS["very_high"]
    
    def calculate_bfi_scores(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate scores for all five traits.
        
        Args:
            answers: Dictionary mapping question IDs to answer values
            
        Returns:
            Dict: Complete BFI scoring results
        """
        # Initialize results
        results = {
            "traits": {},
            "summary": {
                "questions_answered": 0,
                "questions_total": len(self.questions),
                "completion_percentage": 0
            }
        }
        
        # Score each trait
        for trait in self.TRAITS:
            trait_result = self.score_trait(trait, answers)
            results["traits"][trait.lower()] = trait_result
            results["summary"]["questions_answered"] += trait_result["count"]
            
        # Calculate completion percentage
        if results["summary"]["questions_total"] > 0:
            results["summary"]["completion_percentage"] = round(
                (results["summary"]["questions_answered"] / results["summary"]["questions_total"]) * 100, 1
            )
            
        return results
    
    def get_trait_interpretations(self, trait: str, level: str) -> Dict[str, str]:
        """
        Get explanatory text for a trait level.
        
        Args:
            trait: Trait name
            level: Trait level descriptor
            
        Returns:
            Dict: Interpretation texts
        """
        # Define interpretations for each trait and level
        interpretations = {
            "Openness": {
                self.LEVEL_LABELS["very_low"]: "You strongly prefer the conventional and familiar over new experiences or abstract thinking. You likely focus on concrete, practical matters rather than imagination or artistic pursuits.",
                self.LEVEL_LABELS["low"]: "You tend to be practical and traditional rather than exploratory or creative. You prefer the familiar over the new or unusual.",
                self.LEVEL_LABELS["moderate"]: "You balance traditional approaches with some willingness to try new experiences. You appreciate both practical matters and occasional creative or abstract thinking.",
                self.LEVEL_LABELS["high"]: "You are curious and appreciative of art, new ideas, and varied experiences. You tend to be more creative and aware of your feelings than most people.",
                self.LEVEL_LABELS["very_high"]: "You have an exceptional appreciation for discovery, variety, and creativity. You're likely to have broad interests and seek out novel experiences. You tend to be very imaginative and intellectually curious."
            },
            "Conscientiousness": {
                self.LEVEL_LABELS["very_low"]: "You tend to be very spontaneous and flexible rather than organized and detail-oriented. You may find systematic approaches restrictive and prefer to approach tasks as they come.",
                self.LEVEL_LABELS["low"]: "You tend to be relaxed and somewhat disorganized. You may find rules, planning, and schedules limiting or unnecessary.",
                self.LEVEL_LABELS["moderate"]: "You balance being organized with being flexible. You can follow plans and be reliable, but you don't require rigid structure.",
                self.LEVEL_LABELS["high"]: "You are organized and mindful of details. You plan ahead, consider consequences, and prefer to complete tasks thoroughly.",
                self.LEVEL_LABELS["very_high"]: "You are exceptionally disciplined, dutiful, and organized. You strive for achievement and are highly self-motivated to accomplish goals. You're likely to be thorough, reliable, and persistent."
            },
            "Extraversion": {
                self.LEVEL_LABELS["very_low"]: "You strongly prefer solitude and quiet environments. Social gatherings likely drain your energy significantly, and you're most comfortable either alone or with a few close friends.",
                self.LEVEL_LABELS["low"]: "You tend to be more reserved and prefer smaller, intimate gatherings to large social events. You value your independence and time for reflection.",
                self.LEVEL_LABELS["moderate"]: "You can enjoy social activities but also value your alone time. Your energy levels in social situations depend on the context and the people involved.",
                self.LEVEL_LABELS["high"]: "You are outgoing and draw energy from social interactions. You prefer being with others rather than alone and find it easy to approach new people.",
                self.LEVEL_LABELS["very_high"]: "You are exceptionally sociable, enthusiastic, and action-oriented. You thrive in group settings, readily take charge, and enjoy being the center of attention. Social interaction energizes you significantly."
            },
            "Agreeableness": {
                self.LEVEL_LABELS["very_low"]: "You tend to be very direct, blunt, and competitive rather than accommodating. You're skeptical of others' motives and may prioritize your own interests above group harmony.",
                self.LEVEL_LABELS["low"]: "You tend to be critical and question others' intentions. You're not afraid to express disagreement and may come across as challenging at times.",
                self.LEVEL_LABELS["moderate"]: "You balance cooperation with standing up for yourself. You can be warm and considerate but also assert your needs when important.",
                self.LEVEL_LABELS["high"]: "You are generally warm, friendly, considerate, and helpful to others. You believe in the best of people and value getting along.",
                self.LEVEL_LABELS["very_high"]: "You are exceptionally compassionate, trusting, and eager to help others. You consistently place group harmony and others' needs as high priorities. You naturally see the best in people and avoid conflict whenever possible."
            },
            "Neuroticism": {
                self.LEVEL_LABELS["very_low"]: "You are exceptionally calm, composed, and resilient under stress. You rarely experience negative emotions and tend to be very emotionally stable even in challenging situations.",
                self.LEVEL_LABELS["low"]: "You are generally relaxed and handle stress well. You tend to remain calm in tense situations and recover quickly from setbacks.",
                self.LEVEL_LABELS["moderate"]: "You experience normal ups and downs in your emotional life. You feel negative emotions when appropriate but can manage and bounce back from them.",
                self.LEVEL_LABELS["high"]: "You tend to experience negative emotions more readily than most people. You may worry more often and be more sensitive to stress and criticism.",
                self.LEVEL_LABELS["very_high"]: "You frequently experience strong negative emotions such as anxiety, sadness, or irritability. You may be highly sensitive to stress, find it difficult to cope with daily pressures, and take longer to recover from emotional setbacks."
            }
        }
        
        # Get the appropriate interpretation
        if trait in interpretations and level in interpretations[trait]:
            interpretation = interpretations[trait][level]
        else:
            interpretation = "No specific interpretation available for this trait level."
            
        return {
            "trait": trait,
            "level": level,
            "interpretation": interpretation
        }
    
    def generate_comprehensive_report(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive BFI report with scores and interpretations.
        
        Args:
            answers: Dictionary mapping question IDs to answer values
            
        Returns:
            Dict: Complete report with scores and interpretations
        """
        # Calculate the scores
        scores = self.calculate_bfi_scores(answers)
        
        # Add interpretations
        report = {
            "scores": scores,
            "interpretations": {},
            "timestamp": None,  # To be filled by caller if needed
            "summary": scores["summary"]
        }
        
        # Generate interpretations for each trait
        for trait_key, trait_data in scores["traits"].items():
            trait_name = trait_key.capitalize()
            # Find the full trait name in our TRAITS list (case-insensitive match)
            full_trait_name = next((t for t in self.TRAITS if t.lower() == trait_key.lower()), trait_name)
            
            level = trait_data["level"]
            interpretation = self.get_trait_interpretations(full_trait_name, level)
            report["interpretations"][trait_key] = interpretation
            
        return report

# For direct testing
if __name__ == "__main__":
    # Sample BFI questions for testing
    test_questions = [
        {"id": 1, "text": "is talkative", "trait": "Extraversion", "reverse": False},
        {"id": 2, "text": "tends to find fault with others", "trait": "Agreeableness", "reverse": True},
        {"id": 3, "text": "does a thorough job", "trait": "Conscientiousness", "reverse": False},
        {"id": 4, "text": "is depressed, blue", "trait": "Neuroticism", "reverse": False},
        {"id": 5, "text": "is original, comes up with new ideas", "trait": "Openness", "reverse": False},
        {"id": 6, "text": "is reserved", "trait": "Extraversion", "reverse": True},
        {"id": 7, "text": "is helpful and unselfish with others", "trait": "Agreeableness", "reverse": False},
        {"id": 8, "text": "can be somewhat careless", "trait": "Conscientiousness", "reverse": True},
        {"id": 9, "text": "is relaxed, handles stress well", "trait": "Neuroticism", "reverse": True},
        {"id": 10, "text": "is curious about many different things", "trait": "Openness", "reverse": False}
    ]
    
    # Sample answers for testing
    test_answers = {
        "1": 4,  # Extraversion+
        "2": 2,  # Agreeableness- (reverse scored)
        "3": 5,  # Conscientiousness+
        "4": 3,  # Neuroticism+
        "5": 4,  # Openness+
        "6": 2,  # Extraversion- (reverse scored)
        "7": 5,  # Agreeableness+
        "8": "skipped",  # Skipped question
        "9": 4,  # Neuroticism- (reverse scored)
        "10": 5  # Openness+
    }
    
    # Create scorer and test
    scorer = BFIScorer(test_questions)
    
    # Test individual trait scoring
    print("--- Testing individual trait scoring ---")
    for trait in scorer.TRAITS:
        result = scorer.score_trait(trait, test_answers)
        print(f"{trait}: {result['score']} ({result['level']})")
        
    # Test full report generation
    print("\n--- Testing full report generation ---")
    report = scorer.generate_comprehensive_report(test_answers)
    print(f"Completion: {report['summary']['completion_percentage']}%")
    
    # Print a sample interpretation
    if 'openness' in report['interpretations']:
        print("\nSample interpretation:")
        print(f"Openness: {report['interpretations']['openness']['interpretation']}")