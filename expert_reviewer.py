# expert_reviewer.py
"""
Expert Reviewer module
- Handles human/medical expert review for flagged outputs
- No LLM or ML model calls — purely rule-based and manual workflow
"""

from datetime import datetime

class ExpertReviewer:
    """Handles expert review workflow when AI confidence is low or hallucination detected"""

    def __init__(self):
        # store pending reviews in memory (could be replaced with database later)
        self.pending_reviews = []

    def review(self, question, answer, detection):
        """
        Handle review decision.
        If flagged by the detection system, submit for expert review.

        Args:
            question (str): The medical question.
            answer (str): The answer text.
            detection (dict): The model's detection result containing
                              'is_hallucination' and 'confidence'.
        """
        review_needed = detection.get("is_hallucination") or detection.get("confidence", 1) < 0.7

        if review_needed:
            review_item = {
                "id": len(self.pending_reviews) + 1,
                "question": question,
                "answer": answer,
                "model_confidence": detection.get("confidence"),
                "is_hallucination": detection.get("is_hallucination"),
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            self.pending_reviews.append(review_item)

            return (
                "⚠️  Automatic detection flagged this answer for review.\n"
                f"- Status: Pending human expert evaluation.\n"
                "Please forward this case to a qualified medical reviewer."
            )
        else:
            return (
                "✅ Detection confidence is sufficient. "
                "No human review required."
            )

    def list_pending(self):
        """Return all pending expert reviews"""
        return [r for r in self.pending_reviews if r["status"] == "pending"]

    def finalize_review(self, review_id, expert_decision, reviewer_name="expert"):
        """
        Finalize the review with an expert decision.
        Args:
            review_id (int): ID of the pending review.
            expert_decision (str): 'accurate' or 'hallucination'.
        """
        for r in self.pending_reviews:
            if r["id"] == review_id:
                r["status"] = "reviewed"
                r["expert_decision"] = expert_decision
                r["reviewed_by"] = reviewer_name
                r["reviewed_at"] = datetime.now().isoformat()
                return {
                    "review_id": review_id,
                    "decision": expert_decision,
                    "reviewed_by": reviewer_name,
                    "timestamp": r["reviewed_at"]
                }
        return {"error": f"Review ID {review_id} not found."}
