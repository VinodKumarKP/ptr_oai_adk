import asyncio
import json
import re

EVALUATION_PROMPT_TEMPLATE = """
You are an expert AI quality evaluator. Analyze the following conversation between a user and an AI agent, then provide a comprehensive quality assessment.

**USER PROMPT:**
{user_prompt}

**AGENT RESPONSE:**
{bot_response}

Respond with ONLY a valid JSON object (no markdown, no explanations) containing these metrics:
  "sentiment_score": <0-10 scale: 0=very negative, 5=neutral, 10=very positive>,
  "toxicity_score": <0-10 scale: 0=not toxic, 10=highly toxic>,
  "satisfaction_score": <0-10 scale: user satisfaction with response>,
  "text_analysis_explanation": "<brief explanation of sentiment, toxicity, and satisfaction scores>",
  
  "bleu_score": <0-1 scale: n-gram overlap quality>,
  "rouge_score": <0-1 scale: recall-oriented quality>,
  "perplexity_score": <low numbers like 1-3 for good quality, higher for uncertain/poor quality>,
  "llm_evaluation_explanation": "<explain BLEU, ROUGE, and perplexity assessment>",
  
  "quality_score": <0-10 scale: overall response quality>,
  "accuracy_score": <0-10 scale: factual correctness>,
  "precision_score": <0-10 scale: focused without irrelevant info>,
  "recall_score": <0-10 scale: completeness addressing all aspects>,
  "ml_performance_explanation": "<explain quality, accuracy, precision, and recall scores>",
  
  "hallucination_detected": <boolean: contains unsupported claims>,
  "hallucination_score": <0-10 scale: severity if detected, 0 if none>,
  "context_relevance": <0-10 scale: how relevant context is>,
  "context_adherence": <0-10 scale: stays grounded in context>,
  "context_precision": <0-10 scale: quality of context retrieval>,
  "context_recall": <0-10 scale: completeness of context>,
  "context_monitoring_explanation": "<explain hallucination detection and context metrics>",
  
  "clarity_score": <0-10 scale: clear and understandable>,
  "coherence_score": <0-10 scale: logical flow>,
  "relevance_score": <0-10 scale: addresses the query>,
  "completeness_score": <0-10 scale: all aspects covered>,
  "readability_score": <0-10 scale: ease of reading>,
  "formality_score": <0-10 scale: professional tone>,
  "empathy_score": <0-10 scale: emotional awareness>,
  "language_style_explanation": "<explain clarity, coherence, and language style scores>",
  
  "detected_intent": "<brief description of user intent>",
  "intent_confidence": <0-10 scale: confidence in intent classification>,
  "complexity_level": "<'simple', 'moderate', or 'complex'>",
  
  "has_code_example": <boolean>,
  "has_structured_format": <boolean>,
  "is_low_quality": <boolean>,
  
  "overall_assessment": "<comprehensive summary of the evaluation across all dimensions>"
"""


class LLMJudgeService:
    def __init__(self, agent_class, config_root, logger, db_logger,
                 judge_model_id="bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        self.agent_class = agent_class
        self.config_root = config_root
        self.logger = logger
        self.db_logger = db_logger
        self.judge_model_id = judge_model_id
        self.judge_agent = None
        self._judge_lock = asyncio.Lock()

    async def _initialize_judge_agent(self):
        async with self._judge_lock:
            if self.judge_agent is None:
                self.logger.info("Initializing LLM Judge Agent...")
                judge_system_prompt = "You are an expert AI quality evaluator. Respond ONLY with the valid JSON object requested."
                judge_config = {
                    'model': {'model_id': self.judge_model_id},
                    'agent_list': [{'judge_agent': {'system_prompt': judge_system_prompt}}]
                }
                self.judge_agent = self.agent_class(
                    agent_name="LLMJudgeAgent",
                    agent_config=judge_config
                )
                await self.judge_agent.initialize()
                self.logger.info("LLM Judge Agent initialized.")

    async def judge_interaction(self, interaction_id, user_message, agent_response, session_id, user_id):
        if self.judge_agent is None:
            await self._initialize_judge_agent()

        self.logger.info(f"LLM Judge: Evaluating interaction {interaction_id}")
        evaluation_request = self._format_evaluation_request(user_message, agent_response)

        try:
            judge_response_raw = await self.judge_agent.ainvoke(evaluation_request)
            if isinstance(judge_response_raw, dict) and 'content' in judge_response_raw:
                judge_response_text = judge_response_raw['content'][0]['text'] if isinstance(
                    judge_response_raw['content'], list) else judge_response_raw['content']['text']
            else:
                judge_response_text = str(judge_response_raw)

            evaluation_results = self._parse_judge_response(judge_response_text)

            if "error" not in evaluation_results:
                max_retries = 3
                retry_delay = 0.5  # seconds
                for attempt in range(max_retries):
                    try:
                        await self.db_logger.log_llm_judge_evaluation(interaction_id, evaluation_results)
                        self.logger.info(f"LLM Judge: Evaluation for {interaction_id} logged successfully.")
                        break  # Success
                    except Exception as db_exc:
                        # Crude check for foreign key errors
                        if "foreign key" in str(db_exc).lower() and attempt < max_retries - 1:
                            self.logger.warning(
                                f"LLM Judge: Retrying log for {interaction_id} due to FK error (attempt {attempt + 1})")
                            await asyncio.sleep(retry_delay)
                        else:
                            raise  # Re-raise the final exception
            else:
                self.logger.error(f"LLM Judge: Skipping database logging for {interaction_id} due to parsing error.")

        except Exception as e:
            self.logger.error(f"LLM Judge: Error during evaluation for {interaction_id}: {e}", exc_info=True)

    def _format_evaluation_request(self, user_prompt, bot_response):
        return EVALUATION_PROMPT_TEMPLATE.format(user_prompt=user_prompt, bot_response=bot_response)

    def _parse_judge_response(self, response_text):
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            else:
                self.logger.error(f"Failed to find JSON in judge response. Response: {response_text}")
                return {"error": "No valid JSON object found in the response."}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from judge response: {e}\nResponse: {response_text}")
            return {"error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while parsing judge response: {e}\nResponse: {response_text}")
            return {"error": f"An unexpected error occurred: {str(e)}"}
