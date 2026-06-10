"""Tests for LLM Judge Service including retry logic."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from oai_agent_server.services.llm_judge_service import LLMJudgeService


class TestLLMJudgeServiceInitialization:
    """Test LLMJudgeService initialization."""
    
    def test_initialization(self):
        """Service should initialize with default judge model."""
        logger = MagicMock()
        db_logger = MagicMock()
        agent_class = MagicMock()
        
        service = LLMJudgeService(
            agent_class=agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        assert service.judge_agent is None
        assert service.logger == logger
        assert service.db_logger == db_logger
        assert service.agent_class == agent_class
        assert "claude" in service.judge_model_id.lower()
    
    def test_initialization_custom_model(self):
        """Service should accept custom judge model ID."""
        logger = MagicMock()
        db_logger = MagicMock()
        agent_class = MagicMock()
        custom_model = "bedrock/custom-model:1.0"
        
        service = LLMJudgeService(
            agent_class=agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger,
            judge_model_id=custom_model
        )
        
        assert service.judge_model_id == custom_model


class TestLLMJudgeServiceInitializeAgent:
    """Test judge agent initialization."""
    
    @pytest.mark.asyncio
    async def test_initialize_judge_agent_success(self):
        """Judge agent should initialize successfully."""
        logger = MagicMock()
        db_logger = MagicMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        await service._initialize_judge_agent()
        
        assert service.judge_agent is not None
        mock_agent.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_agent_idempotent(self):
        """Initialization should be idempotent."""
        logger = MagicMock()
        db_logger = MagicMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        # Initialize twice
        await service._initialize_judge_agent()
        await service._initialize_judge_agent()
        
        # Agent class should only be called once due to lock
        assert mock_agent_class.call_count == 1
        assert mock_agent.initialize.call_count == 1


class TestLLMJudgeServiceEvaluation:
    """Test evaluation logic."""
    
    @pytest.mark.asyncio
    async def test_format_evaluation_request(self):
        """Evaluation request should be formatted correctly."""
        logger = MagicMock()
        db_logger = MagicMock()
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        result = service._format_evaluation_request("user prompt", "bot response")
        
        assert "user prompt" in result
        assert "bot response" in result
        assert "quality" in result.lower()
    
    def test_parse_judge_response_json_in_markdown(self):
        """Parser should extract JSON from markdown code block."""
        logger = MagicMock()
        db_logger = MagicMock()
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        response = '```json\n{"quality_score": 8, "sentiment_score": 9}\n```'
        result = service._parse_judge_response(response)
        
        assert result["quality_score"] == 8
        assert result["sentiment_score"] == 9
    
    def test_parse_judge_response_bare_json(self):
        """Parser should extract bare JSON."""
        logger = MagicMock()
        db_logger = MagicMock()
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        response = '{"quality_score": 7}'
        result = service._parse_judge_response(response)
        
        assert result["quality_score"] == 7
    
    def test_parse_judge_response_invalid_json(self):
        """Parser should return error dict for invalid JSON."""
        logger = MagicMock()
        db_logger = MagicMock()
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        response = "not json at all"
        result = service._parse_judge_response(response)
        
        assert "error" in result
    
    def test_parse_judge_response_malformed_json(self):
        """Parser should return error dict for malformed JSON."""
        logger = MagicMock()
        db_logger = MagicMock()
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        response = '{"incomplete": '
        result = service._parse_judge_response(response)
        
        assert "error" in result


class TestLLMJudgeServiceRetryLogic:
    """Test retry logic for evaluation logging."""
    
    @pytest.mark.asyncio
    async def test_log_evaluation_success_first_attempt(self):
        """Successful logging should not retry."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock()
        
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        evaluation_results = {"quality_score": 8}
        await service._log_evaluation_with_retry(
            interaction_id="int123",
            agent_name="test_agent",
            session_id="sess123",
            evaluation_results=evaluation_results
        )
        
        # Should succeed on first attempt
        db_logger.log_llm_judge_evaluation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_evaluation_retries_on_fk_error(self):
        """Should retry on foreign key constraint error."""
        logger = MagicMock()
        db_logger = AsyncMock()
        
        # Fail twice, succeed on third
        db_logger.log_llm_judge_evaluation = AsyncMock(
            side_effect=[
                Exception("Foreign key constraint violation"),
                Exception("Foreign key constraint violation"),
                None  # Success
            ]
        )
        
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        evaluation_results = {"quality_score": 8}
        await service._log_evaluation_with_retry(
            interaction_id="int123",
            agent_name="test_agent",
            session_id="sess123",
            evaluation_results=evaluation_results
        )
        
        # Should have retried
        assert db_logger.log_llm_judge_evaluation.call_count == 3
    
    @pytest.mark.asyncio
    async def test_log_evaluation_exponential_backoff(self):
        """Retry should use exponential backoff."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock(
            side_effect=[
                Exception("Foreign key constraint violation"),
                None  # Success on second try
            ]
        )
        
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        with patch("asyncio.sleep") as mock_sleep:
            evaluation_results = {"quality_score": 8}
            await service._log_evaluation_with_retry(
                interaction_id="int123",
                agent_name="test_agent",
                session_id="sess123",
                evaluation_results=evaluation_results
            )
            
            # Should have slept with exponential backoff (0.1 * 2^0 = 0.1)
            mock_sleep.assert_called_once()
            called_delay = mock_sleep.call_args[0][0]
            assert 0.09 < called_delay < 0.11  # Allow small tolerance
    
    @pytest.mark.asyncio
    async def test_log_evaluation_max_retries_exceeded(self):
        """Should raise after max retries."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock(
            side_effect=Exception("Foreign key constraint violation")
        )
        
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        with pytest.raises(Exception, match="Foreign key constraint"):
            evaluation_results = {"quality_score": 8}
            await service._log_evaluation_with_retry(
                interaction_id="int123",
                agent_name="test_agent",
                session_id="sess123",
                evaluation_results=evaluation_results
            )
        
        # Should have tried max_retries (5) times
        assert db_logger.log_llm_judge_evaluation.call_count == 5
    
    @pytest.mark.asyncio
    async def test_log_evaluation_non_fk_error_fails_immediately(self):
        """Non-FK errors should fail immediately without retry."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock(
            side_effect=Exception("Some other error")
        )
        
        service = LLMJudgeService(
            agent_class=MagicMock(),
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        with pytest.raises(Exception, match="Some other error"):
            evaluation_results = {"quality_score": 8}
            await service._log_evaluation_with_retry(
                interaction_id="int123",
                agent_name="test_agent",
                session_id="sess123",
                evaluation_results=evaluation_results
            )
        
        # Should have only tried once
        assert db_logger.log_llm_judge_evaluation.call_count == 1


class TestLLMJudgeServiceJudgeInteraction:
    """Test judge_interaction method."""
    
    @pytest.mark.asyncio
    async def test_judge_interaction_success(self):
        """Successful evaluation should log results."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={"content": [{"text": '{"quality_score": 8}'}]}
        )
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        await service.judge_interaction(
            interaction_id="int123",
            agent_name="test_agent",
            session_id="sess123",
            user_message="Hello",
            agent_response="Hi there",
            user_id="user123"
        )
        
        # Should have logged the evaluation
        db_logger.log_llm_judge_evaluation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_judge_interaction_handles_dict_content(self):
        """Should handle agent response as dict with content list."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "content": [{"text": '{"quality_score": 9}'}]
            }
        )
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        await service.judge_interaction(
            interaction_id="int123",
            agent_name="test_agent",
            session_id="sess123",
            user_message="Hello",
            agent_response="Hi there",
            user_id="user123"
        )
        
        db_logger.log_llm_judge_evaluation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_judge_interaction_logs_errors(self):
        """Should log errors gracefully without raising."""
        logger = MagicMock()
        db_logger = AsyncMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            side_effect=Exception("Judge agent error")
        )
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        # Should not raise
        await service.judge_interaction(
            interaction_id="int123",
            agent_name="test_agent",
            session_id="sess123",
            user_message="Hello",
            agent_response="Hi there",
            user_id="user123"
        )
        
        # Should have logged the error
        assert logger.error.called
    
    @pytest.mark.asyncio
    async def test_judge_interaction_skips_logging_on_parse_error(self):
        """Should skip database logging if response parsing fails."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={"content": [{"text": "not json at all"}]}
        )
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        await service.judge_interaction(
            interaction_id="int123",
            agent_name="test_agent",
            session_id="sess123",
            user_message="Hello",
            agent_response="Hi there",
            user_id="user123"
        )
        
        # Should NOT have logged evaluation due to parse error
        db_logger.log_llm_judge_evaluation.assert_not_called()
        # Should have logged the parse error
        assert logger.error.called


class TestLLMJudgeServiceConcurrency:
    """Test concurrent access to judge service."""
    
    @pytest.mark.asyncio
    async def test_concurrent_judge_initializations(self):
        """Multiple concurrent initialization calls should serialize properly."""
        logger = MagicMock()
        db_logger = AsyncMock()
        
        call_count = 0
        
        async def slow_init():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
        
        mock_agent = AsyncMock()
        mock_agent.initialize = slow_init
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        # Try to initialize concurrently
        await asyncio.gather(
            service._initialize_judge_agent(),
            service._initialize_judge_agent(),
            service._initialize_judge_agent()
        )
        
        # Should only initialize once due to lock
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self):
        """Multiple concurrent evaluations should work."""
        logger = MagicMock()
        db_logger = AsyncMock()
        db_logger.log_llm_judge_evaluation = AsyncMock()
        
        mock_agent = AsyncMock()
        mock_agent.initialize = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={"content": [{"text": '{"quality_score": 8}'}]}
        )
        mock_agent_class = MagicMock(return_value=mock_agent)
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/config",
            logger=logger,
            db_logger=db_logger
        )
        
        # Run concurrent evaluations
        tasks = [
            service.judge_interaction(
                interaction_id=f"int{i}",
                agent_name="test_agent",
                session_id=f"sess{i}",
                user_message="Hello",
                agent_response="Hi there",
                user_id=f"user{i}"
            )
            for i in range(3)
        ]
        
        await asyncio.gather(*tasks)
        
        # All should succeed
        assert db_logger.log_llm_judge_evaluation.call_count == 3
