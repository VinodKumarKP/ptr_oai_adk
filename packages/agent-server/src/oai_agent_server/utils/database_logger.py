"""
Database logger for tracking agent chat interactions.
Supports PostgreSQL with graceful fallback if database is unavailable.
Includes detailed stream activity logging.
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import asyncpg
    from asyncpg.pool import Pool
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    Pool = None


class DatabaseLogger:
    """
    Asynchronous database logger for agent interactions.
    Gracefully handles database unavailability.
    """

    def __init__(self, logger=None):
        self.pool: Optional[Pool] = None
        self.is_active = False
        self.logger = logger
        self._connection_string = None
        self._db_logging_enabled = False

    async def initialize(self):
        """Initialize database connection pool"""
        if not ASYNCPG_AVAILABLE:
            if self.logger:
                self.logger.warning("asyncpg not installed. Database logging disabled.")
            return

        # Get database connection details from environment
        db_host = os.environ.get('LOGGING_DB_HOST', 'localhost')
        db_port = os.environ.get('LOGGING_DB_PORT', '5432')
        db_name = os.environ.get('LOGGING_DB_NAME', 'agent_logs')
        db_user = os.environ.get('LOGGING_DB_USER', 'postgres')
        db_password = os.environ.get('LOGGING_DB_PASSWORD', 'postgres')
        self._db_logging_enabled = os.environ.get('DB_LOGGING_ENABLED', 'false').lower() == 'true'

        # Connection pool settings (configurable via environment)
        pool_min_size = int(os.environ.get('DB_POOL_MIN_SIZE', '2'))
        pool_max_size = int(os.environ.get('DB_POOL_MAX_SIZE', '4'))
        pool_timeout = int(os.environ.get('DB_POOL_TIMEOUT', '120'))

        if not self._db_logging_enabled:
            if self.logger:
                self.logger.info("Database logging is disabled (DB_LOGGING_ENABLED=false)")
            return

        self._connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=pool_min_size,
                max_size=pool_max_size,
                command_timeout=pool_timeout
            )

            # Create tables if they don't exist
            await self._create_table()
            await self._create_activity_log_table()

            self.is_active = True
            if self.logger:
                self.logger.info(f"Database logging initialized successfully: {db_host}:{db_port}/{db_name}")

        except Exception as e:
            self.is_active = False
            if self.logger:
                self.logger.warning(f"Failed to initialize database logging: {e}. Continuing without DB logging.")

    async def _create_table(self):
        """Create the chat_logs table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            agent_name VARCHAR(255),
            session_id VARCHAR(255),
            user_id VARCHAR(255),
            endpoint VARCHAR(50),
            input_message JSONB,
            output_response JSONB,
            request_headers JSONB,
            model_info JSONB,
            token_usage JSONB,
            total_tokens INT,
            response_time_ms FLOAT,
            status VARCHAR(50) DEFAULT 'success',
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON chat_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_chat_logs_user_id ON chat_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_chat_logs_agent_name ON chat_logs(agent_name);
        CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp ON chat_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_chat_logs_endpoint ON chat_logs(endpoint);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_query)
            try:
                # Ensure column exists for existing tables
                await conn.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS total_tokens INT")
            except Exception:
                pass

    async def _create_activity_log_table(self):
        """Create the agent_activity_log table for streaming content"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS agent_activity_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            agent_name VARCHAR(255),
            session_id VARCHAR(255),
            user_id VARCHAR(255),
            endpoint VARCHAR(50),
            chunk_sequence INT,
            chunk_content JSONB,
            chunk_text TEXT,
            serialization_warning TEXT,
            request_headers JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for efficient querying
        CREATE INDEX IF NOT EXISTS idx_activity_log_session_id ON agent_activity_log(session_id);
        CREATE INDEX IF NOT EXISTS idx_activity_log_agent_name ON agent_activity_log(agent_name);
        CREATE INDEX IF NOT EXISTS idx_activity_log_timestamp ON agent_activity_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_activity_log_user_id ON agent_activity_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_activity_log_session_sequence ON agent_activity_log(session_id, chunk_sequence);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_query)

    async def log_interaction(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        endpoint: str,
        input_message: Any,
        output_response: Any,
        request_headers: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        token_usage: Optional[Dict] = None,
        response_time_ms: Optional[float] = None,
        status: str = 'success',
        error_message: Optional[str] = None
    ):
        """
        Log a chat interaction to the database.
        Only logs if database is both active AND logging is enabled.
        Silently skips if conditions are not met.
        """
        if not self._db_logging_enabled or not self.is_active or not self.pool:
            return

        try:
            # Serialize complex objects to JSON-compatible format
            input_json = self._serialize_for_json(input_message)
            output_json = self._serialize_for_json(output_response)
            headers_json = self._serialize_for_json(request_headers) if request_headers else None
            model_json = self._serialize_for_json(model_info) if model_info else None
            usage_json = self._serialize_for_json(token_usage) if token_usage else None

            # Extract total_tokens handling different casing conventions
            total_tokens = None
            if token_usage and isinstance(token_usage, dict):
                # Check for snake_case first, then camelCase
                total_tokens = token_usage.get('total_tokens')
                if total_tokens is None:
                    total_tokens = token_usage.get('totalTokens')

            insert_query = """
            INSERT INTO chat_logs (
                timestamp, agent_name, session_id, user_id, endpoint,
                input_message, output_response, request_headers, model_info,
                token_usage, total_tokens, response_time_ms, status, error_message
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """

            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    datetime.utcnow(),
                    agent_name,
                    session_id,
                    user_id,
                    endpoint,
                    json.dumps(input_json) if input_json else None,
                    json.dumps(output_json) if output_json else None,
                    json.dumps(headers_json) if headers_json else None,
                    json.dumps(model_json) if model_json else None,
                    json.dumps(usage_json) if usage_json else None,
                    total_tokens,
                    response_time_ms,
                    status,
                    error_message
                )

            if self.logger:
                self.logger.debug(f"Logged interaction: {endpoint} - {session_id}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log interaction to database: {e}")

    async def log_stream_chunk(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        endpoint: str,
        chunk_sequence: int,
        chunk_content: Any,
        chunk_text: Optional[str] = None,
        serialization_warning: Optional[str] = None,
        request_headers: Optional[Dict] = None
    ):
        """
        Log individual streaming chunks to agent_activity_log table.
        Captures every piece of content streamed in /chat/stream endpoint.
        """
        if not self._db_logging_enabled or not self.is_active or not self.pool:
            return

        try:
            # Serialize chunk content
            content_json = self._serialize_for_json(chunk_content)
            headers_json = self._serialize_for_json(request_headers) if request_headers else None

            insert_query = """
            INSERT INTO agent_activity_log (
                timestamp, agent_name, session_id, user_id, endpoint,
                chunk_sequence, chunk_content, chunk_text, serialization_warning,
                request_headers
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """

            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    datetime.utcnow(),
                    agent_name,
                    session_id,
                    user_id,
                    endpoint,
                    chunk_sequence,
                    json.dumps(content_json) if content_json else None,
                    chunk_text,
                    serialization_warning,
                    json.dumps(headers_json) if headers_json else None
                )

            if self.logger:
                self.logger.debug(f"Logged stream chunk {chunk_sequence} for session {session_id}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log stream chunk to database: {e}")

    async def log_stream_chunks_batch(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        endpoint: str,
        chunks: List[Dict[str, Any]],
        request_headers: Optional[Dict] = None
    ):
        """
        Batch insert multiple streaming chunks to agent_activity_log table.
        More efficient than inserting one chunk at a time.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            user_id: User identifier
            endpoint: API endpoint (e.g., "/chat/stream")
            chunks: List of chunk dictionaries with keys:
                - chunk_sequence: int
                - chunk_content: Any
                - chunk_text: Optional[str]
                - serialization_warning: Optional[str]
            request_headers: HTTP request headers
        """
        if not self._db_logging_enabled or not self.is_active or not self.pool or not chunks:
            return

        try:
            # Serialize request headers once
            headers_json = self._serialize_for_json(request_headers) if request_headers else None
            headers_json_str = json.dumps(headers_json) if headers_json else None

            # Prepare batch insert data
            timestamp = datetime.utcnow()
            insert_query = """
            INSERT INTO agent_activity_log (
                timestamp, agent_name, session_id, user_id, endpoint,
                chunk_sequence, chunk_content, chunk_text, serialization_warning,
                request_headers
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """

            async with self.pool.acquire() as conn:
                # Use executemany for batch insert
                await conn.executemany(
                    insert_query,
                    [
                        (
                            chunk.get('timestamp', timestamp),
                            agent_name,
                            session_id,
                            user_id,
                            endpoint,
                            chunk['chunk_sequence'],
                            json.dumps(self._serialize_for_json(chunk['chunk_content'])) if chunk.get('chunk_content') else None,
                            chunk.get('chunk_text'),
                            chunk.get('serialization_warning'),
                            headers_json_str
                        )
                        for chunk in chunks
                    ]
                )

            if self.logger:
                self.logger.info(f"Batch logged {len(chunks)} stream chunks for session {session_id}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to batch log stream chunks to database: {e}")

    def _serialize_for_json(self, obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
        """Convert objects to JSON-serializable format with recursion depth limit"""
        if depth > max_depth:
            return str(obj)

        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item, depth + 1, max_depth) for item in obj]

        if isinstance(obj, dict):
            return {str(key): self._serialize_for_json(value, depth + 1, max_depth) for key, value in obj.items()}

        if hasattr(obj, '__dict__'):
            return {str(key): self._serialize_for_json(value, depth + 1, max_depth) for key, value in obj.__dict__.items()}

        if hasattr(obj, 'dict') and callable(obj.dict):
            try:
                return obj.dict()
            except Exception:
                pass

        if hasattr(obj, 'model_dump') and callable(obj.model_dump):
            try:
                return obj.model_dump()
            except Exception:
                pass

        return str(obj)

    async def get_logs(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs from the database with optional filters.
        Returns empty list if database is not active or logging is disabled.
        """
        if not self._db_logging_enabled or not self.is_active or not self.pool:
            return []

        try:
            # Build dynamic query based on filters
            conditions = []
            params = []
            param_counter = 1

            if agent_name:
                conditions.append(f"agent_name = ${param_counter}")
                params.append(agent_name)
                param_counter += 1

            if session_id:
                conditions.append(f"session_id = ${param_counter}")
                params.append(session_id)
                param_counter += 1

            if user_id:
                conditions.append(f"user_id = ${param_counter}")
                params.append(user_id)
                param_counter += 1

            if endpoint:
                conditions.append(f"endpoint = ${param_counter}")
                params.append(endpoint)
                param_counter += 1

            if start_date:
                conditions.append(f"timestamp >= ${param_counter}")
                params.append(start_date)
                param_counter += 1

            if end_date:
                conditions.append(f"timestamp <= ${param_counter}")
                params.append(end_date)
                param_counter += 1

            if status:
                conditions.append(f"status = ${param_counter}")
                params.append(status)
                param_counter += 1

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
            SELECT 
                id, timestamp, agent_name, session_id, user_id, endpoint,
                input_message, output_response, request_headers, model_info,
                token_usage, total_tokens, response_time_ms, status, error_message, created_at
            FROM chat_logs
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_counter} OFFSET ${param_counter + 1}
            """

            params.extend([limit, offset])

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            # Convert rows to dictionaries
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'].isoformat(),
                    'agent_name': row['agent_name'],
                    'session_id': row['session_id'],
                    'user_id': row['user_id'],
                    'endpoint': row['endpoint'],
                    'input_message': json.loads(row['input_message']) if row['input_message'] else None,
                    'output_response': json.loads(row['output_response']) if row['output_response'] else None,
                    'request_headers': json.loads(row['request_headers']) if row['request_headers'] else None,
                    'model_info': json.loads(row['model_info']) if row['model_info'] else None,
                    'token_usage': json.loads(row['token_usage']) if row['token_usage'] else None,
                    'total_tokens': row.get('total_tokens'),
                    'response_time_ms': row['response_time_ms'],
                    'status': row['status'],
                    'error_message': row['error_message'],
                    'created_at': row['created_at'].isoformat()
                })

            return results

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve logs from database: {e}")
            return []

    async def get_activity_logs(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve streaming activity logs from agent_activity_log table.
        """
        if not self._db_logging_enabled or not self.is_active or not self.pool:
            return []

        try:
            # Build dynamic query based on filters
            conditions = []
            params = []
            param_counter = 1

            if agent_name:
                conditions.append(f"agent_name = ${param_counter}")
                params.append(agent_name)
                param_counter += 1

            if session_id:
                conditions.append(f"session_id = ${param_counter}")
                params.append(session_id)
                param_counter += 1

            if user_id:
                conditions.append(f"user_id = ${param_counter}")
                params.append(user_id)
                param_counter += 1

            if start_date:
                conditions.append(f"timestamp >= ${param_counter}")
                params.append(start_date)
                param_counter += 1

            if end_date:
                conditions.append(f"timestamp <= ${param_counter}")
                params.append(end_date)
                param_counter += 1

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
            SELECT 
                id, timestamp, agent_name, session_id, user_id, endpoint,
                chunk_sequence, chunk_content, chunk_text, serialization_warning,
                request_headers, created_at
            FROM agent_activity_log
            WHERE {where_clause}
            ORDER BY timestamp DESC, chunk_sequence ASC
            LIMIT ${param_counter} OFFSET ${param_counter + 1}
            """

            params.extend([limit, offset])

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            # Convert rows to dictionaries
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'].isoformat(),
                    'agent_name': row['agent_name'],
                    'session_id': row['session_id'],
                    'user_id': row['user_id'],
                    'endpoint': row['endpoint'],
                    'chunk_sequence': row['chunk_sequence'],
                    'chunk_content': json.loads(row['chunk_content']) if row['chunk_content'] else None,
                    'chunk_text': row['chunk_text'],
                    'serialization_warning': row['serialization_warning'],
                    'request_headers': json.loads(row['request_headers']) if row['request_headers'] else None,
                    'created_at': row['created_at'].isoformat()
                })

            return results

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve activity logs from database: {e}")
            return []

    async def get_stats(self, agent_name: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about logged interactions"""
        if not self._db_logging_enabled or not self.is_active or not self.pool:
            return {}

        try:
            conditions = []
            params = []
            param_counter = 1

            if agent_name:
                conditions.append(f"agent_name = ${param_counter}")
                params.append(agent_name)
                param_counter += 1

            if user_id:
                conditions.append(f"user_id = ${param_counter}")
                params.append(user_id)
                param_counter += 1

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            query = f"""
            SELECT 
                COUNT(*) as total_interactions,
                COUNT(DISTINCT session_id) as unique_sessions,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(response_time_ms) as avg_response_time,
                MAX(response_time_ms) as max_response_time,
                MIN(response_time_ms) as min_response_time,
                SUM(total_tokens) as total_tokens_sum,
                COUNT(*) FILTER (WHERE status = 'success') as successful_interactions,
                COUNT(*) FILTER (WHERE status != 'success') as failed_interactions
            FROM chat_logs
            {where_clause}
            """

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)

            return {
                'total_interactions': row['total_interactions'],
                'unique_sessions': row['unique_sessions'],
                'unique_users': row['unique_users'],
                'avg_response_time_ms': float(row['avg_response_time']) if row['avg_response_time'] else None,
                'max_response_time_ms': float(row['max_response_time']) if row['max_response_time'] else None,
                'min_response_time_ms': float(row['min_response_time']) if row['min_response_time'] else None,
                'total_tokens_sum': int(row['total_tokens_sum']) if row['total_tokens_sum'] else 0,
                'successful_interactions': row['successful_interactions'],
                'failed_interactions': row['failed_interactions']
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get stats from database: {e}")
            return {}

    async def get_user_stats(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get statistics grouped by user_id"""
        if not self._db_logging_enabled or not self.is_active or not self.pool:
            return []

        try:
            where_clause = "WHERE agent_name = $1" if agent_name else ""
            params = [agent_name] if agent_name else []

            query = f"""
            SELECT 
                user_id,
                COUNT(*) as total_interactions,
                COUNT(DISTINCT session_id) as unique_sessions,
                AVG(response_time_ms) as avg_response_time,
                MAX(response_time_ms) as max_response_time,
                MIN(response_time_ms) as min_response_time,
                SUM(total_tokens) as total_tokens_sum,
                COUNT(*) FILTER (WHERE status = 'success') as successful_interactions,
                COUNT(*) FILTER (WHERE status != 'success') as failed_interactions
            FROM chat_logs
            {where_clause}
            GROUP BY user_id
            ORDER BY total_interactions DESC
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                results.append({
                    'user_id': row['user_id'],
                    'total_interactions': row['total_interactions'],
                    'unique_sessions': row['unique_sessions'],
                    'avg_response_time_ms': float(row['avg_response_time']) if row['avg_response_time'] else None,
                    'max_response_time_ms': float(row['max_response_time']) if row['max_response_time'] else None,
                    'min_response_time_ms': float(row['min_response_time']) if row['min_response_time'] else None,
                    'total_tokens_sum': int(row['total_tokens_sum']) if row['total_tokens_sum'] else 0,
                    'successful_interactions': row['successful_interactions'],
                    'failed_interactions': row['failed_interactions']
                })

            return results

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get user stats from database: {e}")
            return []

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.is_active = False
            if self.logger:
                self.logger.info("Database connection pool closed")
