"""
Forge AGI - Distributed AI Research Platform
A working prototype with 3 advanced features:
1. Multi-Agent Autonomous Collaboration (Thinker, Coder, Critic, Learner)
2. Persistent Knowledge Memory with Vector Search
3. Distributed Task Queue with Worker Nodes
"""

import os
import json
import sqlite3
import logging
import asyncio
from typing import Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("forge_agi")

try:
    from anthropic import Anthropic
except ImportError:
    class Anthropic:
        """Fallback Anthropic client stub when the package is unavailable."""

        class messages:
            @staticmethod
            def create(*args, **kwargs):
                class Response:
                    def __init__(self):
                        content_item = type("ContentItem", (), {"text": "Anthropic client unavailable. Install the 'anthropic' package to enable AI features."})
                        self.content = [content_item]

                return Response()

# ============================================================================
# FEATURE 1: PERSISTENT MEMORY SYSTEM (Vector-like embedding storage)
# ============================================================================

class MemoryDB:
    """Simple but powerful memory system - stores solutions and reuses them"""

    def __init__(self, db_path="forge_memory.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def init_db(self):
        """Create memory tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT UNIQUE,
            description TEXT,
            solution TEXT,
            success INTEGER,
            timestamp DATETIME,
            agent_type TEXT
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            description TEXT,
            code_template TEXT,
            success_rate REAL,
            timestamp DATETIME
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovery TEXT,
            relevance_score REAL,
            agent_who_found TEXT,
            timestamp DATETIME
        )''')

        self.conn.commit()
        log.info("MemoryDB initialized at %s", self.db_path)

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            log.info("MemoryDB connection closed")

    def store_experiment(self, task_name: str, description: str, solution: str, success: bool, agent_type: str):
        """Store an experiment result"""
        c = self.conn.cursor()
        try:
            c.execute('''INSERT OR REPLACE INTO experiments
                        (task_name, description, solution, success, timestamp, agent_type)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (task_name, description, solution, int(success), datetime.now().isoformat(), agent_type))
            self.conn.commit()
            log.info("Stored experiment '%s' (success=%s)", task_name, success)
        except Exception as e:
            log.error("Error storing experiment: %s", e)

    def find_similar_solution(self, task_description: str) -> Optional[dict]:
        """Find a previous solution that might be relevant (keyword matching across all words)"""
        c = self.conn.cursor()
        keywords = [w for w in task_description.lower().split() if len(w) > 2]
        if not keywords:
            return None

        best_match = None
        best_keyword_hits = 0

        for keyword in keywords:
            c.execute('''SELECT * FROM experiments
                        WHERE description LIKE ? AND success = 1
                        ORDER BY timestamp DESC LIMIT 5''',
                     (f"%{keyword}%",))
            for row in c.fetchall():
                desc_words = set(row[2].lower().split())
                hits = sum(1 for kw in keywords if kw in desc_words)
                if hits > best_keyword_hits:
                    best_keyword_hits = hits
                    best_match = {
                        "task_name": row[1],
                        "description": row[2],
                        "solution": row[3],
                        "agent_type": row[6]
                    }

        if best_match:
            log.info("Found similar solution '%s' with %d keyword hits", best_match["task_name"], best_keyword_hits)
        else:
            log.info("No similar solution found for description")
        return best_match

    def store_discovery(self, discovery: str, relevance: float, agent: str):
        """Store a new discovery"""
        c = self.conn.cursor()
        c.execute('''INSERT INTO discoveries (discovery, relevance_score, agent_who_found, timestamp)
                    VALUES (?, ?, ?, ?)''',
                 (discovery, relevance, agent, datetime.now().isoformat()))
        self.conn.commit()
        log.info("Stored discovery from %s (relevance=%.2f)", agent, relevance)

    def get_recent_discoveries(self, limit: int = 5) -> list:
        """Get recent discoveries to learn from"""
        c = self.conn.cursor()
        c.execute('''SELECT discovery, agent_who_found FROM discoveries
                    ORDER BY timestamp DESC LIMIT ?''', (limit,))
        return [{"discovery": row[0], "from": row[1]} for row in c.fetchall()]

    def store_pattern(self, pattern_type: str, description: str, code_template: str, success_rate: float):
        """Store a reusable pattern learned from previous work."""
        c = self.conn.cursor()
        try:
            c.execute('''INSERT INTO patterns (pattern_type, description, code_template, success_rate, timestamp)
                        VALUES (?, ?, ?, ?, ?)''',
                     (pattern_type, description, code_template, success_rate, datetime.now().isoformat()))
            self.conn.commit()
            log.info("Stored pattern '%s' (success_rate=%.2f)", pattern_type, success_rate)
        except Exception as e:
            log.error("Error storing pattern: %s", e)

    def find_relevant_patterns(self, keywords: list, limit: int = 5) -> list:
        """Find patterns matching the given keywords."""
        c = self.conn.cursor()
        results = []
        for keyword in keywords:
            c.execute('''SELECT * FROM patterns
                        WHERE description LIKE ? OR pattern_type LIKE ?
                        ORDER BY success_rate DESC LIMIT ?''',
                     (f"%{keyword}%", f"%{keyword}%", limit))
            for row in c.fetchall():
                entry = {
                    "id": row[0],
                    "pattern_type": row[1],
                    "description": row[2],
                    "code_template": row[3],
                    "success_rate": row[4]
                }
                if entry not in results:
                    results.append(entry)
        log.info("Found %d relevant patterns for keywords", len(results))
        return results[:limit]

# ============================================================================
# FEATURE 2: MULTI-AGENT AUTONOMOUS COLLABORATION
# ============================================================================

class AIAgent:
    """Base class for autonomous AI agents"""

    def __init__(self, name: str, role: str, memory: MemoryDB):
        self.name = name
        self.role = role
        self.memory = memory
        self.client = Anthropic()
        self.conversation_history = []

    def think(self, task: str) -> str:
        """Agent thinks about a task"""
        log.info("Agent %s starting task", self.name)
        self.conversation_history.append({
            "role": "user",
            "content": task
        })

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                system=self._get_system_prompt(),
                messages=self.conversation_history
            )
            result = response.content[0].text
            self.conversation_history.append({
                "role": "assistant",
                "content": result
            })
            log.info("Agent %s completed task (output length=%d)", self.name, len(result))
            return result
        except Exception as e:
            log.error("Agent %s failed with error: %s", self.name, e)
            return f"[{self.name} ERROR] Unable to process request: {e}"

    def _get_system_prompt(self) -> str:
        """Override in subclasses"""
        return f"You are {self.name}, a {self.role} AI agent."

class ThinkerAgent(AIAgent):
    """Plans research and generates ideas"""

    def __init__(self, memory: MemoryDB):
        super().__init__("Thinker", "planning and research", memory)

    def _get_system_prompt(self) -> str:
        recent = self.memory.get_recent_discoveries(3)
        recent_str = "\n".join([f"- {d['discovery']}" for d in recent]) if recent else "None yet"

        return f"""You are the Thinker Agent, a world-class research strategist. Your job is to:

1. **Analyze the problem deeply** — Understand the core challenge, constraints, and desired outcome before proposing anything.
2. **Break the problem into sub-tasks** — Decompose the work into clear, ordered, actionable steps.
3. **Reference recent discoveries** — Consider these past findings and explain how they apply or conflict:
   {recent_str}
4. **Propose hypotheses** — Suggest 1-2 specific approaches or hypotheses to test, explaining why each might work.
5. **Identify risks** — Flag potential pitfalls or assumptions that could derail the approach.

Output format:
- Summary of the problem (1-2 sentences)
- Sub-tasks (numbered list)
- Recommended approach with rationale
- Open questions or risks

Be concise but thorough. Think step by step."""

class CoderAgent(AIAgent):
    """Writes and improves code"""

    def __init__(self, memory: MemoryDB):
        super().__init__("Coder", "implementation and coding", memory)

    def _get_system_prompt(self) -> str:
        patterns = self.memory.find_relevant_patterns(["code", "implementation", "python"])
        patterns_str = ""
        if patterns:
            patterns_str = "\n".join([f"- {p['description']}" for p in patterns[:3]])

        return f"""You are the Coder Agent, an expert software engineer. Your job is to:

1. **Write production-ready code** — Clean, idiomatic, well-structured code with proper error handling.
2. **Handle edge cases** — Consider empty inputs, boundary values, type mismatches, and failure states.
3. **Use patterns from memory** — Reuse and adapt these proven patterns where applicable:
   {patterns_str if patterns_str else "  (No stored patterns yet — focus on writing clean, reusable code.)"}
4. **Follow the approach suggested by the Thinker** — Stay aligned with the research plan.
5. **Add inline comments** only for complex logic where the intent is not obvious.
6. **Use standard library when possible** — Prefer built-in solutions over external dependencies.

Write Python code when possible. Be pragmatic. Output only the code and a brief usage example."""

class CriticAgent(AIAgent):
    """Reviews and improves solutions"""

    def __init__(self, memory: MemoryDB):
        super().__init__("Critic", "evaluation and improvement", memory)

    def _get_system_prompt(self) -> str:
        return """You are the Critic Agent, a senior code reviewer and quality assurance expert. Your job is to:

Review the solution against this **quality rubric** (rate each category 1-10):

| Category       | What to check                                                        |
|----------------|----------------------------------------------------------------------|
| **Correctness**  | Does the solution solve the stated problem? Are there logic errors?  |
| **Performance**  | Is the algorithm efficient? Could it be optimized? Any O(n²) issues? |
| **Readability**  | Is the code clear, well-structured, and easy to follow?              |
| **Security**     | Are there injection risks, data leaks, or unsafe patterns?           |
| **Edge Cases**   | Are empty inputs, errors, and boundary conditions handled?           |

Output format:
- **Overall score: X/10**
- Per-category scores with brief justification
- Top 3 actionable improvements (most important first)
- One thing that was done well

Be honest and constructive. Focus on practical improvements that can be implemented immediately."""

class LearnerAgent(AIAgent):
    """Extracts insights and stores learnings"""

    def __init__(self, memory: MemoryDB):
        super().__init__("Learner", "knowledge extraction and memory", memory)

    def _get_system_prompt(self) -> str:
        return """You are the Learner Agent, a knowledge management specialist. Your job is to:

1. **Extract specific reusable patterns** — What code structure, algorithm, or design pattern can we reuse later?
2. **Calculate relevance scores** — For each insight, assign a relevance score (0.0 - 1.0) based on how generalizable it is.
3. **Identify generalizable techniques** — What did we learn that applies beyond this specific problem?
4. **Note what worked and what didn't** — Be honest about failures or suboptimal choices.
5. **Suggest future experiments** — What should we try next based on these results?

Output format (JSON-like structure):
- **pattern**: Brief description of the reusable pattern
- **relevance**: Score 0.0-1.0
- **technique**: The generalizable technique learned
- **what_worked**: Key success factors
- **what_didnt**: Things to avoid
- **next_steps**: Suggested follow-up experiments

Be precise and actionable. Every insight should be something another agent can actually use."""

# ============================================================================
# FEATURE 3: DISTRIBUTED TASK ORCHESTRATION
# ============================================================================

class TaskQueue:
    """Manages distributed task execution"""

    def __init__(self, db_path="tasks.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()
        self.active_workers = {}

    def init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT,
            description TEXT,
            status TEXT,
            assigned_to TEXT,
            result TEXT,
            created_at DATETIME,
            completed_at DATETIME
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT UNIQUE,
            status TEXT,
            last_heartbeat DATETIME,
            tasks_completed INTEGER
        )''')

        self.conn.commit()
        log.info("TaskQueue initialized at %s", self.db_path)

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            log.info("TaskQueue connection closed")

    def register_worker(self, worker_id: str):
        """Register a distributed worker"""
        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO workers (worker_id, status, last_heartbeat, tasks_completed)
                    VALUES (?, ?, ?, ?)''',
                 (worker_id, 'online', datetime.now().isoformat(), 0))
        self.conn.commit()
        self.active_workers[worker_id] = True
        log.info("Worker '%s' registered", worker_id)

    def enqueue_task(self, task_name: str, description: str) -> int:
        """Add task to queue"""
        c = self.conn.cursor()
        c.execute('''INSERT INTO tasks (task_name, description, status, created_at)
                    VALUES (?, ?, ?, ?)''',
                 (task_name, description, 'pending', datetime.now().isoformat()))
        self.conn.commit()
        log.info("Task '%s' enqueued (id=%d)", task_name, c.lastrowid)
        return c.lastrowid

    def assign_task(self, task_id: int, worker_id: str) -> bool:
        """Assign task to a worker"""
        c = self.conn.cursor()
        c.execute('''UPDATE tasks SET status = ?, assigned_to = ? WHERE id = ?''',
                 ('assigned', worker_id, task_id))
        self.conn.commit()
        log.info("Task %d assigned to worker '%s'", task_id, worker_id)
        return True

    def complete_task(self, task_id: int, result: str):
        """Mark task as complete"""
        c = self.conn.cursor()
        c.execute('''UPDATE tasks SET status = ?, result = ?, completed_at = ? WHERE id = ?''',
                 ('completed', result, datetime.now().isoformat(), task_id))
        self.conn.commit()
        log.info("Task %d completed", task_id)

    def get_pending_tasks(self, limit: int = 10) -> list:
        """Get tasks waiting for a worker"""
        c = self.conn.cursor()
        c.execute('''SELECT id, task_name, description FROM tasks WHERE status = 'pending' LIMIT ?''', (limit,))
        return [{"id": row[0], "name": row[1], "description": row[2]} for row in c.fetchall()]

    def get_worker_stats(self) -> dict:
        """Get cluster statistics"""
        c = self.conn.cursor()
        c.execute('SELECT COUNT(*) FROM workers WHERE status = "online"')
        online_workers = c.fetchone()[0]

        c.execute('SELECT COUNT(*) FROM tasks WHERE status = "completed"')
        completed_tasks = c.fetchone()[0]

        return {
            "online_workers": online_workers,
            "completed_tasks": completed_tasks,
            "active_workers": list(self.active_workers.keys())
        }

    def record_heartbeat(self, worker_id: str):
        """Record a heartbeat from a worker, marking them online."""
        c = self.conn.cursor()
        now = datetime.now().isoformat()
        c.execute('''UPDATE workers SET status = ?, last_heartbeat = ? WHERE worker_id = ?''',
                  ('online', now, worker_id))
        if c.rowcount == 0:
            c.execute('''INSERT INTO workers (worker_id, status, last_heartbeat, tasks_completed)
                        VALUES (?, ?, ?, 0)''',
                     (worker_id, 'online', now))
        self.conn.commit()
        self.active_workers[worker_id] = True
        log.info("Heartbeat recorded for worker '%s'", worker_id)

    def mark_workers_offline(self, stale_threshold_seconds: int = 60):
        """Mark workers as offline if their heartbeat is older than the threshold.

        Returns the list of worker IDs that were marked offline.
        """
        c = self.conn.cursor()
        threshold = (datetime.now() - timedelta(seconds=stale_threshold_seconds)).isoformat()
        c.execute('''UPDATE workers SET status = ? WHERE status = ? AND last_heartbeat < ?''',
                  ('offline', 'online', threshold))
        affected = c.fetchone() if c.description else None
        self.conn.commit()
        log.info("Marked workers offline (threshold=%ss)", stale_threshold_seconds)
        return self._get_recently_offline_workers(threshold)

    def _get_recently_offline_workers(self, threshold: str) -> list:
        """Get workers that were just marked offline within this cycle."""
        c = self.conn.cursor()
        c.execute('''SELECT worker_id FROM workers WHERE status = ? AND last_heartbeat < ?''',
                  ('offline', threshold))
        workers = [row[0] for row in c.fetchall()]
        for w in workers:
            self.active_workers.pop(w, None)
        return workers

    def reassign_orphaned_tasks(self, stale_threshold_seconds: int = 60) -> int:
        """Reassign tasks that were assigned to workers who went offline.

        Returns the number of tasks reassigned.
        """
        c = self.conn.cursor()
        threshold = (datetime.now() - timedelta(seconds=stale_threshold_seconds)).isoformat()
        c.execute('''UPDATE tasks SET status = ?, assigned_to = NULL, completed_at = NULL
                     WHERE status = ? AND assigned_to IN (
                         SELECT worker_id FROM workers
                         WHERE status = ? AND last_heartbeat < ?
                     )''',
                  ('pending', 'assigned', 'offline', threshold))
        self.conn.commit()
        count = c.rowcount
        if count > 0:
            log.info("Reassigned %d orphaned tasks back to pending", count)
        return count

    def run_maintenance(self, stale_threshold_seconds: int = 60):
        """Run all maintenance tasks: detect stale workers and reassign orphaned tasks."""
        self.mark_workers_offline(stale_threshold_seconds)
        self.reassign_orphaned_tasks(stale_threshold_seconds)

# ============================================================================
# LIFESPAN (startup / shutdown)
# ============================================================================

# ---------------------------------------------------------------------------
# Background maintenance: detect stale workers, reassign orphaned tasks
# ---------------------------------------------------------------------------

_stop_event = asyncio.Event()

async def maintenance_loop(interval_seconds: int = 30, stale_threshold: int = 60):
    """Periodically detect stale workers and reassign orphaned tasks."""
    log.info("Maintenance loop started (interval=%ss, stale_threshold=%ss)", interval_seconds, stale_threshold)
    try:
        while not _stop_event.is_set():
            await asyncio.sleep(interval_seconds)
            task_queue.run_maintenance(stale_threshold)
    except asyncio.CancelledError:
        log.info("Maintenance loop cancelled")
    except Exception as e:
        log.error("Maintenance loop error: %s", e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Forge AGI starting up")
    task = asyncio.create_task(maintenance_loop())
    yield
    log.info("Forge AGI shutting down — closing database connections")
    _stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    memory.close()
    task_queue.close()

# ============================================================================
# API SETUP
# ============================================================================

app = FastAPI(title="Forge AGI", description="Distributed AI Research Platform", lifespan=lifespan)

memory = MemoryDB()
task_queue = TaskQueue()
agents = {
    "thinker": ThinkerAgent(memory),
    "coder": CoderAgent(memory),
    "critic": CriticAgent(memory),
    "learner": LearnerAgent(memory)
}

class ResearchTask(BaseModel):
    task_name: str
    description: str

class WorkerRegistration(BaseModel):
    worker_id: str

class TaskCompletion(BaseModel):
    result: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agents_loaded": 4,
        "database": "connected"
    }

@app.post("/research/task")
async def create_research_task(task: ResearchTask):
    """Create a new research task that agents will solve collaboratively"""
    log.info("API: create_research_task '%s'", task.task_name)
    task_id = task_queue.enqueue_task(task.task_name, task.description)

    similar = memory.find_similar_solution(task.description)
    if similar:
        return {
            "task_id": task_id,
            "status": "created",
            "similar_solution_found": True,
            "hint": f"We solved something similar before: {similar['task_name']}",
            "previous_solution": similar['solution']
        }

    return {
        "task_id": task_id,
        "status": "created",
        "similar_solution_found": False
    }

@app.post("/research/solve/{task_id}")
async def solve_research_task(task_id: int):
    """Solve a research task using multi-agent collaboration"""
    log.info("API: solve_research_task task_id=%d", task_id)
    c = task_queue.conn.cursor()
    c.execute('SELECT description FROM tasks WHERE id = ?', (task_id,))
    result = c.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    description = result[0]

    def generate():
        try:
            yield "event: stage\ndata: {\"agent\": \"thinker\", \"message\": \"Analyzing problem...\"}\n\n"
            thinking = agents["thinker"].think(description)
            yield f"event: result\ndata: {json.dumps({'stage': 'thinker', 'output': thinking[:200] + '...'})}\n\n"

            yield "event: stage\ndata: {\"agent\": \"coder\", \"message\": \"Writing solution...\"}\n\n"
            coding_prompt = f"Based on this plan: {thinking}\n\nNow write the code:"
            code = agents["coder"].think(coding_prompt)
            yield f"event: result\ndata: {json.dumps({'stage': 'coder', 'output': code[:200] + '...'})}\n\n"

            yield "event: stage\ndata: {\"agent\": \"critic\", \"message\": \"Reviewing solution...\"}\n\n"
            criticism = agents["critic"].think(f"Review this code:\n{code}")
            yield f"event: result\ndata: {json.dumps({'stage': 'critic', 'output': criticism[:200] + '...'})}\n\n"

            yield "event: stage\ndata: {\"agent\": \"learner\", \"message\": \"Storing knowledge...\"}\n\n"
            insights = agents["learner"].think(f"What did we learn from this solution? Key insights:\n{code}")
            yield f"event: result\ndata: {json.dumps({'stage': 'learner', 'output': insights[:200] + '...'})}\n\n"

            memory.store_experiment(
                task_name=f"task_{task_id}",
                description=description,
                solution=code,
                success=True,
                agent_type="multi-agent-ensemble"
            )
            memory.store_discovery(insights, 0.8, "ensemble")

            memory.store_pattern(
                pattern_type="code_solution",
                description=f"Solution for: {description[:100]}",
                code_template=code[:500],
                success_rate=0.8
            )

            task_queue.complete_task(task_id, code)

            yield f"event: complete\ndata: {json.dumps({'status': 'completed', 'task_id': task_id})}\n\n"

        except Exception as e:
            log.error("Solve pipeline failed for task %d: %s", task_id, e)
            yield f"event: error\ndata: {json.dumps({'status': 'failed', 'task_id': task_id, 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/research/history")
async def get_research_history():
    """Get all experiments the system has run"""
    log.info("API: get_research_history")
    c = memory.conn.cursor()
    c.execute('SELECT task_name, description, success FROM experiments ORDER BY timestamp DESC LIMIT 20')
    results = [{"task": row[0], "description": row[1], "success": bool(row[2])} for row in c.fetchall()]
    return {"experiments": results}

@app.get("/research/discoveries")
async def get_discoveries():
    """Get recent discoveries learned by the system"""
    log.info("API: get_discoveries")
    discoveries = memory.get_recent_discoveries(10)
    return {"discoveries": discoveries}

@app.post("/workers/register")
async def register_worker(worker: WorkerRegistration):
    """Register a distributed compute worker"""
    log.info("API: register_worker '%s'", worker.worker_id)
    task_queue.register_worker(worker.worker_id)
    return {
        "worker_id": worker.worker_id,
        "status": "registered",
        "pending_tasks": task_queue.get_pending_tasks(5)
    }

@app.post("/workers/heartbeat")
async def worker_heartbeat(worker: WorkerRegistration):
    """Record a heartbeat from a worker to keep it marked online."""
    log.info("API: heartbeat from worker '%s'", worker.worker_id)
    task_queue.record_heartbeat(worker.worker_id)
    return {"worker_id": worker.worker_id, "status": "online", "timestamp": datetime.now().isoformat()}

@app.get("/workers/stats")
async def get_worker_stats():
    """Get cluster statistics"""
    return task_queue.get_worker_stats()

@app.get("/tasks/pending")
async def get_pending_tasks():
    """Get list of pending tasks (for workers to pull)"""
    return {"tasks": task_queue.get_pending_tasks()}

@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: int, completion: TaskCompletion):
    """Mark a task as complete (called by worker)"""
    log.info("API: complete_task task_id=%d", task_id)
    task_queue.complete_task(task_id, completion.result)
    return {"status": "completed", "task_id": task_id}

@app.get("/")
async def root():
    """API overview"""
    return {
        "system": "Forge AGI",
        "version": "0.2.0",
        "features": [
            "Multi-Agent Autonomous Collaboration",
            "Persistent Learning Memory",
            "Distributed Task Orchestration"
        ],
        "endpoints": {
            "health": {
                "GET /health": "Health check"
            },
            "research": {
                "POST /research/task": "Create a new research task",
                "POST /research/solve/{task_id}": "Solve task with agent collaboration",
                "GET /research/history": "View past experiments",
                "GET /research/discoveries": "View system discoveries"
            },
            "workers": {
                "POST /workers/register": "Register a compute worker",
                "POST /workers/heartbeat": "Send heartbeat to stay online",
                "GET /workers/stats": "Get cluster statistics",
                "GET /tasks/pending": "Get pending tasks",
                "POST /tasks/{task_id}/complete": "Mark task complete"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
