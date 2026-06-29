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
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
from anthropic import Anthropic

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
        self.conn = sqlite3.connect(self.db_path)
        c = self.conn.cursor()
        
        # Store research ideas and their outcomes
        c.execute('''CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT UNIQUE,
            description TEXT,
            solution TEXT,
            success INTEGER,
            timestamp DATETIME,
            agent_type TEXT
        )''')
        
        # Store learned patterns
        c.execute('''CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            description TEXT,
            code_template TEXT,
            success_rate REAL,
            timestamp DATETIME
        )''')
        
        # Store agent discoveries
        c.execute('''CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovery TEXT,
            relevance_score REAL,
            agent_who_found TEXT,
            timestamp DATETIME
        )''')
        
        self.conn.commit()
    
    def store_experiment(self, task_name: str, description: str, solution: str, success: bool, agent_type: str):
        """Store an experiment result"""
        c = self.conn.cursor()
        try:
            c.execute('''INSERT OR REPLACE INTO experiments 
                        (task_name, description, solution, success, timestamp, agent_type)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (task_name, description, solution, int(success), datetime.now().isoformat(), agent_type))
            self.conn.commit()
        except Exception as e:
            print(f"Error storing experiment: {e}")
    
    def find_similar_solution(self, task_description: str) -> Optional[dict]:
        """Find a previous solution that might be relevant (simple keyword matching)"""
        c = self.conn.cursor()
        # Very basic similarity: look for keywords
        keywords = task_description.lower().split()[:3]
        
        for keyword in keywords:
            c.execute('''SELECT * FROM experiments 
                        WHERE description LIKE ? AND success = 1 
                        ORDER BY timestamp DESC LIMIT 1''',
                     (f"%{keyword}%",))
            result = c.fetchone()
            if result:
                return {
                    "task_name": result[1],
                    "description": result[2],
                    "solution": result[3],
                    "agent_type": result[6]
                }
        return None
    
    def store_discovery(self, discovery: str, relevance: float, agent: str):
        """Store a new discovery"""
        c = self.conn.cursor()
        c.execute('''INSERT INTO discoveries (discovery, relevance_score, agent_who_found, timestamp)
                    VALUES (?, ?, ?, ?)''',
                 (discovery, relevance, agent, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_recent_discoveries(self, limit: int = 5) -> list:
        """Get recent discoveries to learn from"""
        c = self.conn.cursor()
        c.execute('''SELECT discovery, agent_who_found FROM discoveries 
                    ORDER BY timestamp DESC LIMIT ?''', (limit,))
        return [{"discovery": row[0], "from": row[1]} for row in c.fetchall()]

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
        self.conversation_history.append({
            "role": "user",
            "content": task
        })
        
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
        
        return result
    
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
        
        return f"""You are the Thinker Agent. Your job is to:
1. Analyze the problem deeply
2. Consider recent discoveries: {recent_str}
3. Break the problem into sub-tasks
4. Suggest the best approach

Be concise but thorough. Think step by step."""

class CoderAgent(AIAgent):
    """Writes and improves code"""
    
    def __init__(self, memory: MemoryDB):
        super().__init__("Coder", "implementation and coding", memory)
    
    def _get_system_prompt(self) -> str:
        return """You are the Coder Agent. Your job is to:
1. Write clean, working code
2. Follow the approach suggested by the Thinker
3. Reuse patterns from similar past solutions
4. Test your code logic

Write Python code when possible. Be pragmatic."""

class CriticAgent(AIAgent):
    """Reviews and improves solutions"""
    
    def __init__(self, memory: MemoryDB):
        super().__init__("Critic", "evaluation and improvement", memory)
    
    def _get_system_prompt(self) -> str:
        return """You are the Critic Agent. Your job is to:
1. Review the solution critically
2. Point out potential issues
3. Suggest improvements
4. Rate the solution quality (1-10)

Be honest and constructive. Focus on practical improvements."""

class LearnerAgent(AIAgent):
    """Extracts insights and stores learnings"""
    
    def __init__(self, memory: MemoryDB):
        super().__init__("Learner", "knowledge extraction and memory", memory)
    
    def _get_system_prompt(self) -> str:
        return """You are the Learner Agent. Your job is to:
1. Extract key insights from the completed work
2. Identify reusable patterns
3. Find generalizable solutions
4. Note what worked and what didn't

Format insights as clear bullet points."""

# ============================================================================
# FEATURE 3: DISTRIBUTED TASK ORCHESTRATION
# ============================================================================

class TaskQueue:
    """Manages distributed task execution"""
    
    def __init__(self, db_path="tasks.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()
        self.active_workers = {}  # In-memory tracking for demo
    
    def init_db(self):
        self.conn = sqlite3.connect(self.db_path)
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
    
    def register_worker(self, worker_id: str):
        """Register a distributed worker"""
        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO workers (worker_id, status, last_heartbeat, tasks_completed)
                    VALUES (?, ?, ?, ?)''',
                 (worker_id, 'online', datetime.now().isoformat(), 0))
        self.conn.commit()
        self.active_workers[worker_id] = True
    
    def enqueue_task(self, task_name: str, description: str) -> int:
        """Add task to queue"""
        c = self.conn.cursor()
        c.execute('''INSERT INTO tasks (task_name, description, status, created_at)
                    VALUES (?, ?, ?, ?)''',
                 (task_name, description, 'pending', datetime.now().isoformat()))
        self.conn.commit()
        return c.lastrowid
    
    def assign_task(self, task_id: int, worker_id: str) -> bool:
        """Assign task to a worker"""
        c = self.conn.cursor()
        c.execute('''UPDATE tasks SET status = ?, assigned_to = ? WHERE id = ?''',
                 ('assigned', worker_id, task_id))
        self.conn.commit()
        return True
    
    def complete_task(self, task_id: int, result: str):
        """Mark task as complete"""
        c = self.conn.cursor()
        c.execute('''UPDATE tasks SET status = ?, result = ?, completed_at = ? WHERE id = ?''',
                 ('completed', result, datetime.now().isoformat(), task_id))
        self.conn.commit()
    
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

# ============================================================================
# API SETUP
# ============================================================================

app = FastAPI(title="Forge AGI", description="Distributed AI Research Platform")

# Initialize systems
memory = MemoryDB()
task_queue = TaskQueue()
agents = {
    "thinker": ThinkerAgent(memory),
    "coder": CoderAgent(memory),
    "critic": CriticAgent(memory),
    "learner": LearnerAgent(memory)
}

# Request models
class ResearchTask(BaseModel):
    task_name: str
    description: str

class WorkerRegistration(BaseModel):
    worker_id: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/research/task")
async def create_research_task(task: ResearchTask):
    """
    Create a new research task that agents will solve collaboratively
    """
    task_id = task_queue.enqueue_task(task.task_name, task.description)
    
    # Check if we've solved something similar before
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
    """
    Solve a research task using multi-agent collaboration
    Agents work together: Thinker → Coder → Critic → Learner
    """
    c = task_queue.conn.cursor()
    c.execute('SELECT description FROM tasks WHERE id = ?', (task_id,))
    result = c.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    description = result[0]
    
    # Multi-agent collaboration pipeline
    def generate():
        # Stage 1: Thinker - Plan the approach
        yield "data: [THINKER] Analyzing problem...\n\n"
        thinking = agents["thinker"].think(description)
        yield f"data: {json.dumps({'stage': 'thinker', 'output': thinking[:200] + '...'})}\n\n"
        
        # Stage 2: Coder - Implement solution
        yield "data: [CODER] Writing solution...\n\n"
        coding_prompt = f"Based on this plan: {thinking}\n\nNow write the code:"
        code = agents["coder"].think(coding_prompt)
        yield f"data: {json.dumps({'stage': 'coder', 'output': code[:200] + '...'})}\n\n"
        
        # Stage 3: Critic - Review and improve
        yield "data: [CRITIC] Reviewing solution...\n\n"
        criticism = agents["critic"].think(f"Review this code:\n{code}")
        yield f"data: {json.dumps({'stage': 'critic', 'output': criticism[:200] + '...'})}\n\n"
        
        # Stage 4: Learner - Extract insights
        yield "data: [LEARNER] Storing knowledge...\n\n"
        insights = agents["learner"].think(f"What did we learn from this solution? Key insights:\n{code}")
        yield f"data: {json.dumps({'stage': 'learner', 'output': insights[:200] + '...'})}\n\n"
        
        # Store results
        memory.store_experiment(
            task_name=f"task_{task_id}",
            description=description,
            solution=code,
            success=True,
            agent_type="multi-agent-ensemble"
        )
        memory.store_discovery(insights, 0.8, "ensemble")
        
        task_queue.complete_task(task_id, code)
        
        yield f"data: {json.dumps({'status': 'completed', 'task_id': task_id})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/research/history")
async def get_research_history():
    """Get all experiments the system has run"""
    c = memory.conn.cursor()
    c.execute('SELECT task_name, description, success FROM experiments ORDER BY timestamp DESC LIMIT 20')
    results = [{"task": row[0], "description": row[1], "success": bool(row[2])} for row in c.fetchall()]
    return {"experiments": results}

@app.get("/research/discoveries")
async def get_discoveries():
    """Get recent discoveries learned by the system"""
    discoveries = memory.get_recent_discoveries(10)
    return {"discoveries": discoveries}

@app.post("/workers/register")
async def register_worker(worker: WorkerRegistration):
    """Register a distributed compute worker"""
    task_queue.register_worker(worker.worker_id)
    return {
        "worker_id": worker.worker_id,
        "status": "registered",
        "pending_tasks": task_queue.get_pending_tasks(5)
    }

@app.get("/workers/stats")
async def get_worker_stats():
    """Get cluster statistics"""
    return task_queue.get_worker_stats()

@app.get("/tasks/pending")
async def get_pending_tasks():
    """Get list of pending tasks (for workers to pull)"""
    return {"tasks": task_queue.get_pending_tasks()}

@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: int, result: str):
    """Mark a task as complete (called by worker)"""
    task_queue.complete_task(task_id, result)
    return {"status": "completed", "task_id": task_id}

@app.get("/")
async def root():
    """API overview"""
    return {
        "system": "Forge AGI",
        "version": "0.1.0",
        "features": [
            "Multi-Agent Autonomous Collaboration",
            "Persistent Learning Memory",
            "Distributed Task Orchestration"
        ],
        "endpoints": {
            "research": {
                "POST /research/task": "Create a new research task",
                "POST /research/solve/{task_id}": "Solve task with agent collaboration",
                "GET /research/history": "View past experiments",
                "GET /research/discoveries": "View system discoveries"
            },
            "workers": {
                "POST /workers/register": "Register a compute worker",
                "GET /workers/stats": "Get cluster statistics",
                "GET /tasks/pending": "Get pending tasks",
                "POST /tasks/{task_id}/complete": "Mark task complete"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
