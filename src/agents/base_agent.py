import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class Task:
    """
    Represents a task that can be passed between agents.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict) # type: ignore
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1
    requester_id: Optional[str] = None
    status: str = "pending"

@dataclass
class AgentMessage:
    """
    Message format for inter-agent communication
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4))
    sender_id: str = ""
    recepient_id: str = ""
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class BaseAgent(ABC):
    """
    Base Class for all Agents in the research sytem
    """
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.is_active = False
        self.task_queue = asyncio.Queue()
        self.message_handlers = {}
        self.logger = logging.getLogger(f"agent.{self.name}")

        # Agent Capabilities - What this agent can do
        self.capabilities = []

        # Other agents this agent can communicate to
        self.known_agents = {}

    async def start(self):
        """Start the agent and begin processing tasks"""
        self.is_active = True
        self.logger.info(f"Agent {self.name} started")

        # Start the main processing loop
        asyncio.create_task(self._process_tasks())

    async def stop(self):
        """Stop the agent gracefully"""
        self.is_active = False
        self.logger.info(f"Agent {self.name} stopped")

    async def add_task(self, task: Task):
        """Add a task to the agent's queue"""
        await self.task_queue.put(task)
        self.logger.debug(f"Task {task.id} added to the queue.")

    async def send_message(self, recepient_id: str, message_type: str, content: Dict[str, Any]):
        """Send a message to another Agent"""
        message = AgentMessage(
            sender_id=self.agent_id, 
            recepient_id=recepient_id, 
            message_type=message_type, 
            content=content
        )

        # MUST USE MESSAGE BROKER IN REAL TIME
        if recepient_id in self.known_agents:
            await self.known_agents[recepient_id].receive_message(message)
        else:
            self.logger.warning(f"Unknown recepient: {recepient_id}")
    
    async def receive_message(self, message: AgentMessage):
        """Receive and handle a message from another agent"""
        self.logger.debug(f"Received message from {message.sender_id}")

        # Handle based on message type
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")

    def register_message_handler(self, message_type: str, handler):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler

    def add_known_agent(self, agent_id: str, agent_instance):
        """Add another agent to the known agent."""
        self.known_agents[agent_id] = agent_instance

    async def _process_tasks(self):
        """Main task processing loop"""
        while self.is_active:
            try:
                # Get task with timeout to allow for graceful shutdown
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                self.logger.info(f"Processing task {task.id} of type {task.type}")
                task.status = "processing"

                # process the task
                result = await self.process_task(task)

                task.status = "completed"
                self.logger.info(f"Task {task.id} completed successfully.")

                # Notify task completion if needed
                await self._handle_task_completion(task, result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error Processing task: {str(e)}")
                if 'task' in locals():
                    task.status = 'failed'

    async def _handle_task_completion(self, task: Task, result: Any):
        """Handle task completion - notify requester if needed"""
        if task.requester_id and task.requester_id in self.known_agents:
            await self.send_message(
                task.requester_id, 
                "task_result", 
                {
                    "task_id": task.id, 
                    "result": result, 
                    "status": task.status
                }
            )

    @abstractmethod
    async def process_task(self, task: Task) -> Any:
        """Process a specific task - must be implemented by subclass"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass

    def __str__(self):
        return f"Agent({self.name}, id={self.agent_id}, active={self.is_active})"