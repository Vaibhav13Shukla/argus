import logging
from typing import Any, Dict

from dotenv import load_dotenv

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, openrouter

from argus_processor import ArgusProcessor, ObjectAppearedEvent, ObjectDisappearedEvent, ObjectMovedEvent

load_dotenv()
logger = logging.getLogger(__name__)


async def create_agent(**kwargs) -> Agent:
    # Use a fast model for lower latency
    llm = openrouter.LLM(model="meta-llama/llama-4-scout-17b-16e-instruct")

    argus = ArgusProcessor(
        fps=5,
        yolo_model="yolo26n.pt",
        device="cpu",
        conf_threshold=0.35,
        disappear_threshold=5.0,
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="ARGUS", id="argus-agent"),
        instructions="Read @instructions.md",
        processors=[argus],
        llm=llm,
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    agent.events.merge(argus.events)

    # --- TOOLS ---

    @llm.register_function(description="Search ARGUS temporal memory for objects, people, or events. Use when user asks where is, did anyone, what happened, find, locate, or search for something.")
    async def search_memory(query: str) -> Dict[str, Any]:
        common_objects = ["person","cup","bottle","phone","laptop","book","bag","chair","keys","remote","mouse","keyboard","cat","dog","car","backpack","umbrella","handbag","cell phone","tv","clock","bowl","spoon","fork","knife"]
        for obj_type in common_objects:
            if obj_type in query.lower():
                result = argus.memory.find_object(obj_type)
                if "No '" not in result:
                    return {"found": True, "result": result, "query": query}
        return {"found": True, "result": argus.memory.build_context(query), "query": query}

    @llm.register_function(description="Get activity summary including objects seen, movements, and events. Use for what happened, summarize, give update, recap, tell me everything.")
    async def get_activity_summary(minutes: int = 5) -> Dict[str, Any]:
        stats = argus.memory.get_stats()
        context = argus.memory.build_context(f"summarize last {minutes} minutes")
        timeline = argus.memory.get_timeline(minutes)
        return {"stats": stats, "context": context, "recent_events": timeline[-20:]}

    @llm.register_function(description="Get current location and status of all tracked objects. Use for what do you see, what is in the frame, list objects, describe the scene.")
    async def get_object_locations() -> Dict[str, Any]:
        import time as t
        now = t.time()
        objects = []
        for obj in argus.memory.objects.values():
            objects.append({
                "track_id": obj.track_id,
                "class": obj.class_name,
                "zone": obj.last_zone,
                "visible": obj.is_active,
                "last_seen": obj.to_summary(now),
            })
        return {
            "objects": objects,
            "total": len(objects),
            "visible": sum(1 for o in objects if o["visible"]),
        }

    @llm.register_function(description="Get chronological event log showing when objects appeared, disappeared, or moved. Use for show timeline, what events, history.")
    async def get_event_timeline(minutes: int = 10) -> Dict[str, Any]:
        timeline = argus.memory.get_timeline(minutes)
        return {
            "events": timeline,
            "total_events": len(timeline),
            "period_minutes": minutes,
        }

    # --- EVENT HANDLERS ---

    @agent.events.subscribe
    async def on_object_appeared(event: ObjectAppearedEvent):
        logger.info(f"🟢 {event.class_name} (ID:{event.track_id}) appeared at {event.zone}")

    @agent.events.subscribe
    async def on_object_disappeared(event: ObjectDisappearedEvent):
        logger.info(f"🔴 {event.class_name} (ID:{event.track_id}) left (was at {event.last_zone})")

    @agent.events.subscribe
    async def on_object_moved(event: ObjectMovedEvent):
        logger.info(f"🔄 {event.class_name} (ID:{event.track_id}) moved {event.from_zone} -> {event.to_zone}")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.llm.simple_response(
            text="Say exactly: Hi, I'm ARGUS. Show me something."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()