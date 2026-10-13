import asyncio
import contextlib
from collections.abc import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import InvalidStatus, WebSocketException

from config import Settings, get_settings
from gnani.prisma import PrismaConfig, PrismaStream, load_bias_words

app = FastAPI(title="Dhwani Desk", version="0.1.0")


def create_stt_stream(settings: Settings) -> PrismaStream:
    return PrismaStream(
        PrismaConfig(
            api_key=settings.gnani_api_key.get_secret_value(),
            url=settings.gnani_stt_url,
            lang_code=settings.gnani_stt_lang,
            sample_rate=settings.gnani_stt_sample_rate,
            output_format=settings.gnani_stt_format,
            bias_words=load_bias_words(settings.gnani_stt_bias_file),
            bias_score=settings.gnani_stt_bias_score,
        )
    )


stt_stream_factory: Callable[[Settings], PrismaStream] = create_stt_stream


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.websocket("/ws/stt")
async def stt_proxy(websocket: WebSocket) -> None:
    """Relay browser audio to Prisma and Prisma's transcripts back to the browser.

    The browser sends binary frames of 16-bit mono PCM at the configured sample rate and may
    send the text frame "flush" to push out a trailing partial frame. It receives JSON events.
    """
    await websocket.accept()
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        await websocket.send_json({"type": "error", "message": "GNANI_API_KEY is not set"})
        await websocket.close(code=1011)
        return

    await websocket.send_json({"type": "config", "sample_rate": settings.gnani_stt_sample_rate})
    try:
        async with stt_stream_factory(settings) as stream:

            async def pump_audio() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("bytes"):
                        await stream.send_audio(message["bytes"])
                    elif message.get("text") == "flush":
                        await stream.flush()

            async def pump_events() -> None:
                async for event in stream.events():
                    await websocket.send_json(event.to_client())

            tasks = {asyncio.create_task(pump_audio()), asyncio.create_task(pump_events())}
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()
    except InvalidStatus as exc:
        status = exc.response.status_code
        await _send_error(websocket, f"Prisma rejected the connection: HTTP {status}")
    except (OSError, WebSocketException) as exc:
        await _send_error(websocket, f"Prisma connection failed: {type(exc).__name__}")
    except WebSocketDisconnect:
        return


async def _send_error(websocket: WebSocket, message: str) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close(code=1011)
