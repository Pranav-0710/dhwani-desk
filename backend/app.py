import asyncio
import contextlib
import logging
from collections.abc import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from config import Settings, get_settings
from gnani.prisma import PrismaConfig, PrismaError, PrismaStream, load_bias_words

logger = logging.getLogger("dhwani.stt")

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

    Contract (see backend/README.md): after `config` and `connected` arrive, the browser
    streams binary frames of 16-bit mono PCM at `config.sample_rate`, in real time, and may
    send the text frame "flush". It receives JSON events; `closed` or `error` followed by a
    close ends the session.
    """
    await websocket.accept()
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        await _fail(websocket, "GNANI_API_KEY is not set")
        return

    try:
        async with stt_stream_factory(settings) as stream:
            await websocket.send_json(
                {"type": "config", "sample_rate": settings.gnani_stt_sample_rate}
            )
            ended_by = await _relay(websocket, stream)
            if ended_by == "browser":
                stream.discard_pending()
        if ended_by == "prisma":
            await _send_and_close(
                websocket,
                {
                    "type": "closed",
                    "by": "prisma",
                    "code": stream.close_code,
                    "reason": stream.close_reason,
                },
                code=1000,
            )
    except PrismaError as exc:
        logger.warning("STT session failed: %s", exc)
        await _fail(websocket, str(exc))
    except WebSocketDisconnect:
        return


async def _relay(websocket: WebSocket, stream: PrismaStream) -> str:
    """Pump audio up and events down until one side ends; returns which side ended."""

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

    audio = asyncio.create_task(pump_audio())
    events = asyncio.create_task(pump_events())
    try:
        done, _ = await asyncio.wait({audio, events}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        return "browser" if audio in done else "prisma"
    finally:
        for task in (audio, events):
            task.cancel()
        await asyncio.gather(audio, events, return_exceptions=True)


async def _send_and_close(websocket: WebSocket, payload: dict, code: int) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json(payload)
        await websocket.close(code=code)


async def _fail(websocket: WebSocket, message: str) -> None:
    await _send_and_close(websocket, {"type": "error", "message": message}, code=1011)
