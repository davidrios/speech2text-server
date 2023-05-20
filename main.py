import asyncio
import logging
import os
import tempfile
import typing as t
from asyncio import CancelledError, Queue, get_running_loop
from collections import Counter
from copy import deepcopy
from enum import Enum
from io import StringIO
from pathlib import Path
from queue import Empty, SimpleQueue

import aiofiles
import ffmpeg  # type:ignore
import openai
import yaml
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "WARN"))
logging.basicConfig()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionError(Exception):
    def __init__(
        self,
        *args: object,
        message: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(*args)
        self.message = message
        self.original_exception = original_exception


TranscriptionStreamItem = str | float | None | TranscriptionError
"""
The item type defines what the partial result is:

- `str`: A transcription snippet.
- `float`: The current progress in percentage.
- `None`: The process finished, no more results are coming.
- `TranscriptionError`: The transcription process failed.
"""


class TranscriberBase:
    def transcribe(
        self, audio_file: str
    ) -> t.AsyncGenerator[TranscriptionStreamItem, None]:
        """
        Implement a generator that streams the results of the transcription.
        """
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class TranscriptionProcessor:
    """
    This class is designed to transcribe audio files using the provided `TranscriberBase`-derived
    class. This class takes an audio file path as input and uses multiple instances of the
    provided transcriber to run the transcription process concurrently.

    The class constructor takes the following parameters:

        `transcriber_factory`: A class that derives from `TranscriberBase` and provides the
            `transcribe` method to perform the transcription. Multiple instances of this class
            are created to concurrently run the transcription process.

        `num_instances`: An optional parameter that specifies the number of transcriber
            instances to create.

    The `process` method takes an audio file path and returns an `asyncio.Queue` that the user
    can use to read the results of the transcription process as they are generated.

    The `_process_queue` method runs in a separate `asyncio` task for each transcriber instance
    and dequeues an audio file from the queue, gets an available transcriber instance, and
    starts the transcription process using the transcribe method of the transcriber
    instance. It then stores the results in the appropriate queue and puts the transcriber
    instance back in the pool.

    The `start` method starts the transcription process by creating a task for each transcriber
    instance to run the `_process_queue` method, and the stop method cancels all running tasks.
    """

    def __init__(
        self,
        transcriber_factory: type[TranscriberBase],
        num_instances: int = 1,
    ):
        self.transcriber_factory = transcriber_factory
        self.num_instances = num_instances
        self.transcribers: list[TranscriberBase] = []

        self.is_running = False
        self.is_flushing = False
        self.processing_queue: Queue[tuple[int, str]] = Queue(num_instances)
        self.result_queues: dict[int, Queue[TranscriptionStreamItem]] = {}
        self.cleanup_queue: Queue[int] = Queue()
        self.cleanup_delayed: Counter[int] = Counter()
        self.to_stop: set[int] = set()
        self.last_job_id = 0
        self.running_tasks = []

    async def process(self, audio_file: str):
        if not self.is_running:
            raise Exception("processor is not running.")

        self.last_job_id += 1
        job_id = self.last_job_id
        result_queue: Queue[TranscriptionStreamItem] = Queue()
        self.result_queues[job_id] = result_queue
        await self.processing_queue.put((job_id, audio_file))
        log.debug("%s: put job in processing queue", job_id)

        async def stop_fn():
            log.debug("%s: requested to stop", job_id)
            self.to_stop.add(job_id)
            await self._schedule_cleanup(job_id)

        return result_queue, stop_fn

    async def _process_queue(self):
        while True:
            try:
                job_id, audio_file = await self.processing_queue.get()
                transcriber = self.transcribers.pop()
                result_queue = self.result_queues[job_id]
                try:
                    async for stream_item in transcriber.transcribe(audio_file):
                        if job_id in self.to_stop:
                            transcriber.stop()
                            log.debug("%s: stopped processing", job_id)
                            break

                        log.debug("%s: put transcribe result in result queue", job_id)
                        await result_queue.put(stream_item)
                    await result_queue.put(None)
                except Exception as ex:
                    log.exception("%s: error while processing job", job_id)
                    await result_queue.put(TranscriptionError(original_exception=ex))

                await self._schedule_cleanup(job_id)
                self.transcribers.append(transcriber)
                self.processing_queue.task_done()
            except CancelledError:
                log.debug("process worker cancelled, stopping")
                await self._flush_result_queues()
                break
            except Exception as ex:
                log.exception("process worker crashed, exiting...")

                await self._flush_result_queues(
                    stop_after=not isinstance(ex, CancelledError)
                )
                break

    async def _flush_result_queues(self, stop_after: bool = False):
        if self.is_flushing:
            return

        self.is_flushing = True
        for result_queue in self.result_queues.values():
            await result_queue.put(TranscriptionError(message="worker stopped"))

        log.debug("flushed result queues")

        if stop_after:
            self.stop()

    async def _schedule_cleanup(self, job_id: int, delay: float = 0):
        log.debug("%s: scheduled cleanup", job_id)
        if delay > 0:
            await asyncio.sleep(delay)
        await self.cleanup_queue.put(job_id)

    async def _process_cleanup(self):
        while True:
            try:
                job_id = await self.cleanup_queue.get()
                try:
                    result_queue = self.result_queues[job_id]
                except KeyError:
                    continue

                if result_queue.empty() or job_id in self.to_stop:
                    del self.result_queues[job_id]
                    try:
                        self.to_stop.remove(job_id)
                    except KeyError:
                        pass
                    log.debug("%s: cleaned up", job_id)
                    continue

                self.cleanup_delayed[job_id] += 1
                delay = self.cleanup_delayed[job_id] ** 4
                asyncio.create_task(self._schedule_cleanup(job_id, delay))
                level = (
                    logging.DEBUG
                    if self.cleanup_delayed[job_id] < 4
                    else logging.WARNING
                )
                log.log(
                    level,
                    "%s: result queue not empty, trying to cleanup again in %s seconds",
                    job_id,
                    delay,
                )
            except CancelledError:
                log.debug("Cleanup worker cancelled.")
                break
            except Exception:
                log.exception("Cleanup worker crashed!")
                self.stop()
                break

    def start(self):
        self.transcribers = [
            self.transcriber_factory() for _ in range(self.num_instances)
        ]
        self.running_tasks = [
            asyncio.create_task(self._process_queue())
            for _ in range(self.num_instances)
        ]
        self.running_tasks.extend(
            [
                asyncio.create_task(self._process_cleanup())
                for _ in range(self.num_instances)
            ]
        )
        self.is_running = True
        log.debug("TranscriptionProcessor started")

    def stop(self):
        self.is_running = False

        while len(self.running_tasks) > 0:
            self.running_tasks.pop().cancel()

        log.debug("stopping transcribers")
        while len(self.transcribers):
            self.transcribers.pop().stop()

        log.debug("TranscriptionProcessor stopped")


class ConvertedAudio:
    def __init__(self, original_audio: str):
        self.original_audio = original_audio

    def convert_file(self):
        out_file = str(self.tmpdir.name / Path("out.wav"))
        ffmpeg.input(self.original_audio).output(  # type: ignore
            out_file, acodec="pcm_s16le", ac=1, ar=16000
        ).run()  # type: ignore
        return out_file

    async def __aenter__(self) -> str:
        self.tmpdir = tempfile.TemporaryDirectory()
        loop = get_running_loop()
        return await loop.run_in_executor(None, self.convert_file)

    async def __aexit__(self, exc_type: t.Any, exc_val: t.Any, exc_tb: t.Any):
        self.tmpdir.cleanup()


ENGINE = os.environ.get("ENGINE")
if ENGINE == "openai":

    class OpenAITranscriber(TranscriberBase):
        async def transcribe(
            self, audio_file: str
        ) -> t.AsyncGenerator[TranscriptionStreamItem, None]:
            with open(audio_file, "rb") as fp:
                yield await openai.Audio.atranscribe(  # type:ignore
                    "whisper-1", fp, response_format="text", language="en"
                )
                yield 1.0

        def stop(self):
            return

    transcribe_processor = TranscriptionProcessor(OpenAITranscriber, 1)
elif ENGINE == "whisper":
    import whisper

    class WhisperTranscriber(TranscriberBase):
        def __init__(self):
            self._whisper = whisper.load_model(os.environ["MODEL"])

        async def transcribe(
            self, audio_file: str
        ) -> t.AsyncGenerator[TranscriptionStreamItem, None]:
            loop = get_running_loop()
            result = await loop.run_in_executor(
                None, self._whisper.transcribe, audio_file
            )
            yield result["text"]
            yield 1.0

        def stop(self):
            return

    transcribe_processor = TranscriptionProcessor(WhisperTranscriber, 1)
else:
    from whispercpp import Whisper, api

    class WhisperCppTranscriber(TranscriberBase):
        def __init__(self):
            self._queue: SimpleQueue[TranscriptionStreamItem] = SimpleQueue()
            self._transcript: list[str] = []
            num_threads = int(os.environ.get("WHISPER_CPP_THREADS") or -1)
            if num_threads == -1:
                num_threads = os.cpu_count() or 4

            params = (
                api.Params.from_enum(api.SAMPLING_GREEDY)
                .with_print_progress(False)
                .with_print_realtime(False)
                .with_suppress_blank(True)
                .with_num_threads(num_threads)
                .build()
            )

            self._transcript: list[str] = []
            params.on_new_segment(self.on_new_segment, self._transcript)

            self._progress: list[t.Any] = []
            params.on_progress(self.on_progress, self._progress)

            self._whisper = Whisper.from_params(os.environ["MODEL"], params)
            self._do_stop = False

        def on_new_segment(self, ctx: api.Context, n_new: int, data: list[str]):
            segment = ctx.full_n_segments() - n_new
            while segment < ctx.full_n_segments():
                segment_text = ctx.full_get_segment_text(segment)
                self._queue.put(segment_text)
                segment += 1

        def on_progress(self, ctx: api.Context, progress: int, data: t.Any):
            self._queue.put(min(progress, 100) / 100)

        async def transcribe(
            self, audio_file: str
        ) -> t.AsyncGenerator[TranscriptionStreamItem, None]:
            self._do_stop = False

            async with ConvertedAudio(audio_file) as fname:
                loop = get_running_loop()
                future = loop.run_in_executor(
                    None, self._whisper.transcribe_from_file, fname
                )
                while True:
                    if self._do_stop:
                        future.cancel()
                        break

                    try:
                        yield self._queue.get_nowait()
                        if future.done() and self._queue.empty():
                            break
                    except Empty:
                        await asyncio.sleep(0.1)
                await future

        def stop(self):
            self._do_stop = True
            # self._whisper
            return

    transcribe_processor = TranscriptionProcessor(
        WhisperCppTranscriber, int(os.environ.get("WHISPER_CPP_INSTANCES") or 1)
    )


@app.on_event("startup")
async def startup_event():
    transcribe_processor.start()


@app.on_event("shutdown")
async def shutdown_event():
    transcribe_processor.stop()


class MessageType(Enum):
    TRANSCRIBE = "TRANSCRIBE"
    PROCEED_WITH_FILE = "PROCEED_WITH_FILE"
    TRANSCRIPT = "TRANSCRIPT"
    TRANSCRIPTION_ERROR = "TRANSCRIPTION_ERROR"


@app.get("/")
def read_root():
    return {"hello": "world"}


async def process_transcribe_message(audio_data: bytes) -> t.Any:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input_file"
        async with aiofiles.open(input_file, "wb") as afp:
            await afp.write(audio_data)

        queue, _ = await transcribe_processor.process(str(input_file))
        progress = 0
        content = StringIO()
        while True:
            result = await queue.get()
            if result is None:
                break

            if isinstance(result, float):
                progress = int(result * 100)
                log.debug("progress %s%%", progress)
            elif isinstance(result, TranscriptionError):
                return {
                    "type": MessageType.TRANSCRIPTION_ERROR.value,
                    "content": str(result.original_exception),
                }
            else:
                content.write(result)

        content.seek(0)
        return {
            "type": MessageType.TRANSCRIPT.value,
            "content": content.read().strip(),
        }


def get_profiles_config() -> dict[str, t.Any]:
    with open(os.environ["USE_PROFILES"]) as fp:
        return yaml.load(fp, yaml.SafeLoader)


async def process_by_profile(text: str, profile: str | None) -> str:
    config = get_profiles_config()
    profile_config = config["profiles"].get(
        profile, config["profiles"][config["default"]]
    )
    if profile_config["type"] == "openai":
        params = deepcopy(profile_config["params"])
        params["prompt"] += text
        res = await openai.Completion.acreate(**params)
        if len(res["choices"]) > 0:
            text = res["choices"][0]["text"].strip()
    return text


@app.websocket("/api/transcribe")
async def api_transcribe(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            try:
                msg_type = MessageType[message["type"]]
            except KeyError:
                log.debug("invalid message")
                continue

            if msg_type is MessageType.TRANSCRIBE:
                try:
                    await websocket.send_json(
                        {"type": MessageType.PROCEED_WITH_FILE.value}
                    )
                    response = await process_transcribe_message(
                        await websocket.receive_bytes()
                    )
                except:
                    log.exception("error decoding transcribe message")
                    continue

                if response["type"] == MessageType.TRANSCRIPT.value:
                    response["content"] = await process_by_profile(
                        response["content"], message.get("profile")
                    )

                await websocket.send_json(response)
            else:
                log.debug("invalid message")
                continue
    finally:
        pass


@app.get("/api/profile")
def get_profiles():
    return get_profiles_config()


if os.environ.get("USE_GRADIO"):
    import gradio as gr

    use_async_iter = True
    try:
        from gradio.utils import SyncToAsyncIterator
    except ImportError:
        log.warn("** Your Gradio version doesn't support streaming responses. **")
        use_async_iter = False

    async def gradio_output_iterator(
        audio_mic: t.Any,
        audio_upload: t.Any,
    ) -> t.AsyncGenerator[str, None]:
        stop_fns: list[t.Callable[..., t.Coroutine[t.Any, t.Any, None]]] = []
        try:
            output = ""
            output += "** From microphone **\n"
            progress = 0

            async def process_queue(queue: Queue[TranscriptionStreamItem]) -> bool:
                nonlocal output
                nonlocal progress

                result = await queue.get()
                if result is None:
                    return False

                if isinstance(result, float):
                    progress = int(result * 100)
                elif isinstance(result, TranscriptionError):
                    output += (
                        "*** Error processing the request: "
                        + str(result.original_exception)
                        + "***"
                    )
                    return False
                else:
                    output += result

                return True

            yield output
            if audio_mic is None:
                output += "** No audio **"
                yield output
            else:
                queue, stop_fn = await transcribe_processor.process(audio_mic)
                stop_fns.append(stop_fn)
                while True:
                    if not await process_queue(queue):
                        break

                    yield output + f"\n\n(Progress: {progress}%)"

            output += "\n\n"
            yield output

            output += "** From upload **\n"
            yield output
            if audio_upload is None:
                output += "** No audio **"
                yield output
            else:
                progress = 0
                queue, stop_fn = await transcribe_processor.process(audio_upload)
                log.debug("*************** %s %s", queue, stop_fn)
                stop_fns.append(stop_fn)
                while True:
                    if not await process_queue(queue):
                        break

                    yield output + f"\n\n(Progress: {progress}%)"

            progress = 100
            yield output + f"\n\n(Progress: {progress}%)"
        finally:
            log.debug("################ %s", stop_fns)
            while len(stop_fns) > 0:
                await stop_fns.pop()()
            log.debug("++++++++++++++ gradio output finished")

    # typing here causes issues with gradio
    async def gradio_output(audio_mic, audio_upload):  # type: ignore
        res = ""
        async for res in gradio_output_iterator(audio_mic, audio_upload):
            continue
        return res

    demo = gr.Interface(
        gradio_output_iterator if use_async_iter else gradio_output,
        inputs=[
            gr.Audio(source="microphone", label="", type="filepath"),
            gr.Audio(source="upload", label="", type="filepath"),
        ],
        outputs=gr.Text(label="Transcription"),
        allow_flagging="never",
    )

    app = gr.mount_gradio_app(app, demo.queue(concurrency_count=10), path="/gradio")
