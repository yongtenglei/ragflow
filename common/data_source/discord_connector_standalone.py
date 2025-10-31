"""Discord connector - Standalone version for testing"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterable, Iterable

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
try:
    from discord import Client, MessageType
    from discord.channel import TextChannel, Thread
    from discord.flags import Intents
    from discord.message import Message as DiscordMessage
except ImportError as e:
    print(f"Failed to import discord modules: {e}")
    sys.exit(1)


# Define required classes and constants directly in this file to avoid import issues
class DocumentSource:
    DISCORD = "discord"


class Document:
    def __init__(self, id: str, source: str, semantic_identifier: str, doc_updated_at: datetime, blob: bytes):
        self.id = id
        self.source = source
        self.semantic_identifier = semantic_identifier
        self.doc_updated_at = doc_updated_at
        self.blob = blob


class TextSection:
    def __init__(self, text: str, link: str):
        self.text = text
        self.link = link


class ConnectorMissingCredentialError(Exception):
    def __init__(self, connector_name: str):
        super().__init__(f"Missing credentials for {connector_name}")


# Constants
_DISCORD_DOC_ID_PREFIX = "DISCORD_"
_SNIPPET_LENGTH = 30
INDEX_BATCH_SIZE = 16


def _convert_message_to_document(
    message: DiscordMessage,
    sections: list[TextSection],
) -> Document:
    """
    Convert a discord message to a document
    Sections are collected before calling this function because it relies on async
        calls to fetch the thread history if there is one
    """

    metadata: dict[str, str | list[str]] = {}
    semantic_substring = ""

    # Only messages from TextChannels will make it here but we have to check for it anyways
    if isinstance(message.channel, TextChannel) and (channel_name := message.channel.name):
        metadata["Channel"] = channel_name
        semantic_substring += f" in Channel: #{channel_name}"

    # If there is a thread, add more detail to the metadata, title, and semantic identifier
    if isinstance(message.channel, Thread):
        # Threads do have a title
        title = message.channel.name

        # Add more detail to the semantic identifier if available
        semantic_substring += f" in Thread: {title}"

    snippet: str = message.content[:_SNIPPET_LENGTH].rstrip() + "..." if len(message.content) > _SNIPPET_LENGTH else message.content

    semantic_identifier = f"{message.author.name} said{semantic_substring}: {snippet}"

    return Document(
        id=f"{_DISCORD_DOC_ID_PREFIX}{message.id}", source=DocumentSource.DISCORD, semantic_identifier=semantic_identifier, doc_updated_at=message.edited_at, blob=message.content.encode("utf-8")
    )


async def _fetch_filtered_channels(
    discord_client: Client,
    server_ids: list[int] | None,
    channel_names: list[str] | None,
) -> list[TextChannel]:
    filtered_channels: list[TextChannel] = []
    print(f"Searching for channels in {len(server_ids) if server_ids else 0} servers")

    for channel in discord_client.get_all_channels():
        print(f"Checking channel: {channel.name} ({channel.id}) in guild {channel.guild.name} ({channel.guild.id})")
        if not channel.permissions_for(channel.guild.me).read_message_history:
            print(f"  No read_message_history permission for {channel.name}")
            continue
        if not isinstance(channel, TextChannel):
            print(f"  Not a text channel: {channel.name}")
            continue
        if server_ids and len(server_ids) > 0 and channel.guild.id not in server_ids:
            print(f"  Guild {channel.guild.id} not in specified server IDs")
            continue
        if channel_names and channel.name not in channel_names:
            print(f"  Channel {channel.name} not in specified channel names")
            continue
        print(f"  Adding channel: {channel.name}")
        filtered_channels.append(channel)

    print(f"Found {len(filtered_channels)} channels for the authenticated user")
    return filtered_channels


async def _fetch_documents_from_channel(
    channel: TextChannel,
    start_time: datetime | None,
    end_time: datetime | None,
) -> AsyncIterable[Document]:
    # Discord's epoch starts at 2015-01-01
    discord_epoch = datetime(2015, 1, 1, tzinfo=timezone.utc)
    if start_time and start_time < discord_epoch:
        start_time = discord_epoch

    # NOTE: limit=None is the correct way to fetch all messages and threads with pagination
    # The discord package erroneously uses limit for both pagination AND number of results
    # This causes the history and archived_threads methods to return 100 results even if there are more results within the filters
    # Pagination is handled automatically (100 results at a time) when limit=None

    async for channel_message in channel.history(
        limit=None,
        after=start_time,
        before=end_time,
    ):
        # Skip messages that are not the default type
        if channel_message.type != MessageType.default:
            continue

        sections: list[TextSection] = [
            TextSection(
                text=channel_message.content,
                link=channel_message.jump_url,
            )
        ]

        yield _convert_message_to_document(channel_message, sections)

    for active_thread in channel.threads:
        async for thread_message in active_thread.history(
            limit=None,
            after=start_time,
            before=end_time,
        ):
            # Skip messages that are not the default type
            if thread_message.type != MessageType.default:
                continue

            sections = [
                TextSection(
                    text=thread_message.content,
                    link=thread_message.jump_url,
                )
            ]

            yield _convert_message_to_document(thread_message, sections)

    async for archived_thread in channel.archived_threads(
        limit=None,
    ):
        async for thread_message in archived_thread.history(
            limit=None,
            after=start_time,
            before=end_time,
        ):
            # Skip messages that are not the default type
            if thread_message.type != MessageType.default:
                continue

            sections = [
                TextSection(
                    text=thread_message.content,
                    link=thread_message.jump_url,
                )
            ]

            yield _convert_message_to_document(thread_message, sections)


def _manage_async_retrieval(
    token: str,
    requested_start_date_string: str,
    channel_names: list[str],
    server_ids: list[int],
    start: datetime | None = None,
    end: datetime | None = None,
) -> Iterable[Document]:
    # parse requested_start_date_string to datetime
    pull_date: datetime | None = datetime.strptime(requested_start_date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc) if requested_start_date_string else None

    # Set start_time to the later of start and pull_date, or whichever is provided
    start_time = max(filter(None, [start, pull_date])) if start or pull_date else None

    end_time: datetime | None = end

    async def _async_fetch() -> AsyncIterable[Document]:
        intents = Intents.default()
        intents.message_content = True
        print(f"Attempting to connect with token: {token[:10]}...{token[-10:]}")  # Print part of token for debugging

        # Check for proxy settings
        proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
        if proxy_url:
            print(f"Using proxy: {proxy_url}")
        else:
            print("No proxy configured")

        # Try to connect with proxy, with timeout handling
        try:
            import signal

            async with Client(intents=intents, proxy=proxy_url) as cli:
                print("-----------------------------------------", flush=True)
                # Store start task for later handling
                start_task = asyncio.create_task(coro=cli.start(token))
                print("-----------------------------------------", flush=True)
                await cli.wait_until_ready()
                print("connected ...", flush=True)

                filtered_channels: list[TextChannel] = await _fetch_filtered_channels(
                    discord_client=cli,
                    server_ids=server_ids,
                    channel_names=channel_names,
                )
                print("connected ...", filtered_channels, flush=True)

                for channel in filtered_channels:
                    async for doc in _fetch_documents_from_channel(
                        channel=channel,
                        start_time=start_time,
                        end_time=end_time,
                    ):
                        yield doc
        except Exception as e:
            print(f"Error during connection: {e}")
            import traceback
            traceback.print_exc()
            raise e

            filtered_channels: list[TextChannel] = await _fetch_filtered_channels(
                discord_client=cli,
                server_ids=server_ids,
                channel_names=channel_names,
            )
            print("connected ...", filtered_channels, flush=True)

            for channel in filtered_channels:
                async for doc in _fetch_documents_from_channel(
                    channel=channel,
                    start_time=start_time,
                    end_time=end_time,
                ):
                    yield doc

    def run_and_yield() -> Iterable[Document]:
        loop = asyncio.new_event_loop()
        try:
            # Get the async generator
            async_gen = _async_fetch()
            # Convert to AsyncIterator
            async_iter = async_gen.__aiter__()
            while True:
                try:
                    # Create a coroutine by calling anext with the async iterator
                    next_coro = anext(async_iter)
                    # Run the coroutine to get the next document
                    doc = loop.run_until_complete(next_coro)
                    yield doc
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    return run_and_yield()


class DiscordConnector:
    """Discord connector for accessing Discord messages and channels"""

    def __init__(
        self,
        server_ids: list[str] = [],
        channel_names: list[str] = [],
        # YYYY-MM-DD
        start_date: str | None = None,
        batch_size: int = INDEX_BATCH_SIZE,
    ):
        self.batch_size = batch_size
        self.channel_names: list[str] = channel_names if channel_names else []
        self.server_ids: list[int] = [int(server_id) for server_id in server_ids] if server_ids else []
        self._discord_bot_token: str | None = None
        self.requested_start_date_string: str = start_date or ""

    @property
    def discord_bot_token(self) -> str:
        if self._discord_bot_token is None:
            raise ConnectorMissingCredentialError("Discord")
        return self._discord_bot_token

    def _manage_doc_batching(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterable[list[Document]]:
        doc_batch = []
        for doc in _manage_async_retrieval(
            token=self.discord_bot_token,
            requested_start_date_string=self.requested_start_date_string,
            channel_names=self.channel_names,
            server_ids=self.server_ids,
            start=start,
            end=end,
        ):
            doc_batch.append(doc)
            if len(doc_batch) >= self.batch_size:
                yield doc_batch
                doc_batch = []

        if doc_batch:
            yield doc_batch

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self._discord_bot_token = credentials["discord_bot_token"]
        return None

    def poll_source(self, start: float, end: float) -> Any:
        """Poll Discord for recent messages"""
        return self._manage_doc_batching(
            datetime.fromtimestamp(start, tz=timezone.utc),
            datetime.fromtimestamp(end, tz=timezone.utc),
        )

    def load_from_state(self) -> Any:
        """Load messages from Discord state"""
        return self._manage_doc_batching(None, None)


if __name__ == "__main__":
    end = time.time()
    # 1 day for testing
    start = end - 24 * 60 * 60 * 1
    # "1,2,3"
    server_ids: str | None = os.environ.get("server_ids", None)
    # "channel1,channel2"
    channel_names: str | None = os.environ.get("channel_names", None)

    connector = DiscordConnector(
        server_ids=server_ids.split(",") if server_ids else [],
        channel_names=channel_names.split(",") if channel_names else [],
        start_date=os.environ.get("start_date", None),
    )
    connector.load_credentials({"discord_bot_token": os.environ.get("discord_bot_token")})

    # Try to poll source and print documents
    try:
        print("Starting to poll Discord source...")
        doc_count = 0
        for doc_batch in connector.poll_source(start, end):
            for doc in doc_batch:
                doc_count += 1
                print(f"Document #{doc_count}")
                print(f"Document ID: {doc.id}")
                print(f"Semantic Identifier: {doc.semantic_identifier}")
                print(f"Content: {doc.blob.decode('utf-8')[:100]}...")
                print("---")

                # Limit output for testing
                if doc_count >= 5:
                    print("Reached document limit for testing")
                    break
            if doc_count >= 5:
                break

        if doc_count == 0:
            print("No documents found in the specified time range")
    except ConnectorMissingCredentialError as e:
        print(f"Error: {e}")
        print("Please set the 'discord_bot_token' environment variable.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
