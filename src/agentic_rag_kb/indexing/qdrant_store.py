"""Qdrant storage adapter.

TODO:
- Wrap qdrant-client collection creation, upsert, search, and payload filtering.
- Support separate child and parent collections.
- Add health checks for local development and deployment.
"""


class QdrantStore:
    """Adapter around qdrant-client for vector storage."""

    def connect(self) -> None:
        """Connect to Qdrant."""

        raise NotImplementedError("Qdrant connection will be implemented with qdrant-client.")

