from async_lru import alru_cache

from db.prisma.generated.client import Prisma

prisma = Prisma()


@alru_cache()
async def get_db():
    """
    Asynchronously retrieves a Prisma database client instance.

    Ensures that the Prisma client is connected before returning it.
    Uses async LRU cache to avoid redundant connections.
    """
    # Check if the Prisma client is connected; connect if not
    if not prisma.is_connected():
        await prisma.connect()
    # Return the connected Prisma client
    return prisma
