from typing import Optional

from fastapi import HTTPException, status
from fastapi.responses import RedirectResponse

from api.v1.schema.connector import ConnectorsResponse
from db.prisma.generated.enums import ConnectorType as PrismaConnectorType
from db.prisma.utils import get_db
from enums.connector_type import ConnectorType


async def upsert_connector_of_user(
    user_id: str,
    connector_id: Optional[str],
    connector_type: Optional[str],
    current_uri: Optional[str],
) -> RedirectResponse:
    if not connector_id or not connector_type or not current_uri:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Something went wrong"
        )

    db = await get_db()

    await db.connector.upsert(
        where={"connector_id": connector_id},
        data={
            "create": {
                "connector_id": connector_id,
                "connector_type": PrismaConnectorType(connector_type),
                "userId": user_id,
            },
            "update": {},
        },
    )

    return RedirectResponse(current_uri, status_code=status.HTTP_302_FOUND)


async def get_connectors_of_user(user_id: str) -> ConnectorsResponse:
    db = await get_db()

    connectors = await db.connector.find_many(
        where={"userId": user_id}, order={"createdAt": "asc"}
    )

    return ConnectorsResponse(
        connectors=[ConnectorType(connector.connector_type) for connector in connectors]
    )
