from typing import List

from pydantic import BaseModel

from enums.connector_type import ConnectorType


class ConnectorsResponse(BaseModel):
    connectors: List[ConnectorType]
