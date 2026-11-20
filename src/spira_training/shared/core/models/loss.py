from pydantic import BaseModel


class Loss(BaseModel):
    value: list[float] | float
