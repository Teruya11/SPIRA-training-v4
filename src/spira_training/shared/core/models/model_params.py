from pydantic import BaseModel
from typing_extensions import TypeVar


class ModelParams(BaseModel): ...


ModelParamsT = TypeVar("ModelParamsT", bound=ModelParams)
