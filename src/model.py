from .base_model import BaseModel

class Model(BaseModel):
  def __init__(self):
    super().__init__()
    print('Model')