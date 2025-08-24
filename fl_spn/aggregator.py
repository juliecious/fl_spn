from typing import Optional

from config import FederatedEiNetConfig
from simple_einet.einet import Einet


class FederatedEiNetClient:
    """
    Federated EiNet Client using real SimpleEiNet
    Each client maintains its own EiNet model
    """

    def __init__(self, client_id: int, config: FederatedEiNetConfig):
        self.client_id = client_id
        self.config = config
        self.model: Optional[Einet] = None
        self.training_history = []
        self.client_data = None

    # def receive_data(selfself,):
