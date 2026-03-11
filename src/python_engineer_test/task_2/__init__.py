"""Task 2: Signal Bot with ATAK Integration.

Signal bot to send geolocation and target information to ATAK client via CoT protocol.
"""

from .cot_protocol import format_cot_message
from .signal_bot import SignalBot

__all__ = ["SignalBot", "format_cot_message"]
