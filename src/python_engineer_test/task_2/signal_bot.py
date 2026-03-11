"""Signal Bot Implementation for ATAK Integration.

This module provides a Signal bot that can:
1. Receive messages from Signal
2. Parse location/target information
3. Send CoT messages to ATAK

Requirements:
- signal-cli or signald running
- Signal account registered
"""

import logging
import socket
import subprocess
from typing import Any

from python_engineer_test.preflight import require_command

from .cot_protocol import format_cot_message, parse_signal_message

logger = logging.getLogger(__name__)


class SignalBot:
    """Signal bot for ATAK integration."""

    def __init__(
        self,
        phone_number: str,
        atak_host: str = "239.2.3.1",
        atak_port: int = 6969,
        signal_cli_path: str = "signal-cli",
        order: str = "lonlat",
    ):
        """Initialize Signal bot.

        Args:
            phone_number: Signal phone number (with country code)
            atak_host: ATAK multicast address
            atak_port: ATAK port
            signal_cli_path: Path to signal-cli binary
        """
        self.phone_number = phone_number
        self.atak_host = atak_host
        self.atak_port = atak_port
        self.signal_cli_path = signal_cli_path
        self.order = order

    def send_to_atak(self, cot_message: str) -> bool:
        """Send CoT message to ATAK via UDP multicast.

        Args:
            cot_message: CoT XML message

        Returns:
            True if sent successfully
        """
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.sendto(cot_message.encode("utf-8"), (self.atak_host, self.atak_port))
            return True
        except OSError as e:
            logger.error("Error sending to ATAK: %s", e)
            return False
        finally:
            if sock is not None:
                sock.close()

    def send_signal_message(self, recipient: str, message: str) -> bool:
        """Send a message via Signal.

        Args:
            recipient: Recipient phone number or group ID
            message: Message content

        Returns:
            True if sent successfully
        """
        try:
            result = subprocess.run(
                [
                    self.signal_cli_path,
                    "-u",
                    self.phone_number,
                    "send",
                    "-m",
                    message,
                    recipient,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error("Error sending Signal message: %s", e.stderr)
            return False

    def receive_messages(self) -> list[dict[str, Any]]:
        """Receive messages from Signal.

        Returns:
            List of message dictionaries
        """
        try:
            result = subprocess.run(
                [
                    self.signal_cli_path,
                    "-u",
                    self.phone_number,
                    "receive",
                    "--json",
                ],
                capture_output=True,
                text=True,
            )

            # Parse JSON output
            import json

            messages = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        msg = json.loads(line)
                        messages.append(msg)
                    except json.JSONDecodeError:
                        continue

            return messages
        except Exception as e:
            logger.error("Error receiving messages: %s", e)
            return []

    def process_message(self, message: str) -> str | None:
        """Process a Signal message and send to ATAK.

        Args:
            message: Raw message text

        Returns:
            CoT message if successful, None otherwise
        """
        try:
            parsed = parse_signal_message(message, order=self.order)
            cot = format_cot_message(
                latitude=parsed["latitude"],
                longitude=parsed["longitude"],
                target_description=parsed["description"],
            )

            if self.send_to_atak(cot):
                logger.info(
                    "Sent to ATAK: %s at (%s, %s)",
                    parsed["description"],
                    parsed["latitude"],
                    parsed["longitude"],
                )
                return cot

        except ValueError as e:
            logger.error("Invalid message format: %s", e)
        except Exception as e:
            logger.error("Error processing message: %s", e)

        return None

    def run(self, poll_interval: int = 5) -> None:
        """Run the bot in polling mode.

        Args:
            poll_interval: Seconds between polling for messages
        """
        import time

        logger.info("Starting Signal bot (%s)", self.phone_number)
        logger.info("Sending to ATAK at %s:%s", self.atak_host, self.atak_port)

        while True:
            try:
                messages = self.receive_messages()
                for msg in messages:
                    # Extract message text
                    envelope = msg.get("envelope", {})
                    data_message = envelope.get("dataMessage", {})
                    message_text = data_message.get("message", "")

                    if message_text:
                        self.process_message(message_text)

            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e)

            time.sleep(poll_interval)


def main(
    phone_number: str,
    atak_host: str = "239.2.3.1",
    atak_port: int = 6969,
    *,
    dry_run: str | None = None,
    order: str = "lonlat",
) -> int:
    if dry_run is not None:
        parsed = parse_signal_message(dry_run, order=order)
        cot = format_cot_message(
            latitude=parsed["latitude"],
            longitude=parsed["longitude"],
            target_description=parsed["description"],
        )
        logger.info("%s", cot)
        return 0

    signal_cli_path = require_command("signal-cli")

    bot = SignalBot(
        phone_number=phone_number,
        atak_host=atak_host,
        atak_port=atak_port,
        signal_cli_path=signal_cli_path,
        order=order,
    )

    bot.run()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Signal Bot for ATAK")
    parser.add_argument("phone", help="Signal phone number (with country code)")
    parser.add_argument("--atak-host", default="239.2.3.1", help="ATAK multicast address")
    parser.add_argument("--atak-port", type=int, default=6969, help="ATAK port")
    parser.add_argument(
        "--dry-run",
        metavar="MESSAGE",
        help="Parse one message and print CoT XML to stdout, then exit",
    )
    parser.add_argument(
        "--order",
        choices=["latlon", "lonlat"],
        default="lonlat",
        help="Coordinate order (default: lonlat, matches assignment PDF)",
    )
    args = parser.parse_args()

    try:
        raise SystemExit(
            main(
                args.phone,
                args.atak_host,
                args.atak_port,
                dry_run=args.dry_run,
                order=args.order,
            )
        )
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
