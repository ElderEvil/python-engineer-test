# Task 2: Signal Bot with ATAK Integration

Signal bot that receives location messages and sends CoT (Cursor on Target) to ATAK.

## Usage

```bash
# Via CLI
pet task2 +1234567890 --atak-host 239.2.3.1 --atak-port 6969

# Smoke-friendly (no Signal/ATAK required): parse one message and print CoT XML
# Assignment PDF uses lon/lat order for the example message.
pet task2 +1234567890 --dry-run "48.567123 39.87897 tank" --order lonlat

# Direct module
python -m python_engineer_test.task_2.signal_bot +1234567890
```

## Message Format

Send messages to the bot in format:
```
<longitude> <latitude> <target_description>
```

Example:
```
48.567123 39.87897 enemy tank
```

If you need to interpret input as `lat lon ...`, use:

```bash
pet task2 +1234567890 --dry-run "39.87897 48.567123 tank" --order latlon
```

## CoT Protocol

Generates CoT XML messages with:
- Event type: `a-h-G` (hostile ground)
- Coordinate point with HAE/CE/LE
- Contact callsign and remarks

## Requirements

- signal-cli installed and configured
- Signal account registered

## Dependencies

```bash
pip install -e ".[task2]"
```

## Resources

- ATAK/TAK: https://www.civtak.org/
